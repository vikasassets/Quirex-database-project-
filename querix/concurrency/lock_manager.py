"""
Lock Manager for Querix Storage Engine.
Implements Row-Level Locking, Shared (S) and Exclusive (X) modes,
Strict Two-Phase Locking (Strict 2PL), and Wait-Die Deadlock Prevention.
"""

from __future__ import annotations

import threading
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from querix.concurrency.transaction import Transaction


class LockMode(IntEnum):
    """Supported lock modes for row-level concurrency."""
    SHARED = 1      # S Lock (Read)
    EXCLUSIVE = 2   # X Lock (Write)

    def __str__(self) -> str:
        return "SHARED" if self == LockMode.SHARED else "EXCLUSIVE"


class LockError(Exception):
    """Base exception for concurrency and locking violations."""
    pass


class TransactionAbortError(LockError):
    """Raised when a transaction must abort (e.g., deadlock prevention)."""
    pass


class DeadlockError(TransactionAbortError):
    """Raised when a deadlock cycle or Wait-Die prevention aborts a transaction."""
    pass


class Strict2PLError(LockError):
    """Raised when Strict Two-Phase Locking rules are violated."""
    pass


class InvalidTransactionStateError(LockError):
    """Raised when an operation is invalid for the transaction's current state."""
    pass


def _normalize_res(res: Any) -> Any:
    """Normalizes resource identifier (e.g., converts str to bytes)."""
    if isinstance(res, str):
        return res.encode("utf-8")
    return res


class LockManager:
    """
    Thread-safe row-level Lock Manager.

    Guarantees:
    - Strict 2PL: Locks are acquired in Growing phase and held until commit/abort.
    - S/X Compatibility: S locks can be shared; X locks are mutually exclusive.
    - Lock Upgrades: Upgrading an S lock to an X lock is supported.
    - Deadlock Handling: Wait-Die deadlock prevention prevents engine freezes
      when transactions are waiting on each other.
    """

    def __init__(self, strict_wait_die: bool = False) -> None:
        """
        :param strict_wait_die: If True, younger transactions immediately die whenever
                                requesting a lock held by an older transaction.
                                If False (default), transactions wait in queue unless
                                mutual waiting / cycle occurs, at which point the younger dies.
        """
        self.strict_wait_die = strict_wait_die
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # resource_id -> {txn_id: LockMode}
        self._holders: Dict[Any, Dict[int, LockMode]] = {}

        # resource_id -> [(Transaction, LockMode)]
        self._waiters: Dict[Any, List[Tuple[Transaction, LockMode]]] = {}

        # txn_id -> set of holder txn_ids currently waited on
        self._waiting_for: Dict[int, Set[int]] = {}

        # txn_id -> Transaction object
        self._txn_objects: Dict[int, Transaction] = {}

        # Set of txn_ids that were aborted by deadlock resolution
        self._aborted_txns: Set[int] = set()

        # Cumulative count of times any lock request was queued to wait
        self._wait_count: int = 0

    @property
    def wait_count(self) -> int:
        """Returns the total number of times requests entered the wait queue."""
        with self._lock:
            return self._wait_count

    def has_waiters(self, resource_id: Any) -> bool:
        """Checks if there are transactions actively waiting for a resource."""
        res = _normalize_res(resource_id)
        with self._lock:
            return bool(self._waiters.get(res))

    def get_waiters(self, resource_id: Any) -> List[int]:
        """Returns the list of transaction IDs currently waiting for a resource."""
        res = _normalize_res(resource_id)
        with self._lock:
            return [txn.txn_id for txn, _ in self._waiters.get(res, [])]

    def holders_of(self, resource_id: Any) -> Dict[int, LockMode]:
        """Returns a copy of the current holders dictionary for a resource."""
        res = _normalize_res(resource_id)
        with self._lock:
            return dict(self._holders.get(res, {}))

    def acquire(
        self,
        txn: Transaction,
        resource_id: Any,
        mode: LockMode,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquires a row-level lock on `resource_id` in mode `mode` for transaction `txn`.
        Enforces Strict 2PL and Wait-Die deadlock prevention.

        :param txn: The requesting Transaction.
        :param resource_id: Key or slot identifier.
        :param mode: LockMode.SHARED or LockMode.EXCLUSIVE.
        :param timeout: Optional timeout in seconds to wait before returning False.
        :return: True if lock is granted.
        :raises InvalidTransactionStateError: If txn is not in GROWING state.
        :raises DeadlockError: If Wait-Die detects deadlock or aborts transaction.
        """
        # Strict 2PL check: must be in GROWING phase
        from querix.concurrency.transaction import TransactionState

        if txn.state != TransactionState.GROWING:
            raise InvalidTransactionStateError(
                f"Strict 2PL violation: Cannot acquire lock in state {txn.state.name}. "
                "Locks can only be acquired during the Growing phase."
            )

        res = _normalize_res(resource_id)

        with self._lock:
            self._txn_objects[txn.txn_id] = txn

            if txn.txn_id in self._aborted_txns:
                self._aborted_txns.discard(txn.txn_id)
                raise DeadlockError(
                    f"Transaction {txn.txn_id} was aborted due to deadlock."
                )

            # Check if lock can be granted immediately
            if self._can_grant_now(txn, res, mode):
                self._grant(txn, res, mode)
                return True

            # Incompatible: examine conflicting holders
            holders = self._holders.get(res, {})
            conflicting_holders = [tid for tid in holders if tid != txn.txn_id]

            # Deadlock Handling / Wait-Die Strategy:
            # 1. Strict Wait-Die mode
            if self.strict_wait_die:
                for hid in conflicting_holders:
                    if txn.txn_id > hid:
                        # Younger transaction requests lock held by older transaction -> DIE!
                        raise DeadlockError(
                            f"Strict Wait-Die: Younger transaction {txn.txn_id} died requesting "
                            f"lock held by older transaction {hid}."
                        )

            # 2. Wait-Die Cycle Prevention (Mutual Waiting)
            # Check if any holder is directly or indirectly waiting for txn
            for hid in conflicting_holders:
                if self._reaches(hid, txn.txn_id, set()):
                    # Mutual waiting detected (Deadlock cycle)!
                    if txn.txn_id > hid:
                        # Younger transaction dies to break deadlock
                        raise DeadlockError(
                            f"Wait-Die: Younger transaction {txn.txn_id} died waiting for "
                            f"older holder {hid}."
                        )
                    else:
                        # txn is older -> younger holder hid must die!
                        self._abort_holder_due_to_deadlock(hid)

            # Transaction is queued
            self._wait_count += 1
            if res not in self._waiters:
                self._waiters[res] = []
            wait_entry = (txn, mode)
            self._waiters[res].append(wait_entry)

            if txn.txn_id not in self._waiting_for:
                self._waiting_for[txn.txn_id] = set()
            for hid in conflicting_holders:
                self._waiting_for[txn.txn_id].add(hid)

            start_time = time.time()
            try:
                while not self._can_grant_now(txn, res, mode):
                    if txn.txn_id in self._aborted_txns or txn.state == TransactionState.ABORTED:
                        self._aborted_txns.discard(txn.txn_id)
                        raise DeadlockError(
                            f"Transaction {txn.txn_id} was aborted due to deadlock while waiting for lock."
                        )

                    if timeout is not None:
                        elapsed = time.time() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            return False
                        self._cv.wait(remaining)
                    else:
                        self._cv.wait()

                    if txn.txn_id in self._aborted_txns or txn.state == TransactionState.ABORTED:
                        self._aborted_txns.discard(txn.txn_id)
                        raise DeadlockError(
                            f"Transaction {txn.txn_id} was aborted due to deadlock while waiting for lock."
                        )

                # Granted after unblocking!
                self._grant(txn, res, mode)
                return True

            finally:
                # Clean up waiter queue entry
                if res in self._waiters and wait_entry in self._waiters[res]:
                    self._waiters[res].remove(wait_entry)
                    if not self._waiters[res]:
                        del self._waiters[res]
                self._waiting_for.pop(txn.txn_id, None)

    def acquire_shared(
        self,
        txn: Transaction,
        resource_id: Any,
        timeout: Optional[float] = None,
    ) -> bool:
        """Convenience method to acquire a Shared (S) lock."""
        return self.acquire(txn, resource_id, LockMode.SHARED, timeout=timeout)

    def acquire_exclusive(
        self,
        txn: Transaction,
        resource_id: Any,
        timeout: Optional[float] = None,
    ) -> bool:
        """Convenience method to acquire an Exclusive (X) lock."""
        return self.acquire(txn, resource_id, LockMode.EXCLUSIVE, timeout=timeout)

    def release_all(self, txn: Transaction) -> None:
        """
        Releases all locks held by transaction `txn`.
        Called strictly during transaction commit() or abort().
        """
        with self._lock:
            tid = txn.txn_id
            for res in list(txn.locks_held.keys()):
                if res in self._holders and tid in self._holders[res]:
                    del self._holders[res][tid]
                    if not self._holders[res]:
                        del self._holders[res]

            txn._clear_locks()
            self._waiting_for.pop(tid, None)
            self._txn_objects.pop(tid, None)
            self._aborted_txns.discard(tid)

            # Wake up all waiting threads so eligible ones can proceed
            self._cv.notify_all()

    def release(self, txn: Transaction, resource_id: Any) -> None:
        """
        Individual lock release is strictly prohibited under Strict 2PL.
        """
        raise Strict2PLError(
            "Strict 2PL violation: Cannot release locks individually during transaction execution. "
            "All locks are released simultaneously upon commit() or abort()."
        )

    def _can_grant_now(self, txn: Transaction, res: Any, mode: LockMode) -> bool:
        """Checks whether the requested lock can be granted immediately."""
        holders = self._holders.get(res, {})
        if not holders:
            return True

        if txn.txn_id in holders:
            current_mode = holders[txn.txn_id]
            if current_mode == LockMode.EXCLUSIVE:
                # Already holds X lock, sufficient for both S and X
                return True
            if current_mode == LockMode.SHARED:
                if mode == LockMode.SHARED:
                    return True
                if mode == LockMode.EXCLUSIVE:
                    # Upgrade: grant if txn is the only holder
                    return len(holders) == 1
        else:
            if mode == LockMode.SHARED:
                # Can grant S if all current holders hold S
                return all(m == LockMode.SHARED for m in holders.values())
            return False

        return False

    def _grant(self, txn: Transaction, res: Any, mode: LockMode) -> None:
        """Assigns the lock and records it in both the LockManager and Transaction."""
        if res not in self._holders:
            self._holders[res] = {}
        self._holders[res][txn.txn_id] = mode
        txn._record_lock(res, mode)

    def _reaches(self, curr_tid: int, target_tid: int, visited: Set[int]) -> bool:
        """
        DFS traversal on wait-for graph: checks if curr_tid waits for target_tid
        directly or indirectly.
        """
        if curr_tid == target_tid:
            return True
        if curr_tid in visited:
            return False
        visited.add(curr_tid)
        for next_tid in self._waiting_for.get(curr_tid, set()):
            if self._reaches(next_tid, target_tid, visited):
                return True
        return False

    def _abort_holder_due_to_deadlock(self, victim_id: int) -> None:
        """Aborts a younger holder transaction to break a mutual wait cycle."""
        self._aborted_txns.add(victim_id)
        victim_txn = self._txn_objects.get(victim_id)
        if victim_txn is not None:
            try:
                victim_txn.abort()
            except Exception:
                pass
        self._cv.notify_all()
