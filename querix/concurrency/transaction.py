"""
Transaction Context and Lifecycle Management for Querix Storage Engine.
Implements ACID transaction boundaries, Strict 2PL state transitions
(Growing -> Shrinking -> Committed/Aborted), WAL commit persistence,
and ARIES Undo rollback on abort.
"""

from __future__ import annotations

import itertools
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from querix.concurrency.lock_manager import (
    InvalidTransactionStateError,
    LockManager,
    LockMode,
)
from querix.wal.log_manager import LogManager

if TYPE_CHECKING:
    from querix.engine import QuerixEngine
    from querix.wal.recovery import RecoveryManager


class TransactionState(Enum):
    """
    Strict Two-Phase Locking (Strict 2PL) lifecycle states.
    - GROWING: Transaction is active and acquiring locks. Cannot release locks.
    - SHRINKING: Transaction is terminating and releasing all held locks. Cannot acquire locks.
    - COMMITTED: Transaction is durable (WAL COMMIT record fsynced) and all locks released.
    - ABORTED: Transaction changes rolled back via WAL Undo and all locks released.
    """
    GROWING = "GROWING"
    SHRINKING = "SHRINKING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"

    def __str__(self) -> str:
        return self.value


# Global atomic transaction ID generator
_TXN_COUNTER = itertools.count(1)
_TXN_COUNTER_LOCK = threading.Lock()


def generate_txn_id() -> int:
    """Generates a globally unique, strictly ascending transaction ID."""
    with _TXN_COUNTER_LOCK:
        return next(_TXN_COUNTER)


class Transaction:
    """
    Encapsulates the context, state, and lifecycle of an ACID transaction.
    """

    def __init__(
        self,
        txn_id: Optional[int] = None,
        lock_manager: Optional[LockManager] = None,
        log_manager: Optional[LogManager] = None,
        recovery_manager: Optional[RecoveryManager] = None,
        engine: Optional[QuerixEngine] = None,
    ) -> None:
        """
        :param txn_id: Unique integer transaction identifier. Generated if None.
        :param lock_manager: The engine LockManager managing concurrency.
        :param log_manager: The Write-Ahead Log manager.
        :param recovery_manager: The crash and rollback recovery manager.
        :param engine: The parent QuerixEngine instance.
        """
        self.txn_id = txn_id if txn_id is not None else generate_txn_id()
        self.state = TransactionState.GROWING
        self.lock_manager = lock_manager
        self.log_manager = log_manager
        self.recovery_manager = recovery_manager
        self.engine = engine

        # Dictionary of locks currently held: resource_id -> LockMode
        self.locks_held: Dict[Any, LockMode] = {}

        # Log Sequence Number of the last WAL record written by this transaction
        self.prev_lsn: int = 0

        # Savepoints for key-level rollback on abort: (op, key_bytes, old_entry)
        self.savepoints: List[Tuple[str, bytes, Optional[Any]]] = []

        self._lock = threading.RLock()

    @property
    def is_active(self) -> bool:
        """Returns True if transaction is actively running in the Growing phase."""
        return self.state == TransactionState.GROWING

    @property
    def is_committed(self) -> bool:
        """Returns True if transaction has committed."""
        return self.state == TransactionState.COMMITTED

    @property
    def is_aborted(self) -> bool:
        """Returns True if transaction has aborted."""
        return self.state == TransactionState.ABORTED

    def begin(self) -> Transaction:
        """
        Explicitly begins the transaction and records a BEGIN entry in the WAL.
        """
        with self._lock:
            if self.state in (TransactionState.COMMITTED, TransactionState.ABORTED):
                raise InvalidTransactionStateError(
                    f"Cannot begin a transaction in terminal state {self.state.name}."
                )

            self.state = TransactionState.GROWING
            if self.log_manager is not None:
                self.prev_lsn = self.log_manager.log_begin(self.txn_id)

            return self

    def commit(self) -> None:
        """
        Commits the transaction:
        1. Writes a COMMIT record to the WAL and forces an fsync to disk (Durability).
        2. Releases all locks held by this transaction (Strict 2PL Shrinking phase).
        3. Sets state to COMMITTED.
        """
        with self._lock:
            if self.state == TransactionState.COMMITTED:
                return
            if self.state == TransactionState.ABORTED:
                raise InvalidTransactionStateError("Cannot commit an aborted transaction.")

            # Enter Shrinking phase
            self.state = TransactionState.SHRINKING

            # Interface with WAL: write commit record and force fsync
            if self.log_manager is not None:
                self.prev_lsn = self.log_manager.log_commit(self.txn_id)

            # Strict 2PL: Release all held locks
            if self.lock_manager is not None:
                self.lock_manager.release_all(self)

            self.savepoints.clear()
            self.state = TransactionState.COMMITTED

    def abort(self) -> None:
        """
        Aborts the transaction:
        1. Reverts any uncommitted data updates via WAL Undo / savepoint rollback.
        2. Writes an ABORT record to the WAL.
        3. Releases all locks held by this transaction (Strict 2PL Shrinking phase).
        4. Sets state to ABORTED.
        """
        with self._lock:
            if self.state == TransactionState.ABORTED:
                return
            if self.state == TransactionState.COMMITTED:
                raise InvalidTransactionStateError("Cannot abort a committed transaction.")

            # Enter Shrinking phase
            self.state = TransactionState.SHRINKING

            # Step 1: Trigger WAL Undo phase for page-level modifications
            if self.recovery_manager is not None:
                self.recovery_manager.rollback_transaction(self.txn_id)

            # Step 2: Roll back any key-level index modifications in reverse chronological order
            if self.engine is not None and self.savepoints:
                for op, key_bytes, old_entry in reversed(self.savepoints):
                    if op == "put":
                        if old_entry is None:
                            self.engine.bplus_tree.delete(key_bytes)
                        else:
                            self.engine.bplus_tree.insert(key_bytes, old_entry)
                    elif op == "delete":
                        if old_entry is not None:
                            self.engine.bplus_tree.insert(key_bytes, old_entry)
                self.savepoints.clear()

            # Step 3: Write ABORT record to WAL
            if self.log_manager is not None:
                self.prev_lsn = self.log_manager.log_abort(self.txn_id)

            # Step 4: Strict 2PL: Release all held locks
            if self.lock_manager is not None:
                self.lock_manager.release_all(self)

            self.state = TransactionState.ABORTED

    def acquire_shared(self, resource_id: Any, timeout: Optional[float] = None) -> bool:
        """Acquires a Shared (S) lock on the specified resource."""
        if self.lock_manager is None:
            raise RuntimeError("No LockManager attached to this transaction.")
        return self.lock_manager.acquire_shared(self, resource_id, timeout=timeout)

    def acquire_exclusive(self, resource_id: Any, timeout: Optional[float] = None) -> bool:
        """Acquires an Exclusive (X) lock on the specified resource."""
        if self.lock_manager is None:
            raise RuntimeError("No LockManager attached to this transaction.")
        return self.lock_manager.acquire_exclusive(self, resource_id, timeout=timeout)

    def get(
        self,
        key: Union[str, bytes],
        for_update: bool = False,
        verify_checksum: bool = False,
    ) -> bytes:
        """Retrieves value for key under this transaction context."""
        if self.engine is None:
            raise RuntimeError("No QuerixEngine attached to this transaction.")
        return self.engine.get(
            key=key,
            verify_checksum=verify_checksum,
            txn=self,
            for_update=for_update,
        )

    def put(self, key: Union[str, bytes], value: Union[str, bytes]) -> None:
        """Stores value for key under this transaction context."""
        if self.engine is None:
            raise RuntimeError("No QuerixEngine attached to this transaction.")
        self.engine.put(key=key, value=value, txn=self)

    def delete(self, key: Union[str, bytes]) -> None:
        """Deletes key under this transaction context."""
        if self.engine is None:
            raise RuntimeError("No QuerixEngine attached to this transaction.")
        self.engine.delete(key=key, txn=self)

    def record_savepoint(self, op: str, key: bytes, old_entry: Optional[Any]) -> None:
        """Records an index mutation savepoint for rollback upon abort."""
        with self._lock:
            self.savepoints.append((op, key, old_entry))

    def _record_lock(self, resource_id: Any, mode: LockMode) -> None:
        """Internal callback invoked by LockManager when a lock is granted."""
        with self._lock:
            self.locks_held[resource_id] = mode

    def _clear_locks(self) -> None:
        """Internal callback invoked by LockManager when all locks are released."""
        with self._lock:
            self.locks_held.clear()

    def __enter__(self) -> Transaction:
        if self.state != TransactionState.GROWING:
            self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            if self.state == TransactionState.GROWING:
                self.abort()
        else:
            if self.state == TransactionState.GROWING:
                self.commit()

    def __repr__(self) -> str:
        return (
            f"Transaction(id={self.txn_id}, state={self.state.name}, "
            f"locks_count={len(self.locks_held)})"
        )
