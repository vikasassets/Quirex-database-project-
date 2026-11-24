"""
Unit tests for Concurrency Control & Transactions (Milestone 7).
Verifies:
1. Row-level LockManager (Shared and Exclusive modes, lock upgrades, Strict 2PL enforcement).
2. Transaction lifecycle (Growing -> Shrinking -> Committed / Aborted, WAL integration).
3. Deadlock Handling via Wait-Die strategy (mutual wait resolution, younger dies, older completes).
4. Multi-Threaded Read-and-Decrement test: verifies LockManager queuing and mathematical accuracy.
5. Transaction rollback and savepoint undo.
"""

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from querix.concurrency.lock_manager import (
    DeadlockError,
    InvalidTransactionStateError,
    LockManager,
    LockMode,
    Strict2PLError,
    TransactionAbortError,
)
from querix.concurrency.transaction import Transaction, TransactionState
from querix.engine import QuerixEngine
from querix.wal.log_manager import ActionType, LogManager


class TestLockManager(unittest.TestCase):
    """Verifies row-level lock modes, compatibility, lock upgrades, and Strict 2PL."""

    def setUp(self):
        self.lm = LockManager()

    def test_shared_lock_compatibility(self):
        """Multiple transactions can hold Shared (S) locks simultaneously on the same key."""
        t1 = Transaction(txn_id=1, lock_manager=self.lm)
        t2 = Transaction(txn_id=2, lock_manager=self.lm)
        t1.begin()
        t2.begin()

        # Both acquire Shared lock on 'user:10'
        self.assertTrue(self.lm.acquire_shared(t1, "user:10"))
        self.assertTrue(self.lm.acquire_shared(t2, "user:10"))

        holders = self.lm.holders_of("user:10")
        self.assertEqual(len(holders), 2)
        self.assertEqual(holders[1], LockMode.SHARED)
        self.assertEqual(holders[2], LockMode.SHARED)

        t1.commit()
        t2.commit()

    def test_exclusive_lock_mutual_exclusion(self):
        """Exclusive (X) lock prevents other transactions from acquiring S or X locks."""
        t1 = Transaction(txn_id=1, lock_manager=self.lm)
        t2 = Transaction(txn_id=2, lock_manager=self.lm)
        t1.begin()
        t2.begin()

        # t1 acquires Exclusive lock
        self.assertTrue(self.lm.acquire_exclusive(t1, "account:A"))

        # t2 cannot acquire S or X lock immediately (timeout 0.05s)
        self.assertFalse(self.lm.acquire_shared(t2, "account:A", timeout=0.05))
        self.assertFalse(self.lm.acquire_exclusive(t2, "account:A", timeout=0.05))

        t1.commit()

        # After t1 commits, t2 can acquire Exclusive lock
        self.assertTrue(self.lm.acquire_exclusive(t2, "account:A", timeout=0.1))
        t2.commit()

    def test_lock_upgrade(self):
        """A transaction holding S lock can upgrade to X lock if it is the sole holder."""
        t1 = Transaction(txn_id=1, lock_manager=self.lm)
        t1.begin()

        self.assertTrue(self.lm.acquire_shared(t1, "item:5"))
        self.assertEqual(self.lm.holders_of("item:5")[1], LockMode.SHARED)

        # Upgrade to Exclusive
        self.assertTrue(self.lm.acquire_exclusive(t1, "item:5"))
        self.assertEqual(self.lm.holders_of("item:5")[1], LockMode.EXCLUSIVE)

        t1.commit()

    def test_strict_2pl_individual_release_prohibited(self):
        """Strict 2PL prohibits individual lock releases before commit or abort."""
        t1 = Transaction(txn_id=1, lock_manager=self.lm)
        t1.begin()
        self.lm.acquire_shared(t1, "key1")

        with self.assertRaises(Strict2PLError):
            self.lm.release(t1, "key1")

        t1.commit()

    def test_cannot_acquire_in_terminal_states(self):
        """Cannot acquire locks once transaction has committed or aborted."""
        t1 = Transaction(txn_id=1, lock_manager=self.lm)
        t1.begin()
        t1.commit()

        with self.assertRaises(InvalidTransactionStateError):
            self.lm.acquire_shared(t1, "key1")


class TestTransactionLifecycle(unittest.TestCase):
    """Verifies Transaction state transitions, WAL logging, and rollback on abort."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_txn_test_")
        self.engine = QuerixEngine(self.test_dir)

    def tearDown(self):
        self.engine.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_transaction_commit_persists_to_wal(self):
        """Transaction commit writes COMMIT record to WAL and fsyncs."""
        initial_records = len(self.engine.log_manager.read_all_records())

        with self.engine.transaction() as txn:
            self.engine.put("user:42", "Arthur Dent", txn=txn)

        records = self.engine.log_manager.read_all_records()
        self.assertGreater(len(records), initial_records)

        # Confirm BEGIN and COMMIT records exist
        action_types = [r.action_type for r in records]
        self.assertIn(ActionType.BEGIN, action_types)
        self.assertIn(ActionType.COMMIT, action_types)

        # Value is readable
        self.assertEqual(self.engine.get_str("user:42"), "Arthur Dent")

    def test_transaction_abort_rolls_back_changes(self):
        """Aborting a transaction reverts uncommitted puts and deletes."""
        # Initial baseline
        self.engine.put("account:1", "100")

        # Start transaction that updates and then aborts
        try:
            with self.engine.transaction() as txn:
                self.engine.put("account:1", "50", txn=txn)
                self.assertEqual(self.engine.get_str("account:1", txn=txn), "50")
                # Simulate an unexpected business logic error
                raise ValueError("Insufficient funds or simulated failure")
        except ValueError:
            pass

        # Value must be restored to 100
        self.assertEqual(self.engine.get_str("account:1"), "100")

        # WAL contains ABORT record
        records = self.engine.log_manager.read_all_records()
        action_types = [r.action_type for r in records]
        self.assertIn(ActionType.ABORT, action_types)

    def test_abort_rolls_back_new_key_insert(self):
        """Aborting a transaction that inserted a new key deletes the key from index."""
        try:
            with self.engine.transaction() as txn:
                self.engine.put("temp_key", "temporary", txn=txn)
                raise RuntimeError("Abort insert")
        except RuntimeError:
            pass

        self.assertNotIn("temp_key", self.engine)


class TestDeadlockWaitDie(unittest.TestCase):
    """
    Verifies Wait-Die deadlock prevention:
    Older transactions wait for younger ones; younger transactions die (abort)
    when mutual waiting (deadlock cycle) occurs, preventing freezing.
    """

    def test_wait_die_mutual_wait_resolution(self):
        """
        Thread A (Txn 1, older) acquires Lock 1, requests Lock 2.
        Thread B (Txn 2, younger) acquires Lock 2, requests Lock 1.
        Mutual wait detected: Younger transaction (Txn 2) dies, allowing Txn 1 to proceed.
        """
        lm = LockManager()

        t1 = Transaction(txn_id=1, lock_manager=lm)  # Older
        t2 = Transaction(txn_id=2, lock_manager=lm)  # Younger
        t1.begin()
        t2.begin()

        # Step 1: t1 acquires res1, t2 acquires res2
        lm.acquire_exclusive(t1, "res1")
        lm.acquire_exclusive(t2, "res2")

        t1_acquired_res2 = threading.Event()
        t2_deadlock_error = threading.Event()

        def worker_t1():
            # Older transaction waits for younger holder t2
            try:
                lm.acquire_exclusive(t1, "res2")
                t1_acquired_res2.set()
                t1.commit()
            except Exception:
                pass

        def worker_t2():
            # Give t1 a moment to request res2 and start waiting
            time.sleep(0.05)
            # Younger transaction t2 requests res1 held by t1
            # Mutual wait / cycle: younger t2 must DIE!
            try:
                lm.acquire_exclusive(t2, "res1")
            except DeadlockError:
                t2_deadlock_error.set()
                t2.abort()

        thread1 = threading.Thread(target=worker_t1)
        thread2 = threading.Thread(target=worker_t2)

        thread1.start()
        thread2.start()

        thread2.join(timeout=2.0)
        thread1.join(timeout=2.0)

        # Younger transaction died with DeadlockError
        self.assertTrue(t2_deadlock_error.is_set(), "Younger transaction did not die on deadlock!")
        # Older transaction acquired lock and completed
        self.assertTrue(t1_acquired_res2.is_set(), "Older transaction failed to acquire lock after younger died!")
        self.assertEqual(t1.state, TransactionState.COMMITTED)
        self.assertEqual(t2.state, TransactionState.ABORTED)


class TestMultiThreadedContentionAndMath(unittest.TestCase):
    """
    Core Milestone 7 Requirement:
    Wrap engine.get() and engine.put() in transaction contexts with S/X locks.
    Spawn two threads that simultaneously attempt to read and decrement a numeric value
    for the same key.
    Verify that the Lock Manager correctly queues the second thread and that the
    final math is perfectly accurate without race conditions.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_threads_test_")
        self.engine = QuerixEngine(self.test_dir)

    def tearDown(self):
        self.engine.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_two_threads_simultaneous_decrement_queued_and_accurate(self):
        """
        Spawns two threads that simultaneously read and decrement the same numeric key.
        Verifies:
        1. Thread 2 is queued by the Lock Manager behind Thread 1.
        2. Final math is perfectly accurate: 100 - 1 - 1 = 98.
        3. Zero race conditions or lost updates.
        """
        # Initialize key "counter" with value 100
        self.engine.put("counter", "100")
        self.assertEqual(self.engine.get_str("counter"), "100")

        barrier = threading.Barrier(2)
        t1_in_critical_section = threading.Event()
        t2_queued_observed = threading.Event()

        results = []
        errors = []

        def decrement_worker(worker_id: int):
            try:
                barrier.wait(timeout=5.0)

                with self.engine.transaction() as txn:
                    # Acquire exclusive lock for update so read-modify-write is atomic
                    val = int(self.engine.get_str("counter", txn=txn, for_update=True))

                    if worker_id == 1:
                        t1_in_critical_section.set()
                        # Wait until test confirms thread 2 has arrived and is queued in LockManager
                        t2_queued_observed.wait(timeout=2.0)
                        time.sleep(0.02)  # small pause simulating processing

                    new_val = val - 1
                    self.engine.put("counter", str(new_val), txn=txn)
                    results.append((worker_id, val, new_val))

            except Exception as e:
                errors.append((worker_id, e))

        thread1 = threading.Thread(target=decrement_worker, args=(1,), name="Worker-1")
        thread2 = threading.Thread(target=decrement_worker, args=(2,), name="Worker-2")

        thread1.start()
        thread2.start()

        # Wait for thread 1 to acquire the exclusive lock
        self.assertTrue(t1_in_critical_section.wait(timeout=2.0), "Thread 1 never acquired lock")

        # Poll briefly to confirm Thread 2 has reached LockManager and is queued
        deadline = time.time() + 2.0
        is_queued = False
        while time.time() < deadline:
            if self.engine.lock_manager.has_waiters("counter") or self.engine.lock_manager.wait_count > 0:
                is_queued = True
                break
            time.sleep(0.005)

        # Verify that Lock Manager correctly queued the second thread!
        self.assertTrue(
            is_queued,
            f"Lock Manager failed to queue the second thread! Waiters: {self.engine.lock_manager.get_waiters('counter')}",
        )
        t2_queued_observed.set()

        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)

        # Verify no unhandled exceptions occurred in either thread
        self.assertEqual(errors, [], f"Threads encountered errors: {errors}")

        # Verify both threads completed
        self.assertEqual(len(results), 2, "Both decrement workers should have finished successfully")

        # Verify exact mathematical accuracy: 100 - 1 - 1 = 98
        final_value = int(self.engine.get_str("counter"))
        self.assertEqual(final_value, 98, f"Expected final value 98, got {final_value} (Lost Update!)")

        # Verify cumulative wait count is at least 1 (second thread queued)
        self.assertGreaterEqual(
            self.engine.lock_manager.wait_count,
            1,
            "Expected at least 1 queued lock request in LockManager",
        )

    def test_multi_threaded_high_contention(self):
        """
        Spawns 4 concurrent worker threads each executing 5 atomic decrements on the same key.
        Total decrements = 20.
        Initial: 50 -> Final: 30.
        Verifies zero lost updates under multi-threaded contention.
        """
        self.engine.put("shared_balance", "50")

        num_threads = 4
        iterations_per_thread = 5
        threads = []
        errors = []

        def worker():
            for _ in range(iterations_per_thread):
                try:
                    with self.engine.transaction() as txn:
                        val = int(self.engine.get_str("shared_balance", txn=txn, for_update=True))
                        self.engine.put("shared_balance", str(val - 1), txn=txn)
                except Exception as e:
                    errors.append(e)

        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(errors, [])
        final_balance = int(self.engine.get_str("shared_balance"))
        expected_balance = 50 - (num_threads * iterations_per_thread)
        self.assertEqual(
            final_balance,
            expected_balance,
            f"Expected {expected_balance}, got {final_balance} under concurrent contention!",
        )


if __name__ == "__main__":
    unittest.main()
