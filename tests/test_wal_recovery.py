"""
Unit tests for Write-Ahead Log (WAL) & Crash Recovery (Milestone 6).
Verifies binary log record framing, WAL protocol enforcement in BufferPoolManager,
and ARIES crash recovery (Analysis, Redo, Undo) with simulated crash and bypassed buffer pool flush.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from querix.buffer.pool_manager import BufferPoolManager
from querix.engine import QuerixEngine
from querix.storage.disk import DiskManager
from querix.storage.page import SlottedPage
from querix.wal.log_manager import (
    ActionType,
    CorruptedLogRecordError,
    LogManager,
    LogRecord,
)
from querix.wal.recovery import RecoveryManager


class TestLogRecordFormatting(unittest.TestCase):
    """Verifies binary formatting, CRC32 integrity, and serialization of LogRecords."""

    def test_serialize_and_deserialize_log_record(self):
        record = LogRecord(
            lsn=42,
            txn_id=101,
            page_id=3,
            action_type=ActionType.UPDATE,
            undo_data=b"old_slotted_page_payload_12345",
            redo_data=b"new_slotted_page_payload_67890",
            prev_lsn=39,
        )

        wire_bytes = record.serialize()
        restored = LogRecord.deserialize(wire_bytes)

        self.assertEqual(restored.lsn, 42)
        self.assertEqual(restored.txn_id, 101)
        self.assertEqual(restored.page_id, 3)
        self.assertEqual(restored.action_type, ActionType.UPDATE)
        self.assertEqual(restored.undo_data, b"old_slotted_page_payload_12345")
        self.assertEqual(restored.redo_data, b"new_slotted_page_payload_67890")
        self.assertEqual(restored.prev_lsn, 39)

    def test_corrupted_crc_raises_error(self):
        record = LogRecord(
            lsn=1,
            txn_id=1,
            page_id=0,
            action_type=ActionType.INSERT,
            undo_data=b"aaa",
            redo_data=b"bbb",
        )
        wire_bytes = bytearray(record.serialize())

        # Tamper with the payload
        wire_bytes[42] ^= 0xFF

        with self.assertRaises(CorruptedLogRecordError):
            LogRecord.deserialize(bytes(wire_bytes))


class TestWALProtocol(unittest.TestCase):
    """
    Verifies that BufferPoolManager strictly enforces the WAL Protocol:
    All log records up to page_LSN must be flushed via fsync to querix_wal.log
    before a dirty page can be written to disk.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_wal_test_")
        self.db_path = Path(self.test_dir) / "test_pages.db"
        self.wal_path = Path(self.test_dir) / "querix_wal.log"

        self.disk_mgr = DiskManager(self.db_path)
        self.bpm = BufferPoolManager(self.disk_mgr, pool_size=5)
        self.log_mgr = LogManager(self.wal_path)
        self.bpm.set_log_manager(self.log_mgr)

    def tearDown(self):
        self.bpm.close()
        self.log_mgr.close()
        self.disk_mgr.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_wal_flushed_before_dirty_page_eviction(self):
        page0 = self.bpm.new_page()
        page0_id = page0.page_id

        # Update page and log it without explicit flush
        page0.insert_record(b"WAL Protocol Test Data")
        lsn = self.log_mgr.append_record(
            txn_id=1,
            page_id=page0_id,
            action_type=ActionType.UPDATE,
            undo_data=b"",
            redo_data=b"WAL Protocol Test Data",
        )
        page0.page_lsn = lsn
        self.bpm.unpin_page(page0_id, is_dirty=True)

        # Before flush, flushed_lsn is less than lsn
        self.assertLess(self.log_mgr.flushed_lsn, lsn)

        # Flush page 0 from buffer pool: WAL protocol must force WAL flush up to LSN
        self.bpm.flush_page(page0_id)

        # Confirm WAL is now flushed on disk up to or past lsn
        self.assertGreaterEqual(self.log_mgr.flushed_lsn, lsn)


class TestCrashRecovery(unittest.TestCase):
    """
    Verifies the complete ARIES Crash Recovery Algorithm:
    1. Perform page updates and log them to WAL.
    2. Intentionally bypass buffer pool flush (simulating a sudden power failure / crash).
    3. Reopen fresh engine / recovery manager.
    4. Run recover() and verify Analysis, Redo, and Undo successfully restore lost states.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_crash_test_")
        self.db_path = Path(self.test_dir) / "test_pages.db"
        self.wal_path = Path(self.test_dir) / "querix_wal.log"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_crash_recovery_with_bypassed_buffer_pool_flush(self):
        """
        Simulates crash:
        - Txn 101 updates Page 0 and COMMITS.
        - Txn 102 updates Page 1 but NEVER commits (uncommitted loser).
        - Intentionally bypasses buffer pool flush!
        - Recovery manager executes Analysis, Redo, and Undo.
        - Verifies Page 0 has committed updates restored.
        - Verifies Page 1 has uncommitted changes rolled back.
        """
        # Step 1: Initial setup of pages on disk
        disk_mgr = DiskManager(self.db_path)
        bpm = BufferPoolManager(disk_mgr, pool_size=10)
        log_mgr = LogManager(self.wal_path)
        bpm.set_log_manager(log_mgr)

        p0 = bpm.new_page()  # Page 0
        p1 = bpm.new_page()  # Page 1
        p0.insert_record(b"Initial Page 0 Content")
        p1.insert_record(b"Initial Page 1 Content")

        # Flush initial baseline pages to disk
        bpm.flush_all_pages()
        bpm.unpin_page(0, is_dirty=False)
        bpm.unpin_page(1, is_dirty=False)

        # Capture baseline page bytes
        initial_p0_bytes = p0.serialize()
        initial_p1_bytes = p1.serialize()

        # Step 2: Transaction 101 updates Page 0 and COMMITS
        txn1 = 101
        log_mgr.log_begin(txn1)

        p0_active = bpm.fetch_page(0)
        p0_active.insert_record(b"Txn101 Committed Record on Page 0")
        updated_p0_bytes = p0_active.serialize()

        lsn1 = log_mgr.append_record(
            txn_id=txn1,
            page_id=0,
            action_type=ActionType.UPDATE,
            undo_data=initial_p0_bytes,
            redo_data=updated_p0_bytes,
        )
        p0_active.page_lsn = lsn1
        bpm.unpin_page(0, is_dirty=True)

        # Commit Txn 101 (flushes WAL to disk)
        log_mgr.log_commit(txn1)

        # Step 3: Transaction 102 updates Page 1 but DOES NOT COMMIT
        txn2 = 102
        log_mgr.log_begin(txn2)

        p1_active = bpm.fetch_page(1)
        p1_active.insert_record(b"Txn102 Uncommitted Dirty Record on Page 1")
        updated_p1_bytes = p1_active.serialize()

        lsn2 = log_mgr.append_record(
            txn_id=txn2,
            page_id=1,
            action_type=ActionType.UPDATE,
            undo_data=initial_p1_bytes,
            redo_data=updated_p1_bytes,
        )
        p1_active.page_lsn = lsn2
        bpm.unpin_page(1, is_dirty=True)

        # Ensure WAL record for Txn 102 is written to WAL log file
        log_mgr.flush_up_to(lsn2)

        # =========================================================================
        # SIMULATE CRASH:
        # Intentionally BYPASS buffer pool flush!
        # Close the log file, but discard dirty buffer pool memory without writing
        # page 0 or page 1 updates to disk_mgr!
        # =========================================================================
        log_mgr.close()
        disk_mgr.close()

        # Confirm that before recovery, disk file still has ONLY the initial content
        fresh_disk = DiskManager(self.db_path)
        disk_raw_p0 = fresh_disk.read_page(0)
        disk_raw_p1 = fresh_disk.read_page(1)
        self.assertEqual(disk_raw_p0, initial_p0_bytes)
        self.assertEqual(disk_raw_p1, initial_p1_bytes)
        fresh_disk.close()

        # =========================================================================
        # RECOVERY:
        # Instantiate fresh recovery manager on existing database and WAL files
        # =========================================================================
        recovered_disk = DiskManager(self.db_path)
        recovered_bpm = BufferPoolManager(recovered_disk, pool_size=10)
        recovered_log_mgr = LogManager(self.wal_path)
        recovered_bpm.set_log_manager(recovered_log_mgr)

        recovery_mgr = RecoveryManager(recovered_log_mgr, recovered_bpm)
        result = recovery_mgr.recover()

        # Verify Analysis Phase:
        # Txn 101 was committed; Txn 102 was active/uncommitted (loser)
        self.assertIn(101, result.txns_committed)
        self.assertIn(102, result.txns_active)

        # Verify Redo & Undo stats:
        self.assertGreater(result.pages_redone, 0)
        self.assertGreater(result.pages_undone, 0)

        # Verify Page 0 state: Txn 101's committed update is RESTORED!
        recovered_p0 = recovered_bpm.fetch_page(0)
        self.assertEqual(recovered_p0.serialize(), updated_p0_bytes)
        self.assertEqual(recovered_p0.get_record(1), b"Txn101 Committed Record on Page 0")
        recovered_bpm.unpin_page(0, is_dirty=False)

        # Verify Page 1 state: Txn 102's uncommitted change was ROLLED BACK to initial state!
        recovered_p1 = recovered_bpm.fetch_page(1)
        self.assertEqual(recovered_p1.serialize(), initial_p1_bytes)
        self.assertEqual(len(recovered_p1), 1)
        self.assertEqual(recovered_p1.get_record(0), b"Initial Page 1 Content")
        recovered_bpm.unpin_page(1, is_dirty=False)

        recovered_bpm.close()
        recovered_log_mgr.close()
        recovered_disk.close()

    def test_engine_startup_automatic_recovery(self):
        """Verifies that QuerixEngine runs WAL recovery automatically on startup."""
        engine = QuerixEngine(self.test_dir)
        self.assertIsNotNone(engine.recovery_result)
        self.assertEqual(engine.recovery_result.records_scanned, 0)

        # Put a record
        engine.put("user:99", "Crash Recovery Test")
        self.assertEqual(engine.get_str("user:99"), "Crash Recovery Test")
        engine.close()

        # Reopen engine on same dir: recovery runs and finds the state
        engine2 = QuerixEngine(self.test_dir)
        self.assertEqual(engine2.get_str("user:99"), "Crash Recovery Test")
        engine2.close()


if __name__ == "__main__":
    unittest.main()
