"""
Unit and integration tests for QuerixEngine with Multi-Segment support.
Verifies CRUD operations, binary stability, segmentation rolling, and persistence across process restarts.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from querix.engine import QuerixEngine


class TestQuerixEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_test_")
        self.db_dir = Path(self.test_dir) / "test_db"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_put_and_get(self):
        db = QuerixEngine(self.db_dir)
        db.put("user:1", "Alice")
        db.put(b"user:2", b"Bob")

        self.assertEqual(db.get("user:1"), b"Alice")
        self.assertEqual(db.get_str("user:1"), "Alice")
        self.assertEqual(db.get(b"user:2"), b"Bob")
        self.assertEqual(len(db), 2)
        self.assertIn("user:1", db)
        self.assertIn("user:2", db)
        db.close()

    def test_overwrite_key(self):
        db = QuerixEngine(self.db_dir)
        db.put("counter", "1")
        self.assertEqual(db.get_str("counter"), "1")

        db.put("counter", "2")
        self.assertEqual(db.get_str("counter"), "2")

        db.put("counter", "42")
        self.assertEqual(db.get_str("counter"), "42")
        self.assertEqual(len(db), 1)
        db.close()

    def test_delete_key(self):
        db = QuerixEngine(self.db_dir)
        db.put("session:xyz", "active")
        self.assertIn("session:xyz", db)

        db.delete("session:xyz")
        self.assertNotIn("session:xyz", db)
        self.assertEqual(len(db), 0)

        with self.assertRaises(KeyError):
            db.get("session:xyz")

        with self.assertRaises(KeyError):
            db.delete("session:xyz")
        db.close()

    def test_persistence_across_restart(self):
        """
        Critical test: Data must survive completely closing the engine
        and instantiating a fresh engine instance on the same directory.
        """
        # Step 1: Populate initial database
        db1 = QuerixEngine(self.db_dir)
        for i in range(100):
            db1.put(f"item:{i}", f"value_{i}")

        # Overwrite some keys
        for i in range(0, 20):
            db1.put(f"item:{i}", f"updated_{i}")

        # Delete some keys
        for i in range(80, 100):
            db1.delete(f"item:{i}")

        self.assertEqual(len(db1), 80)
        db1.close()

        # Step 2: Simulate complete process restart by opening brand new instance
        db2 = QuerixEngine(self.db_dir)

        self.assertEqual(len(db2), 80)

        # Verify updated keys
        for i in range(0, 20):
            self.assertEqual(db2.get_str(f"item:{i}"), f"updated_{i}")

        # Verify untouched keys
        for i in range(20, 80):
            self.assertEqual(db2.get_str(f"item:{i}"), f"value_{i}")

        # Verify deleted keys are not found
        for i in range(80, 100):
            self.assertNotIn(f"item:{i}", db2)
            with self.assertRaises(KeyError):
                db2.get(f"item:{i}")

        db2.close()

    def test_segmentation_rolling_on_threshold(self):
        """
        Verifies that writing data exceeding max_segment_size triggers automatic
        rolling into multiple segment files.
        """
        # Set a small threshold of 250 bytes (~5 records)
        db = QuerixEngine(self.db_dir, max_segment_size=250)

        for i in range(25):
            db.put(f"key_{i:03d}", f"payload_value_for_segment_test_{i:03d}")

        # We should now have multiple segments created
        self.assertGreater(db.segments_count, 1)
        self.assertGreater(db.active_segment_id, 1)

        # Ensure all 25 keys are correctly retrievable across all segments
        for i in range(25):
            self.assertEqual(
                db.get_str(f"key_{i:03d}"),
                f"payload_value_for_segment_test_{i:03d}",
            )

        db.close()

    def test_multi_segment_boot_recovery(self):
        """
        Verifies that on startup, the engine scans and reconstructs state
        across multiple historical segment files in correct chronological order.
        """
        # Threshold: 200 bytes per segment
        db1 = QuerixEngine(self.db_dir, max_segment_size=200)

        # Write 30 records
        for i in range(30):
            db1.put(f"k:{i}", f"val:{i}")

        # Overwrite earlier keys
        for i in range(0, 10):
            db1.put(f"k:{i}", f"overwritten:{i}")

        # Delete keys
        db1.delete("k:29")
        db1.delete("k:28")

        initial_segment_count = db1.segments_count
        self.assertGreater(initial_segment_count, 1)
        db1.close()

        # Reopen on the same directory
        db2 = QuerixEngine(self.db_dir, max_segment_size=200)
        self.assertEqual(db2.segments_count, initial_segment_count)
        self.assertEqual(len(db2), 28)

        # Overwritten keys check
        for i in range(0, 10):
            self.assertEqual(db2.get_str(f"k:{i}"), f"overwritten:{i}")

        # Untouched keys check
        for i in range(10, 28):
            self.assertEqual(db2.get_str(f"k:{i}"), f"val:{i}")

        # Deleted keys check
        self.assertNotIn("k:28", db2)
        self.assertNotIn("k:29", db2)
        db2.close()

    def test_crash_recovery_with_torn_write(self):
        """
        Verifies that partial/torn writes at the end of the log (simulating a power failure)
        do not prevent the database from recovering prior committed transactions.
        """
        db = QuerixEngine(self.db_dir)
        db.put("committed_key", "committed_value")
        active_path = db.active_segment_path
        db.close()

        # Append corrupted/incomplete bytes to the end of the active segment file
        with open(active_path, "ab") as f:
            f.write(b"\x00\x01\x02\x03JUNK_TRUNCATED_RECORD")

        # Reopen database: should recover without crash and read committed key
        db_recovered = QuerixEngine(self.db_dir)
        self.assertEqual(db_recovered.get_str("committed_key"), "committed_value")
        self.assertEqual(len(db_recovered), 1)
        db_recovered.close()

    def test_context_manager(self):
        with QuerixEngine(self.db_dir) as db:
            db.put("ctx_key", "ctx_val")
            self.assertEqual(db.get_str("ctx_key"), "ctx_val")

        # After context exit, db is closed
        with self.assertRaises(RuntimeError):
            db.put("new_key", "new_val")


if __name__ == "__main__":
    unittest.main()
