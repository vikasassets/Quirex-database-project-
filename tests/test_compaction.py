"""
Unit and integration tests for Log Compaction & Segment Merging.
Verifies space reclamation, dead record purging, tombstone elimination,
and post-compaction persistence.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from querix.engine import QuerixEngine


class TestCompaction(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_compact_test_")
        self.db_dir = Path(self.test_dir) / "test_db"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compaction_reclaims_space_from_overwrites(self):
        """
        Overwrites a key 100 times. Before compaction, the file contains 100 records.
        After compaction, only 1 live record remains, reclaiming massive dead disk space.
        """
        db = QuerixEngine(self.db_dir, max_segment_size=1024)

        for i in range(100):
            db.put("hot_key", f"version_{i:04d}")

        size_before = db.total_disk_size
        self.assertGreater(size_before, 2000)

        # Execute compaction
        result = db.compact()

        size_after = db.total_disk_size
        self.assertGreater(result.bytes_reclaimed, 0)
        self.assertLess(size_after, size_before)
        self.assertEqual(result.records_retained, 1)
        self.assertEqual(result.records_dropped, 99)

        # Verify key value is still exact latest version
        self.assertEqual(db.get_str("hot_key"), "version_0099")
        self.assertEqual(len(db), 1)

        db.close()

    def test_compaction_purges_tombstones(self):
        """
        Inserts 50 keys, deletes 40 keys. Compaction must drop all tombstones
        and their associated deleted payloads.
        """
        db = QuerixEngine(self.db_dir, max_segment_size=1024)

        for i in range(50):
            db.put(f"item:{i}", f"payload_{i}")

        for i in range(0, 40):
            db.delete(f"item:{i}")

        self.assertEqual(len(db), 10)

        result = db.compact()
        self.assertEqual(result.records_retained, 10)
        self.assertEqual(len(db), 10)

        # Verify 40 deleted keys are truly absent
        for i in range(0, 40):
            self.assertNotIn(f"item:{i}", db)
            with self.assertRaises(KeyError):
                db.get(f"item:{i}")

        # Verify remaining 10 keys exist
        for i in range(40, 50):
            self.assertEqual(db.get_str(f"item:{i}"), f"payload_{i}")

        db.close()

    def test_compaction_merges_multiple_segments(self):
        """
        Generates multiple segments by writing past the threshold.
        Compaction merges all frozen segments into a single compact segment.
        """
        # 300 bytes threshold forces segment rolling every ~5 writes
        db = QuerixEngine(self.db_dir, max_segment_size=300)

        for i in range(40):
            db.put(f"k:{i}", f"long_value_string_forcing_segment_split_{i:03d}")

        # Overwrite first 20 keys
        for i in range(20):
            db.put(f"k:{i}", f"overwritten_{i:03d}")

        segments_before = db.segments_count
        self.assertGreater(segments_before, 3)

        result = db.compact()

        # Compaction merged all frozen segments
        self.assertGreater(result.records_scanned, 40)
        self.assertEqual(result.records_retained, 40)
        self.assertEqual(len(db), 40)

        # Check all values
        for i in range(20):
            self.assertEqual(db.get_str(f"k:{i}"), f"overwritten_{i:03d}")
        for i in range(20, 40):
            self.assertEqual(db.get_str(f"k:{i}"), f"long_value_string_forcing_segment_split_{i:03d}")

        db.close()

    def test_persistence_after_compaction(self):
        """
        Verifies that after compaction, reopening the database preserves all
        compacted data and metadata accurately.
        """
        db1 = QuerixEngine(self.db_dir, max_segment_size=500)
        for i in range(30):
            db1.put(f"user:{i}", f"data_{i}")
        for i in range(15):
            db1.put(f"user:{i}", f"updated_data_{i}")
        for i in range(25, 30):
            db1.delete(f"user:{i}")

        db1.compact()
        self.assertEqual(len(db1), 25)
        db1.close()

        # Reopen on same directory
        db2 = QuerixEngine(self.db_dir, max_segment_size=500)
        self.assertEqual(len(db2), 25)

        for i in range(15):
            self.assertEqual(db2.get_str(f"user:{i}"), f"updated_data_{i}")
        for i in range(15, 25):
            self.assertEqual(db2.get_str(f"user:{i}"), f"data_{i}")
        for i in range(25, 30):
            self.assertNotIn(f"user:{i}", db2)

        db2.close()

    def test_compaction_on_empty_db(self):
        """Verifies no-op on empty database."""
        db = QuerixEngine(self.db_dir)
        result = db.compact()
        self.assertEqual(result.records_scanned, 0)
        self.assertEqual(result.records_retained, 0)
        self.assertEqual(result.bytes_reclaimed, 0)
        db.close()


if __name__ == "__main__":
    unittest.main()
