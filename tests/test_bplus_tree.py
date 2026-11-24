"""
Unit tests for Disk-Backed B+ Tree Index (Milestone 5).
Verifies node architecture, leaf & internal node splitting within 4KB SlottedPages,
1,000 sequential key insertions, point lookups, ordered range scans,
buffer pool pin-leak prevention, and engine range API integration.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from querix.buffer.pool_manager import BufferPoolManager
from querix.engine import QuerixEngine
from querix.index.bplus_tree import BPlusTree, InternalNode, LeafNode
from querix.storage.disk import DiskManager
from querix.storage.page import SlottedPage


class TestBPlusTree(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_bpt_test_")
        self.db_path = Path(self.test_dir) / "bpt_index.db"
        self.disk_mgr = DiskManager(self.db_path)
        self.bpm = BufferPoolManager(self.disk_mgr, pool_size=20)

    def tearDown(self):
        self.bpm.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_node_serialization_to_slotted_page(self):
        """Verifies leaf and internal nodes serialize to and deserialize from 4KB SlottedPages."""
        # 1. Test LeafNode
        page = SlottedPage(page_id=5)
        leaf = LeafNode(
            page_id=5,
            keys=[b"apple", b"banana", b"cherry"],
            values=[(1, 10), (1, 20), (2, 5)],
            next_page_id=12,
        )
        leaf.write_to_page(page)

        # Deserialize back
        raw_bytes = page.serialize()
        self.assertEqual(len(raw_bytes), 4096)
        restored_page = SlottedPage.deserialize(raw_bytes)
        restored_leaf = LeafNode.from_page(restored_page)

        self.assertEqual(restored_leaf.page_id, 5)
        self.assertEqual(restored_leaf.keys, [b"apple", b"banana", b"cherry"])
        self.assertEqual(restored_leaf.values, [(1, 10), (1, 20), (2, 5)])
        self.assertEqual(restored_leaf.next_page_id, 12)

        # 2. Test InternalNode
        page_int = SlottedPage(page_id=9)
        internal = InternalNode(
            page_id=9,
            keys=[b"banana", b"mango"],
            children=[1, 2, 3],
        )
        internal.write_to_page(page_int)

        raw_int = page_int.serialize()
        restored_int_page = SlottedPage.deserialize(raw_int)
        restored_internal = InternalNode.from_page(restored_int_page)

        self.assertEqual(restored_internal.page_id, 9)
        self.assertEqual(restored_internal.keys, [b"banana", b"mango"])
        self.assertEqual(restored_internal.children, [1, 2, 3])

    def test_1000_sequential_inserts_splits_and_lookups(self):
        """
        Core Milestone 5 Requirement:
        Inserts 1,000 sequentially ordered keys, forcing multiple root and leaf splits,
        verifies point lookups, ordered range scans, and confirms zero pin leaks.
        """
        tree = BPlusTree(self.bpm)

        # Insert 1,000 keys sequentially
        num_keys = 1000
        for i in range(num_keys):
            key = f"key_{i:06d}".encode("utf-8")
            record_pointer = (i // 100, i % 100, i * 10)
            tree.insert(key, record_pointer)

        # Total keys in B+ Tree
        self.assertEqual(len(tree), num_keys)

        # Point lookups for all 1,000 keys
        for i in range(num_keys):
            key = f"key_{i:06d}".encode("utf-8")
            expected_pointer = (i // 100, i % 100, i * 10)
            found = tree.search(key)
            self.assertEqual(found, expected_pointer, f"Lookup failed for key {key!r}")

        # Lookups for non-existent keys
        self.assertIsNone(tree.search(b"key_999999"))
        self.assertIsNone(tree.search(b"non_existent_key"))
        self.assertNotIn(b"key_001000", tree)

        # Range Scan 1: Sub-range of 100 keys
        scan_results = tree.range_scan(b"key_000200", b"key_000299")
        self.assertEqual(len(scan_results), 100)
        for offset, (k, ptr) in enumerate(scan_results):
            expected_idx = 200 + offset
            expected_key = f"key_{expected_idx:06d}".encode("utf-8")
            expected_ptr = (expected_idx // 100, expected_idx % 100, expected_idx * 10)
            self.assertEqual(k, expected_key)
            self.assertEqual(ptr, expected_ptr)

        # Range Scan 2: Arbitrary slice across page boundaries
        scan_results_2 = tree.range_scan(b"key_000550", b"key_000574")
        self.assertEqual(len(scan_results_2), 25)
        self.assertEqual(scan_results_2[0][0], b"key_000550")
        self.assertEqual(scan_results_2[-1][0], b"key_000574")

        # Range Scan 3: Entire dataset (start_key=None, end_key=None)
        all_results = tree.range_scan(None, None)
        self.assertEqual(len(all_results), 1000)
        for i in range(1000):
            self.assertEqual(all_results[i][0], f"key_{i:06d}".encode("utf-8"))

        # Critical Check: Verify ZERO buffer pool pin leaks after all operations
        for frame in self.bpm.frames:
            self.assertEqual(
                frame.pin_count, 0,
                f"Memory leak: frame {frame.frame_id} holding page {frame.page_id} has pin_count={frame.pin_count}"
            )
        # All cached frames must be eligible for eviction in the LRU replacer
        self.assertEqual(len(self.bpm.replacer), len(self.bpm))

    def test_small_max_keys_rapid_splitting(self):
        """Forces multiple leaf and internal splits with max_keys=4."""
        tree = BPlusTree(self.bpm, max_keys=4)

        keys_to_insert = [
            b"k10", b"k20", b"k30", b"k40", b"k50",
            b"k60", b"k70", b"k80", b"k90", b"k95",
        ]

        for k in keys_to_insert:
            tree.insert(k, (len(k), 1))

        self.assertEqual(len(tree), len(keys_to_insert))

        for k in keys_to_insert:
            self.assertEqual(tree.search(k), (len(k), 1))

        # Range scan
        res = tree.range_scan(b"k20", b"k70")
        expected_keys = [b"k20", b"k30", b"k40", b"k50", b"k60", b"k70"]
        self.assertEqual([k for k, _ in res], expected_keys)

        # Verify no pins leaked
        for frame in self.bpm.frames:
            self.assertEqual(frame.pin_count, 0)

    def test_update_and_delete(self):
        """Verifies in-place update and deletion of keys in B+ Tree."""
        tree = BPlusTree(self.bpm, max_keys=5)

        tree.insert(b"alpha", (1, 100))
        tree.insert(b"beta", (2, 200))
        tree.insert(b"gamma", (3, 300))

        # Update alpha
        tree.insert(b"alpha", (1, 999))
        self.assertEqual(tree.search(b"alpha"), (1, 999))
        self.assertEqual(len(tree), 3)

        # Delete beta
        self.assertTrue(tree.delete(b"beta"))
        self.assertIsNone(tree.search(b"beta"))
        self.assertEqual(len(tree), 2)

        # Deleting non-existent key returns False
        self.assertFalse(tree.delete(b"non_existent"))

        # Verify no pins leaked
        for frame in self.bpm.frames:
            self.assertEqual(frame.pin_count, 0)

    def test_persistence_across_bpm_restart(self):
        """Verifies B+ tree pages persist on disk and reload seamlessly."""
        # Insert keys in first session
        tree1 = BPlusTree(self.bpm, max_keys=4)
        for i in range(50):
            tree1.insert(f"item_{i:03d}".encode("utf-8"), (i, i * 2))

        # Flush all pages and close
        self.bpm.close()

        # Reopen on same database file
        disk_mgr2 = DiskManager(self.db_path)
        bpm2 = BufferPoolManager(disk_mgr2, pool_size=10)
        tree2 = BPlusTree(bpm2, max_keys=4)

        # Confirm all 50 keys are intact
        self.assertEqual(len(tree2), 50)
        for i in range(50):
            self.assertEqual(
                tree2.search(f"item_{i:03d}".encode("utf-8")),
                (i, i * 2),
            )

        bpm2.close()
        disk_mgr2.close()


class TestEngineBPlusTreeIntegration(unittest.TestCase):
    """Verifies that QuerixEngine routes calls through BPlusTree including range_scan API."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_engine_bpt_test_")
        self.db_dir = Path(self.test_dir) / "test_engine_db"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_engine_range_scan_api(self):
        db = QuerixEngine(self.db_dir)

        # Insert 50 users
        for i in range(50):
            db.put(f"user:{i:03d}", f"User_Name_{i:03d}")

        self.assertEqual(len(db), 50)

        # Exact point lookup
        self.assertEqual(db.get_str("user:025"), "User_Name_025")

        # Range scan from user:010 to user:015
        range_results = db.range_scan("user:010", "user:015")
        self.assertEqual(len(range_results), 6)

        expected_pairs = [
            (f"user:{i:03d}".encode("utf-8"), f"User_Name_{i:03d}".encode("utf-8"))
            for i in range(10, 16)
        ]
        self.assertEqual(range_results, expected_pairs)

        # Range convenience alias
        alias_results = db.range("user:040", "user:042")
        self.assertEqual(len(alias_results), 3)

        db.close()


if __name__ == "__main__":
    unittest.main()
