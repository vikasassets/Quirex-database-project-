"""
Unit tests for Buffer Pool Manager (Milestone 4).
Verifies LRU eviction, dirty page flushing and eviction, pin count tracking,
page allocation, and disk persistence.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from querix.buffer.pool_manager import (
    BufferPoolFullError,
    BufferPoolManager,
    Frame,
    LRUReplacer,
)
from querix.storage.disk import DiskManager
from querix.storage.page import PAGE_SIZE, SlottedPage


class TestLRUReplacer(unittest.TestCase):
    """Verifies standalone LRU Replacer eviction and pinning operations."""

    def test_replacer_basic_operations(self):
        replacer = LRUReplacer(pool_size=4)
        self.assertEqual(len(replacer), 0)

        # Unpin frames 0, 1, 2
        replacer.unpin(0)
        replacer.unpin(1)
        replacer.unpin(2)
        self.assertEqual(len(replacer), 3)

        # Pin frame 1 (should remove it from eviction eligibility)
        replacer.pin(1)
        self.assertEqual(len(replacer), 2)
        self.assertNotIn(1, replacer)

        # Victim should be frame 0 (least recently unpinned)
        victim = replacer.victim()
        self.assertEqual(victim, 0)
        self.assertEqual(len(replacer), 1)

        # Victim should now be frame 2
        victim = replacer.victim()
        self.assertEqual(victim, 2)
        self.assertEqual(len(replacer), 0)

        # Empty replacer returns None
        self.assertIsNone(replacer.victim())

    def test_replacer_mru_update_on_re_unpin(self):
        replacer = LRUReplacer(pool_size=3)
        replacer.unpin(0)
        replacer.unpin(1)

        # Re-unpin 0: moves 0 to the MRU position
        replacer.unpin(0)

        # Victim should now be 1, because 0 was touched more recently
        self.assertEqual(replacer.victim(), 1)
        self.assertEqual(replacer.victim(), 0)


class TestBufferPoolManager(unittest.TestCase):
    """
    Verifies BufferPoolManager core requirements:
    1. Fetching cached and uncached pages
    2. Pin counting and unpinning
    3. Dirty page tracking and persistence on eviction
    4. Forcible page flushing
    5. LRU eviction order
    6. Buffer pool full handling
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="querix_bpm_test_")
        self.db_path = Path(self.test_dir) / "test_pages.db"
        self.disk_manager = DiskManager(self.db_path)

    def tearDown(self):
        self.disk_manager.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_new_page_and_pin_counting(self):
        """
        Demonstrates page allocation, pin count increments on fetch,
        and decrements on unpin.
        """
        bpm = BufferPoolManager(self.disk_manager, pool_size=3)

        # Allocate brand new page 0
        page0 = bpm.new_page()
        page0_id = page0.page_id
        self.assertEqual(page0_id, 0)
        self.assertEqual(bpm.get_pin_count(page0_id), 1)
        self.assertFalse(bpm.is_dirty(page0_id))
        self.assertIn(page0_id, bpm)

        # Fetch page 0 while already in pool: pin count increments
        fetched_p0 = bpm.fetch_page(page0_id)
        self.assertIs(fetched_p0, page0)  # Same in-memory object
        self.assertEqual(bpm.get_pin_count(page0_id), 2)

        # First unpin: decrements to 1 (still pinned, not in replacer)
        self.assertTrue(bpm.unpin_page(page0_id, is_dirty=False))
        self.assertEqual(bpm.get_pin_count(page0_id), 1)
        frame_id = bpm.page_table[page0_id]
        self.assertNotIn(frame_id, bpm.replacer)

        # Second unpin: decrements to 0 (unpinned, now in replacer)
        self.assertTrue(bpm.unpin_page(page0_id, is_dirty=False))
        self.assertEqual(bpm.get_pin_count(page0_id), 0)
        self.assertIn(frame_id, bpm.replacer)

        # Unpinning already unpinned page returns False
        self.assertFalse(bpm.unpin_page(page0_id, is_dirty=False))
        bpm.close()

    def test_dirty_page_eviction_and_persistence(self):
        """
        Demonstrates:
        1. Modifying a page and marking it dirty on unpin.
        2. Evicting the dirty page to disk when memory limit (pool_size) is reached.
        3. Fetching the evicted page back from disk and verifying modified data is intact.
        """
        bpm = BufferPoolManager(self.disk_manager, pool_size=2)

        # 1. Allocate Page 0 and write data into it
        p0 = bpm.new_page()
        p0_id = p0.page_id
        slot_0 = p0.insert_record(b"Querix_Record_A_Payload")
        slot_1 = p0.insert_record(b"Querix_Record_B_Payload")

        # Unpin Page 0 as dirty
        bpm.unpin_page(p0_id, is_dirty=True)
        self.assertTrue(bpm.is_dirty(p0_id))
        self.assertEqual(bpm.get_pin_count(p0_id), 0)

        # 2. Allocate Page 1 (uses second available frame in pool)
        p1 = bpm.new_page()
        p1_id = p1.page_id
        bpm.unpin_page(p1_id, is_dirty=False)

        # Both Page 0 and Page 1 are cached; pool is at capacity (pool_size=2)
        self.assertEqual(len(bpm), 2)
        self.assertIn(p0_id, bpm)
        self.assertIn(p1_id, bpm)

        # 3. Allocate Page 2: triggers eviction of LRU page (Page 0)
        # Because Page 0 was dirty, it MUST be flushed to disk before eviction!
        p2 = bpm.new_page()
        p2_id = p2.page_id

        # Verify Page 0 is no longer cached in memory
        self.assertNotIn(p0_id, bpm)
        self.assertIn(p1_id, bpm)
        self.assertIn(p2_id, bpm)

        # Unpin Page 2
        bpm.unpin_page(p2_id, is_dirty=False)

        # 4. Fetch Page 0 back into the pool from disk!
        # This will evict Page 1 (the next LRU victim)
        restored_p0 = bpm.fetch_page(p0_id)
        self.assertIn(p0_id, bpm)
        self.assertEqual(restored_p0.page_id, p0_id)
        self.assertEqual(restored_p0.get_record(slot_0), b"Querix_Record_A_Payload")
        self.assertEqual(restored_p0.get_record(slot_1), b"Querix_Record_B_Payload")
        self.assertEqual(len(restored_p0), 2)

        bpm.unpin_page(p0_id, is_dirty=False)
        bpm.close()

    def test_flush_page_writes_regardless_of_pin_count(self):
        """
        Demonstrates flush_page forcibly writes dirty pages to disk
        even while pin_count > 0, clearing is_dirty flag without evicting.
        """
        bpm = BufferPoolManager(self.disk_manager, pool_size=2)

        page = bpm.new_page()
        page_id = page.page_id
        page.insert_record(b"Immediate Flushed Content")

        # Mark frame dirty while still holding pin_count=1
        frame = bpm.get_page_frame(page_id)
        self.assertIsNotNone(frame)
        frame.is_dirty = True

        self.assertEqual(bpm.get_pin_count(page_id), 1)
        self.assertTrue(bpm.is_dirty(page_id))

        # Forcibly flush page
        self.assertTrue(bpm.flush_page(page_id))

        # Frame should now be clean, but pin count remains 1
        self.assertFalse(bpm.is_dirty(page_id))
        self.assertEqual(bpm.get_pin_count(page_id), 1)
        self.assertIn(page_id, bpm)

        # Read directly from disk to confirm content was written
        raw_bytes = self.disk_manager.read_page(page_id)
        disk_page = SlottedPage.deserialize(raw_bytes)
        self.assertEqual(disk_page.get_record(0), b"Immediate Flushed Content")

        bpm.unpin_page(page_id, is_dirty=False)
        bpm.close()

    def test_buffer_pool_full_when_all_frames_pinned(self):
        """
        Verifies BufferPoolFullError is raised when trying to fetch/allocate
        when all frames in memory are pinned.
        """
        bpm = BufferPoolManager(self.disk_manager, pool_size=2)

        p0 = bpm.new_page()  # frame 0, pin=1
        p1 = bpm.new_page()  # frame 1, pin=1

        # All 2 frames are pinned; neither is in LRU replacer
        self.assertEqual(len(bpm.replacer), 0)

        # Attempting to allocate new page must raise BufferPoolFullError
        with self.assertRaises(BufferPoolFullError):
            bpm.new_page()

        # Unpin p0: makes frame 0 eligible for eviction
        bpm.unpin_page(p0.page_id, is_dirty=False)
        self.assertEqual(len(bpm.replacer), 1)

        # Now new_page should succeed by evicting p0
        p2 = bpm.new_page()
        self.assertEqual(p2.page_id, 2)
        self.assertNotIn(p0.page_id, bpm)
        self.assertIn(p2.page_id, bpm)

        bpm.unpin_page(p1.page_id, is_dirty=False)
        bpm.unpin_page(p2.page_id, is_dirty=False)
        bpm.close()

    def test_lru_eviction_order(self):
        """
        Verifies exact Least Recently Used eviction order across 3 frames.
        """
        bpm = BufferPoolManager(self.disk_manager, pool_size=3)

        p0 = bpm.new_page()
        p1 = bpm.new_page()
        p2 = bpm.new_page()

        # Unpin in order: p0, p1, p2
        bpm.unpin_page(p0.page_id, is_dirty=False)
        bpm.unpin_page(p1.page_id, is_dirty=False)
        bpm.unpin_page(p2.page_id, is_dirty=False)

        # Replacer order is now: [p0, p1, p2] (p0 is oldest)
        # Now access p0 again
        bpm.fetch_page(p0.page_id)
        bpm.unpin_page(p0.page_id, is_dirty=False)

        # p0 was re-unpinned, so replacer order is now: [p1, p2, p0]
        # Allocating p3 should evict p1!
        p3 = bpm.new_page()
        self.assertNotIn(p1.page_id, bpm)
        self.assertIn(p0.page_id, bpm)
        self.assertIn(p2.page_id, bpm)
        self.assertIn(p3.page_id, bpm)

        bpm.unpin_page(p3.page_id, is_dirty=False)
        bpm.close()

    def test_disk_manager_standalone(self):
        """Direct test of DiskManager random-access reads, writes, and allocations."""
        dm = DiskManager(self.db_path)
        self.assertEqual(dm.num_pages, 0)

        # Allocate page 0
        p0_id = dm.allocate_page()
        self.assertEqual(p0_id, 0)
        self.assertEqual(dm.num_pages, 1)

        # Allocate page 1
        p1_id = dm.allocate_page()
        self.assertEqual(p1_id, 1)
        self.assertEqual(dm.num_pages, 2)

        # Write custom page to page 1
        page1 = SlottedPage(page_id=1)
        slot = page1.insert_record(b"DiskManager Direct Test")
        dm.write_page(1, page1.serialize())

        # Read back page 1
        read_bytes = dm.read_page(1)
        restored = SlottedPage.deserialize(read_bytes)
        self.assertEqual(restored.page_id, 1)
        self.assertEqual(restored.get_record(slot), b"DiskManager Direct Test")

        # Invalid page read raises IOError
        with self.assertRaises(IOError):
            dm.read_page(99)

        dm.close()

    def test_context_manager_and_flexible_init(self):
        """Verifies flexible constructor signatures and context manager protocol."""
        # Using string path and default pool_size
        with BufferPoolManager(str(self.db_path), 5) as bpm:
            page = bpm.new_page()
            self.assertEqual(bpm.pool_size, 5)
            bpm.unpin_page(page.page_id, is_dirty=False)

        # Reopen with keyword arguments
        with BufferPoolManager(pool_size=10, disk_manager=self.disk_manager) as bpm:
            self.assertEqual(bpm.pool_size, 10)
            page = bpm.fetch_page(0)
            self.assertEqual(page.page_id, 0)
            bpm.unpin_page(0, is_dirty=False)


if __name__ == "__main__":
    unittest.main()
