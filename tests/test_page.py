"""
Unit tests for SlottedPage architecture.
Verifies meet-in-the-middle layout, insertion, retrieval, defragmentation on deletion,
page-full error handling, and binary serialization/deserialization.
"""

import unittest

from querix.storage.page import (
    HEADER_SIZE,
    PAGE_SIZE,
    SLOT_SIZE,
    PageFullError,
    SlottedPage,
    SlotNotFoundError,
)


class TestSlottedPage(unittest.TestCase):

    def test_page_initial_state(self):
        page = SlottedPage(page_id=42)
        self.assertEqual(page.page_id, 42)
        self.assertEqual(page.lower_bound, HEADER_SIZE)
        self.assertEqual(page.upper_bound, PAGE_SIZE)
        self.assertEqual(page.num_slots, 0)
        self.assertEqual(page.free_space, PAGE_SIZE - HEADER_SIZE)
        self.assertEqual(len(page), 0)

    def test_insert_and_get_records(self):
        page = SlottedPage(page_id=1)

        rec0 = b"First record payload"
        rec1 = b"Short"
        rec2 = b"A much longer record containing JSON structured data: {'id': 100, 'val': 'test'}"

        slot0 = page.insert_record(rec0)
        slot1 = page.insert_record(rec1)
        slot2 = page.insert_record(rec2)

        self.assertEqual(slot0, 0)
        self.assertEqual(slot1, 1)
        self.assertEqual(slot2, 2)
        self.assertEqual(page.num_slots, 3)
        self.assertEqual(len(page), 3)

        self.assertEqual(page.get_record(slot0), rec0)
        self.assertEqual(page.get_record(slot1), rec1)
        self.assertEqual(page.get_record(slot2), rec2)

        expected_used = (
            HEADER_SIZE
            + 3 * SLOT_SIZE
            + len(rec0)
            + len(rec1)
            + len(rec2)
        )
        self.assertEqual(page.free_space, PAGE_SIZE - expected_used)

    def test_delete_middle_record_defragments_space(self):
        """
        Inserts 3 records: A, B, C.
        Deletes middle record B.
        Verifies:
        1. Record A and C remain completely accessible.
        2. Attempting to read B raises SlotNotFoundError.
        3. Surviving data (C) is shifted upward and free space increases by len(B).
        """
        page = SlottedPage(page_id=10)

        data_a = b"AAAA_RECORD_A_16B"
        data_b = b"BBBB_RECORD_B_16B"
        data_c = b"CCCC_RECORD_C_16B"

        slot_a = page.insert_record(data_a)
        slot_b = page.insert_record(data_b)
        slot_c = page.insert_record(data_c)

        space_before = page.free_space

        # Delete middle record B
        page.delete_record(slot_b)

        # Free space should increase by exactly len(data_b)
        space_after = page.free_space
        self.assertEqual(space_after, space_before + len(data_b))

        # Check records
        self.assertEqual(page.get_record(slot_a), data_a)
        self.assertEqual(page.get_record(slot_c), data_c)

        with self.assertRaises(SlotNotFoundError):
            page.get_record(slot_b)

        self.assertFalse(page.is_valid_slot(slot_b))
        self.assertTrue(page.is_valid_slot(slot_a))
        self.assertTrue(page.is_valid_slot(slot_c))
        self.assertEqual(len(page), 2)

    def test_delete_last_record_trims_slot_array(self):
        """
        Deleting the last slot in the slot array reclaims both
        the record data bytes and the 4-byte slot overhead.
        """
        page = SlottedPage(page_id=20)
        slot_a = page.insert_record(b"Alpha")
        slot_b = page.insert_record(b"Beta")

        space_before = page.free_space
        page.delete_record(slot_b)

        # Reclaims len(b"Beta") (4 bytes) + SLOT_SIZE (4 bytes) = 8 bytes
        self.assertEqual(page.free_space, space_before + len(b"Beta") + SLOT_SIZE)
        self.assertEqual(page.num_slots, 1)
        self.assertEqual(page.get_record(slot_a), b"Alpha")

    def test_serialize_and_deserialize(self):
        """
        Verifies that serializing to 4096 bytes and deserializing back
        reconstructs the exact page state, slots, and tuples.
        """
        original = SlottedPage(page_id=99)
        original.insert_record(b"Record 1")
        slot_to_del = original.insert_record(b"Record 2 (To be deleted)")
        original.insert_record(b"Record 3")

        original.delete_record(slot_to_del)

        # Serialize
        raw_bytes = original.serialize()
        self.assertEqual(len(raw_bytes), PAGE_SIZE)

        # Deserialize
        restored = SlottedPage.deserialize(raw_bytes)
        self.assertEqual(restored.page_id, 99)
        self.assertEqual(restored.lower_bound, original.lower_bound)
        self.assertEqual(restored.upper_bound, original.upper_bound)
        self.assertEqual(restored.num_slots, original.num_slots)
        self.assertEqual(restored.free_space, original.free_space)

        self.assertEqual(restored.get_record(0), b"Record 1")
        self.assertEqual(restored.get_record(2), b"Record 3")
        with self.assertRaises(SlotNotFoundError):
            restored.get_record(1)

    def test_page_full_raises_error(self):
        page = SlottedPage(page_id=7)
        # Attempt to insert payload larger than total page capacity
        oversized = b"X" * (PAGE_SIZE - HEADER_SIZE)
        with self.assertRaises(PageFullError):
            page.insert_record(oversized)

    def test_invalid_slot_id_raises_error(self):
        page = SlottedPage(page_id=1)
        with self.assertRaises(SlotNotFoundError):
            page.get_record(0)

        with self.assertRaises(SlotNotFoundError):
            page.get_record(-1)


if __name__ == "__main__":
    unittest.main()
