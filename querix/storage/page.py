"""
Slotted Page Architecture for Querix Storage Engine.
Manages variable-length records within fixed-size 4096-byte (4KB) disk pages.

Meet-in-the-Middle Page Layout:
+-------------------------------------------------------------------------------+
| Page Header (10 bytes)                                                        |
|  - page_id (4B, !I)                                                           |
|  - lower_bound (2B, !H)                                                       |
|  - upper_bound (2B, !H)                                                       |
|  - num_slots (2B, !H)                                                         |
+-------------------------------------------------------------------------------+
| Slot Array (grows downward from lower_bound)                                  |
|  [Slot 0: offset (2B), length (2B)]                                           |
|  [Slot 1: offset (2B), length (2B)]                                           |
|  ...                                                                          |
|  [Slot N-1]                                                                   |
|  <-- lower_bound                                                              |
+-------------------------------------------------------------------------------+
|                          FREE SPACE AREA                                      |
|                 (size = upper_bound - lower_bound)                            |
+-------------------------------------------------------------------------------+
|  <-- upper_bound                                                              |
|  [Record Data N-1]                                                            |
|  ...                                                                          |
|  [Record Data 1]                                                              |
|  [Record Data 0]                                                              |
+-------------------------------------------------------------------------------+
"""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

PAGE_SIZE = 4096

# Header Format:
# ! = Big-endian (network byte order)
# I = 4-byte unsigned int (page_id)
# H = 2-byte unsigned short (lower_bound)
# H = 2-byte unsigned short (upper_bound)
# H = 2-byte unsigned short (num_slots)
HEADER_FORMAT = "!IHHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 10 bytes

# Slot Format:
# H = 2-byte unsigned short (offset where record begins)
# H = 2-byte unsigned short (length of record bytes)
SLOT_FORMAT = "!HH"
SLOT_SIZE = struct.calcsize(SLOT_FORMAT)  # 4 bytes

DELETED_SLOT_OFFSET = 0
DELETED_SLOT_LENGTH = 0


class PageError(Exception):
    """Base class for slotted page errors."""
    pass


class PageFullError(PageError):
    """Raised when there is not enough free space on the page to insert a record."""
    pass


class SlotNotFoundError(PageError, KeyError):
    """Raised when accessing a non-existent or deleted slot."""
    pass


class SlottedPage:
    """
    Fixed-size 4KB slotted page implementing the meet-in-the-middle layout.
    """

    def __init__(
        self,
        page_id: int,
        data: Optional[bytearray] = None,
    ) -> None:
        """
        Initializes a SlottedPage.

        :param page_id: Unique 32-bit identifier for the page.
        :param data: Optional pre-existing 4096-byte buffer (used during deserialization).
        """
        if data is not None:
            if len(data) != PAGE_SIZE:
                raise ValueError(f"Page data must be exactly {PAGE_SIZE} bytes, got {len(data)}")
            self._data = data
            (
                self.page_id,
                self.lower_bound,
                self.upper_bound,
                self.num_slots,
            ) = struct.unpack_from(HEADER_FORMAT, self._data, 0)
        else:
            self._data = bytearray(PAGE_SIZE)
            self.page_id = page_id
            self.lower_bound = HEADER_SIZE
            self.upper_bound = PAGE_SIZE
            self.num_slots = 0
            self._sync_header()
        self.page_lsn: int = 0

    def _sync_header(self) -> None:
        """Writes current header values into the underlying bytearray buffer."""
        struct.pack_into(
            HEADER_FORMAT,
            self._data,
            0,
            self.page_id,
            self.lower_bound,
            self.upper_bound,
            self.num_slots,
        )

    @property
    def free_space(self) -> int:
        """Returns the number of unallocated contiguous bytes between slot array and data tuples."""
        return self.upper_bound - self.lower_bound

    def _get_slot(self, slot_id: int) -> Tuple[int, int]:
        """Returns (offset, length) for the given slot ID."""
        if slot_id < 0 or slot_id >= self.num_slots:
            raise SlotNotFoundError(f"Slot ID {slot_id} out of bounds (num_slots={self.num_slots})")
        slot_pos = HEADER_SIZE + slot_id * SLOT_SIZE
        offset, length = struct.unpack_from(SLOT_FORMAT, self._data, slot_pos)
        return offset, length

    def _set_slot(self, slot_id: int, offset: int, length: int) -> None:
        """Writes (offset, length) into slot array for slot_id."""
        slot_pos = HEADER_SIZE + slot_id * SLOT_SIZE
        struct.pack_into(SLOT_FORMAT, self._data, slot_pos, offset, length)

    def is_valid_slot(self, slot_id: int) -> bool:
        """Checks if a slot exists and contains active (non-deleted) data."""
        if slot_id < 0 or slot_id >= self.num_slots:
            return False
        offset, length = self._get_slot(slot_id)
        return offset != DELETED_SLOT_OFFSET and length != DELETED_SLOT_LENGTH

    def insert_record(self, record_bytes: bytes) -> int:
        """
        Inserts a variable-length record into the page.

        Checks free space (record length + 4 bytes for new slot).
        Stores tuple data at upper_bound, adds slot at lower_bound, and returns slot_id.

        :param record_bytes: Raw bytes to store.
        :return: Allocated slot_id (0-indexed).
        :raises PageFullError: If page has insufficient contiguous free space.
        """
        record_len = len(record_bytes)
        needed_space = record_len + SLOT_SIZE

        if self.free_space < needed_space:
            raise PageFullError(
                f"Page {self.page_id} full: needed {needed_space} bytes, only {self.free_space} available"
            )

        # 1. Allocate tuple space from top down
        new_upper_bound = self.upper_bound - record_len
        self._data[new_upper_bound : self.upper_bound] = record_bytes
        self.upper_bound = new_upper_bound

        # 2. Allocate slot in slot array from bottom up
        slot_id = self.num_slots
        self._set_slot(slot_id, offset=self.upper_bound, length=record_len)

        # 3. Advance lower bound and slot count
        self.lower_bound += SLOT_SIZE
        self.num_slots += 1

        # 4. Sync header state
        self._sync_header()
        return slot_id

    def get_record(self, slot_id: int) -> bytes:
        """
        Reads and returns the raw record bytes for the specified slot ID.

        :param slot_id: 0-indexed slot identifier.
        :return: Record payload bytes.
        :raises SlotNotFoundError: If slot does not exist or has been deleted.
        """
        offset, length = self._get_slot(slot_id)
        if offset == DELETED_SLOT_OFFSET and length == DELETED_SLOT_LENGTH:
            raise SlotNotFoundError(f"Slot {slot_id} in page {self.page_id} has been deleted")

        return bytes(self._data[offset : offset + length])

    def delete_record(self, slot_id: int) -> None:
        """
        Marks slot as deleted and defragments the page by shifting surviving data upward.

        :param slot_id: Slot to delete.
        :raises SlotNotFoundError: If slot does not exist or is already deleted.
        """
        target_offset, target_length = self._get_slot(slot_id)
        if target_offset == DELETED_SLOT_OFFSET and target_length == DELETED_SLOT_LENGTH:
            raise SlotNotFoundError(f"Slot {slot_id} is already deleted in page {self.page_id}")

        # Defragmentation: Shift data residing below target_offset upward
        if target_offset > self.upper_bound:
            shift_len = target_offset - self.upper_bound
            # Slices to shift
            data_to_shift = self._data[self.upper_bound : target_offset]
            self._data[self.upper_bound + target_length : target_offset + target_length] = data_to_shift

        # Zero out the vacated memory
        self._data[self.upper_bound : self.upper_bound + target_length] = b"\x00" * target_length
        self.upper_bound += target_length

        # Update offsets of all other active slots that were shifted
        for other_id in range(self.num_slots):
            if other_id == slot_id:
                continue
            o_offset, o_len = self._get_slot(other_id)
            if o_offset != DELETED_SLOT_OFFSET and o_offset < target_offset:
                self._set_slot(other_id, o_offset + target_length, o_len)

        # Mark deleted slot
        self._set_slot(slot_id, DELETED_SLOT_OFFSET, DELETED_SLOT_LENGTH)

        # Trim trailing deleted slots to reclaim slot array space
        while self.num_slots > 0:
            last_offset, last_length = self._get_slot(self.num_slots - 1)
            if last_offset == DELETED_SLOT_OFFSET and last_length == DELETED_SLOT_LENGTH:
                self.num_slots -= 1
                self.lower_bound -= SLOT_SIZE
            else:
                break

        self._sync_header()

    def serialize(self) -> bytes:
        """
        Packs the complete in-memory page state into exactly 4096 bytes.
        """
        self._sync_header()
        return bytes(self._data)

    @classmethod
    def deserialize(cls, data: bytes) -> SlottedPage:
        """
        Reconstructs a SlottedPage instance from a 4096-byte disk page buffer.

        :param data: 4096-byte raw data read from disk.
        :return: Reconstructed SlottedPage.
        """
        if len(data) != PAGE_SIZE:
            raise ValueError(f"Expected {PAGE_SIZE} bytes for page, got {len(data)}")
        return cls(page_id=0, data=bytearray(data))

    def get_all_records(self) -> Dict[int, bytes]:
        """Returns dictionary of active {slot_id: record_bytes}."""
        records: Dict[int, bytes] = {}
        for s_id in range(self.num_slots):
            if self.is_valid_slot(s_id):
                records[s_id] = self.get_record(s_id)
        return records

    def __len__(self) -> int:
        """Returns the count of active records."""
        count = 0
        for s_id in range(self.num_slots):
            if self.is_valid_slot(s_id):
                count += 1
        return count
