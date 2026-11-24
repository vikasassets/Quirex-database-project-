"""
Querix Storage Subsystem
Handles low-level record framing, disk I/O, segmentation, compaction, and slotted pages.
"""

from querix.storage.disk import DiskManager, DiskStorage, Segment
from querix.storage.page import (
    HEADER_SIZE as PAGE_HEADER_SIZE,
    PAGE_SIZE,
    SLOT_SIZE,
    PageError,
    PageFullError,
    SlottedPage,
    SlotNotFoundError,
)
from querix.storage.record import (
    CorruptedRecordError,
    IncompleteRecordError,
    RecordError,
    decode_header,
    encode_record,
    verify_and_decode_record,
)

__all__ = [
    "DiskManager",
    "DiskStorage",
    "Segment",
    "SlottedPage",
    "PAGE_SIZE",
    "PAGE_HEADER_SIZE",
    "SLOT_SIZE",
    "PageError",
    "PageFullError",
    "SlotNotFoundError",
    "RecordError",
    "CorruptedRecordError",
    "IncompleteRecordError",
    "encode_record",
    "decode_header",
    "verify_and_decode_record",
]
