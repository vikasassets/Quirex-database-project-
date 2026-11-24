"""
Disk Manager for Querix Storage Engine with Multi-Segment Support.
Manages active and frozen segments, automatic threshold-based rolling,
durability (fsync), random-seek reads, and multi-segment crash recovery.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

from querix.storage.page import PAGE_SIZE, SlottedPage
from querix.storage.record import (
    HEADER_SIZE,
    TOMBSTONE_VAL_LEN,
    CorruptedRecordError,
    IncompleteRecordError,
    decode_header,
    encode_record,
    verify_and_decode_record,
)

SEGMENT_FILE_PATTERN = re.compile(r"^segment_(\d{4,})\.qrx$")
DEFAULT_MAX_SEGMENT_SIZE = 10 * 1024 * 1024  # 10 MB


class Segment:
    """
    Represents an individual disk segment file (e.g., segment_0001.qrx).
    Can be in ACTIVE (read-write) or FROZEN (read-only) mode.
    """

    def __init__(
        self,
        segment_id: int,
        file_path: Path,
        is_active: bool = False,
        sync_on_write: bool = True,
    ) -> None:
        self.segment_id = segment_id
        self.file_path = file_path.resolve()
        self.is_active = is_active
        self.sync_on_write = sync_on_write

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()

        # If active, open write handle in append mode and read handle in read mode
        self._write_file = open(self.file_path, "a+b") if is_active else None
        self._read_file = open(self.file_path, "rb")

    @property
    def size(self) -> int:
        """Returns the current size of the segment file in bytes."""
        if self._write_file is not None and not self._write_file.closed:
            self._write_file.flush()
        return self.file_path.stat().st_size

    def append(self, key: bytes, value: Optional[bytes]) -> Tuple[int, int, int]:
        """
        Appends a record to the segment. Only permitted if active.

        :return: (record_offset, record_size, val_len)
        """
        if not self.is_active or self._write_file is None:
            raise RuntimeError(f"Cannot append to frozen segment {self.segment_id}")

        record_bytes = encode_record(key, value)
        record_size = len(record_bytes)
        val_len = TOMBSTONE_VAL_LEN if value is None else len(value)

        self._write_file.seek(0, os.SEEK_END)
        offset = self._write_file.tell()

        self._write_file.write(record_bytes)
        self._write_file.flush()

        if self.sync_on_write:
            os.fsync(self._write_file.fileno())

        return offset, record_size, val_len

    def read_record_at(self, offset: int) -> Tuple[bytes, Optional[bytes], int]:
        """Reads and validates the full record at the specified byte offset."""
        self._read_file.seek(offset)
        header_bytes = self._read_file.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            raise IncompleteRecordError(
                f"Cannot read full header at offset {offset} in segment {self.segment_id}"
            )

        crc, timestamp_ns, key_len, val_len = decode_header(header_bytes)
        payload_len = key_len + max(0, val_len)
        payload_bytes = self._read_file.read(payload_len)

        full_record_bytes = header_bytes + payload_bytes
        return verify_and_decode_record(full_record_bytes)

    def read_value_direct(self, offset: int, key_len: int, val_len: int) -> bytes:
        """Fast path: seeks directly to value bytes within this segment."""
        if val_len == TOMBSTONE_VAL_LEN:
            raise KeyError("Requested value is a tombstone")

        value_offset = offset + HEADER_SIZE + key_len
        self._read_file.seek(value_offset)
        val_bytes = self._read_file.read(val_len)

        if len(val_bytes) != val_len:
            raise IncompleteRecordError(
                f"Expected {val_len} value bytes at offset {value_offset} in segment {self.segment_id}, got {len(val_bytes)}"
            )
        return val_bytes

    def scan_records(
        self,
    ) -> Generator[Tuple[int, int, bytes, Optional[bytes], int], None, None]:
        """
        Sequentially scans all valid records from offset 0 to EOF in this segment.
        Yields: (record_offset, record_size, key, value, timestamp_ns)
        """
        with open(self.file_path, "rb") as scan_file:
            while True:
                offset = scan_file.tell()
                header_bytes = scan_file.read(HEADER_SIZE)

                if not header_bytes or len(header_bytes) < HEADER_SIZE:
                    break

                try:
                    crc, timestamp_ns, key_len, val_len = decode_header(header_bytes)
                    payload_len = key_len + max(0, val_len)
                    payload_bytes = scan_file.read(payload_len)

                    if len(payload_bytes) < payload_len:
                        break

                    key, value, ts = verify_and_decode_record(header_bytes + payload_bytes)
                    record_size = HEADER_SIZE + payload_len
                    yield offset, record_size, key, value, ts

                except (CorruptedRecordError, IncompleteRecordError):
                    break

    def freeze(self) -> None:
        """Transitions this segment to frozen (read-only) mode and closes write handle."""
        if self._write_file is not None and not self._write_file.closed:
            self._write_file.flush()
            if self.sync_on_write:
                os.fsync(self._write_file.fileno())
            self._write_file.close()
            self._write_file = None
        self.is_active = False

    def close(self) -> None:
        """Closes all file handles for this segment."""
        self.freeze()
        if self._read_file is not None and not self._read_file.closed:
            self._read_file.close()


class DiskStorage:
    """
    Manages a collection of segment files within a database directory.
    Maintains one active segment for appends and zero or more frozen segments for reads.
    Handles automatic segment rolling when the active segment reaches max_segment_size.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_segment_size: int = DEFAULT_MAX_SEGMENT_SIZE,
        sync_on_write: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.max_segment_size = max_segment_size
        self.sync_on_write = sync_on_write

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._segments: Dict[int, Segment] = {}
        self._active_segment: Optional[Segment] = None

        self._discover_and_initialize_segments()

    @property
    def active_segment(self) -> Segment:
        """Returns the currently active segment."""
        if self._active_segment is None:
            raise RuntimeError("No active segment available.")
        return self._active_segment

    @property
    def active_segment_id(self) -> int:
        return self.active_segment.segment_id

    @property
    def active_segment_path(self) -> Path:
        return self.active_segment.file_path

    @property
    def total_size(self) -> int:
        """Total size in bytes across all segments."""
        return sum(seg.size for seg in self._segments.values())

    def _segment_file_path(self, segment_id: int) -> Path:
        return self.data_dir / f"segment_{segment_id:04d}.qrx"

    def _discover_and_initialize_segments(self) -> None:
        """
        Discovers existing segment files in data_dir, sorting them by segment ID.
        The segment with the highest ID is reopened as ACTIVE; all prior segments are FROZEN.
        """
        discovered_ids: List[int] = []
        for file in self.data_dir.iterdir():
            match = SEGMENT_FILE_PATTERN.match(file.name)
            if match:
                discovered_ids.append(int(match.group(1)))

        discovered_ids.sort()

        if not discovered_ids:
            # First initialization: start with segment_0001.qrx
            initial_id = 1
            seg = Segment(
                segment_id=initial_id,
                file_path=self._segment_file_path(initial_id),
                is_active=True,
                sync_on_write=self.sync_on_write,
            )
            self._segments[initial_id] = seg
            self._active_segment = seg
        else:
            # Reopen existing segments: highest ID is active, rest are frozen
            active_id = discovered_ids[-1]
            for seg_id in discovered_ids:
                is_active = (seg_id == active_id)
                seg = Segment(
                    segment_id=seg_id,
                    file_path=self._segment_file_path(seg_id),
                    is_active=is_active,
                    sync_on_write=self.sync_on_write,
                )
                self._segments[seg_id] = seg
                if is_active:
                    self._active_segment = seg

    def roll_active_segment(self) -> Segment:
        """
        Freezes the current active segment and creates a new active segment with the next ID.

        :return: Newly created active Segment.
        """
        curr = self.active_segment
        curr.freeze()

        new_id = curr.segment_id + 1
        new_seg = Segment(
            segment_id=new_id,
            file_path=self._segment_file_path(new_id),
            is_active=True,
            sync_on_write=self.sync_on_write,
        )
        self._segments[new_id] = new_seg
        self._active_segment = new_seg
        return new_seg

    def append(self, key: bytes, value: Optional[bytes]) -> Tuple[int, int, int, int]:
        """
        Appends a record to the active segment. Automatically rolls to a new segment
        if the active segment has reached or exceeded max_segment_size.

        :return: (segment_id, offset, record_size, val_len)
        """
        if self.active_segment.size >= self.max_segment_size:
            self.roll_active_segment()

        seg = self.active_segment
        offset, record_size, val_len = seg.append(key, value)
        return seg.segment_id, offset, record_size, val_len

    def read_value_direct(
        self, segment_id: int, offset: int, key_len: int, val_len: int
    ) -> bytes:
        """Reads value bytes directly from the specified segment."""
        seg = self._segments.get(segment_id)
        if seg is None:
            raise KeyError(f"Segment {segment_id} not found in disk storage")
        return seg.read_value_direct(offset, key_len, val_len)

    def read_record_at(
        self, segment_id: int, offset: int
    ) -> Tuple[bytes, Optional[bytes], int]:
        """Reads and validates full record at offset in the specified segment."""
        seg = self._segments.get(segment_id)
        if seg is None:
            raise KeyError(f"Segment {segment_id} not found in disk storage")
        return seg.read_record_at(offset)

    def scan_all_segments(
        self,
    ) -> Generator[Tuple[int, int, int, bytes, Optional[bytes], int], None, None]:
        """
        Sequentially scans all records across all segments in ascending chronological order.
        Yields: (segment_id, offset, record_size, key, value, timestamp_ns)
        """
        for seg_id in sorted(self._segments.keys()):
            seg = self._segments[seg_id]
            for offset, record_size, key, value, ts in seg.scan_records():
                yield seg_id, offset, record_size, key, value, ts

    def get_frozen_segments(self) -> List[Segment]:
        """Returns all frozen (read-only) segments sorted by segment ID."""
        return [
            seg for seg_id, seg in sorted(self._segments.items())
            if not seg.is_active
        ]

    def remove_segment(self, segment_id: int) -> None:
        """
        Safely closes, unregisters, and deletes a segment file from disk.
        Cannot remove the active segment.
        """
        if segment_id == self.active_segment_id:
            raise RuntimeError(f"Cannot remove active segment {segment_id}")

        seg = self._segments.pop(segment_id, None)
        if seg is not None:
            seg.close()
            if seg.file_path.exists():
                os.remove(seg.file_path)

    def register_compacted_segment(self, segment: Segment) -> None:
        """Registers a newly compacted frozen segment in the segment registry."""
        self._segments[segment.segment_id] = segment

    def allocate_page(self) -> int:
        """Allocates a 4KB page in the managed pages database file."""
        if not hasattr(self, "_page_manager") or self._page_manager is None:
            self._page_manager = DiskManager(self.data_dir / "pages.db", sync_on_write=self.sync_on_write)
        return self._page_manager.allocate_page()

    def write_page(self, page_id: int, page_data: Union[bytes, bytearray]) -> None:
        """Writes a 4KB page buffer to the managed pages database file."""
        if not hasattr(self, "_page_manager") or self._page_manager is None:
            self._page_manager = DiskManager(self.data_dir / "pages.db", sync_on_write=self.sync_on_write)
        self._page_manager.write_page(page_id, page_data)

    def read_page(self, page_id: int) -> bytes:
        """Reads a 4KB page buffer from the managed pages database file."""
        if not hasattr(self, "_page_manager") or self._page_manager is None:
            self._page_manager = DiskManager(self.data_dir / "pages.db", sync_on_write=self.sync_on_write)
        return self._page_manager.read_page(page_id)

    def close(self) -> None:
        """Flushes and closes all open segment files and page database handles."""
        if hasattr(self, "_page_manager") and self._page_manager is not None:
            self._page_manager.close()
            self._page_manager = None
        for seg in self._segments.values():
            seg.close()


class DiskManager:
    """
    Disk Manager for fixed-size 4096-byte (4KB) slotted pages.
    Provides random-access page reads, page writes, page allocation,
    and durable synchronous disk writes (fsync).
    """

    def __init__(
        self,
        db_file: Union[str, Path],
        sync_on_write: bool = True,
    ) -> None:
        self.file_path = Path(db_file).resolve()
        self.sync_on_write = sync_on_write

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()

        # Open in read-write binary mode without truncation
        self._file = open(self.file_path, "r+b")
        self._file_size = self.file_path.stat().st_size
        self._num_pages = self._file_size // PAGE_SIZE

    @property
    def num_pages(self) -> int:
        """Returns the number of 4KB pages allocated on disk."""
        return self._num_pages

    def allocate_page(self) -> int:
        """
        Allocates a new page on disk, initializes it with a blank SlottedPage header,
        and returns the newly allocated page_id.

        :return: Allocated 0-indexed page_id.
        """
        page_id = self._num_pages
        blank_page = SlottedPage(page_id=page_id).serialize()
        self.write_page(page_id, blank_page)
        return page_id

    def write_page(self, page_id: int, page_data: Union[bytes, bytearray]) -> None:
        """
        Writes exactly PAGE_SIZE (4096) bytes to the specified page_id offset.

        :param page_id: Target page identifier.
        :param page_data: 4096-byte serialized page buffer.
        """
        if len(page_data) != PAGE_SIZE:
            raise ValueError(f"Page data must be exactly {PAGE_SIZE} bytes, got {len(page_data)}")

        offset = page_id * PAGE_SIZE
        self._file.seek(offset)
        self._file.write(page_data)
        self._file.flush()

        if self.sync_on_write:
            os.fsync(self._file.fileno())

        if page_id >= self._num_pages:
            self._num_pages = page_id + 1

    def read_page(self, page_id: int) -> bytes:
        """
        Reads exactly PAGE_SIZE (4096) bytes for the specified page_id.

        :param page_id: Target page identifier.
        :return: 4096 bytes read from disk.
        :raises IOError: If page cannot be read or is truncated.
        """
        if page_id < 0:
            raise ValueError(f"Invalid page_id: {page_id}")

        offset = page_id * PAGE_SIZE
        self._file.seek(offset)
        data = self._file.read(PAGE_SIZE)

        if len(data) < PAGE_SIZE:
            raise IOError(
                f"Page {page_id} not found or incomplete on disk (expected {PAGE_SIZE} bytes, got {len(data)})"
            )
        return data

    def close(self) -> None:
        """Flushes and closes the database page file handle."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            if self.sync_on_write:
                os.fsync(self._file.fileno())
            self._file.close()

    def __enter__(self) -> DiskManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

