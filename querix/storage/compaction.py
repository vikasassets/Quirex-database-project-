"""
Log Compaction and Segment Merging Subsystem for Querix.
Reclaims disk space by purging overwritten versions and tombstones
from frozen segments, producing a single merged, compact segment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from querix.index.memory_index import IndexEntry, MemoryIndex
from querix.storage.disk import DiskStorage, Segment
from querix.storage.record import encode_record


@dataclass(frozen=True)
class CompactionResult:
    """Statistics returned after a compaction run."""
    segments_compacted: int
    records_scanned: int
    records_retained: int
    records_dropped: int
    bytes_before: int
    bytes_after: int

    @property
    def bytes_reclaimed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


class Compactor:
    """
    Executes garbage collection across frozen segment files.
    Identifies live records using the in-memory KeyDir, writes them to
    a fresh merged segment, atomically swaps index pointers, and deletes
    the obsolete uncompacted segment files.
    """

    def __init__(self, disk: DiskStorage, index: MemoryIndex) -> None:
        self.disk = disk
        self.index = index

    def compact(self, roll_active_if_needed: bool = True) -> CompactionResult:
        """
        Executes a compaction cycle.

        :param roll_active_if_needed: If True and there are no frozen segments,
                                      rolls the active segment so historical writes
                                      can be compacted immediately.
        :return: CompactionResult with bytes and record stats.
        """
        if roll_active_if_needed and self.disk.active_segment.size > 0:
            self.disk.roll_active_segment()

        frozen_segments = self.disk.get_frozen_segments()

        if not frozen_segments:
            return CompactionResult(
                segments_compacted=0,
                records_scanned=0,
                records_retained=0,
                records_dropped=0,
                bytes_before=0,
                bytes_after=0,
            )

        compacted_segment_ids = [seg.segment_id for seg in frozen_segments]
        bytes_before = sum(seg.size for seg in frozen_segments)

        # Temporary merged segment file path
        tmp_merged_path = self.disk.data_dir / "segment_compacting.tmp"
        if tmp_merged_path.exists():
            os.remove(tmp_merged_path)

        records_scanned = 0
        records_retained = 0
        records_dropped = 0

        # Mapping of key -> new IndexEntry in the merged segment
        # We will determine the new segment ID once old segments are freed
        new_entries: Dict[bytes, Tuple[int, int, int, int, int]] = {}

        # Scan each frozen segment and copy only live data
        with open(tmp_merged_path, "wb") as tmp_file:
            for seg in frozen_segments:
                for offset, rec_size, key, value, ts in seg.scan_records():
                    records_scanned += 1

                    # Check current index entry for this key
                    current_entry = self.index.get(key)

                    # A record is valid LIVE data iff:
                    # 1. It is currently in the index.
                    # 2. It points to this exact segment ID and this exact file offset.
                    # 3. It is not a tombstone (value is not None).
                    is_live = (
                        current_entry is not None
                        and current_entry.segment_id == seg.segment_id
                        and current_entry.file_offset == offset
                        and value is not None
                    )

                    if is_live:
                        # Live record: append to merged file
                        record_bytes = encode_record(key, value, timestamp_ns=ts)
                        new_offset = tmp_file.tell()
                        tmp_file.write(record_bytes)
                        records_retained += 1

                        new_entries[key] = (
                            new_offset,
                            len(record_bytes),
                            len(key),
                            len(value),
                            ts,
                        )
                    else:
                        records_dropped += 1

            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        # The new merged segment takes the lowest ID among the compacted segments
        new_segment_id = min(compacted_segment_ids)
        target_segment_path = self.disk._segment_file_path(new_segment_id)

        # Step 1: Safely close and delete all old frozen segment files
        for seg_id in compacted_segment_ids:
            self.disk.remove_segment(seg_id)

        # Step 2: Atomic rename of the merged temp file to target segment name
        if target_segment_path.exists():
            os.remove(target_segment_path)
        os.replace(tmp_merged_path, target_segment_path)

        # Step 3: Register the new compacted frozen segment in disk manager
        new_segment = Segment(
            segment_id=new_segment_id,
            file_path=target_segment_path,
            is_active=False,
            sync_on_write=self.disk.sync_on_write,
        )
        self.disk.register_compacted_segment(new_segment)
        bytes_after = new_segment.size

        # Step 4: Safely update in-memory index
        for key, (new_offset, rec_size, k_len, v_len, ts) in new_entries.items():
            current_entry = self.index.get(key)
            # Concurrency / invariant check: only update if the key still points to
            # one of the compacted segment IDs (i.e. hasn't been overwritten by a newer write)
            if current_entry is not None and current_entry.segment_id in compacted_segment_ids:
                updated_entry = IndexEntry(
                    segment_id=new_segment_id,
                    file_offset=new_offset,
                    record_size=rec_size,
                    key_len=k_len,
                    val_len=v_len,
                    timestamp_ns=ts,
                )
                self.index.put(key, updated_entry)

        return CompactionResult(
            segments_compacted=len(compacted_segment_ids),
            records_scanned=records_scanned,
            records_retained=records_retained,
            records_dropped=records_dropped,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
        )
