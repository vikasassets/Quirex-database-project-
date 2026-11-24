"""
Write-Ahead Log (WAL) Manager for Querix Storage Engine.
Implements binary log record framing, atomic appends to querix_wal.log,
and durable fsync flushing to satisfy the WAL protocol.
"""

from __future__ import annotations

import os
import struct
import zlib
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Union

# 2-byte magic identifier: "QL" (Querix Log)
LOG_RECORD_MAGIC = 0x514C

# Fixed 40-byte header format:
# ! = Big-endian
# H = 2-byte unsigned short (magic)
# Q = 8-byte unsigned long long (lsn)
# Q = 8-byte unsigned long long (txn_id)
# i = 4-byte signed int (page_id)
# H = 2-byte unsigned short (action_type)
# I = 4-byte unsigned int (undo_len)
# I = 4-byte unsigned int (redo_len)
# Q = 8-byte unsigned long long (prev_lsn)
LOG_HEADER_FORMAT = "!HQQiHIIQ"
LOG_HEADER_SIZE = struct.calcsize(LOG_HEADER_FORMAT)  # 40 bytes

# 4-byte CRC32 trailer format
CRC_FORMAT = "!I"
CRC_SIZE = struct.calcsize(CRC_FORMAT)  # 4 bytes


class ActionType(IntEnum):
    """Types of WAL log records."""
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    BEGIN = 4
    COMMIT = 5
    ABORT = 6
    CHECKPOINT = 7


class CorruptedLogRecordError(Exception):
    """Raised when a log record checksum or header is corrupted."""
    pass


class LogRecord:
    """
    In-memory representation of a Write-Ahead Log record.
    Contains LSN, transaction ID, page ID, action type, undo/redo payloads, and CRC checksum.
    """

    def __init__(
        self,
        lsn: int,
        txn_id: int,
        page_id: int,
        action_type: Union[ActionType, int],
        undo_data: bytes = b"",
        redo_data: bytes = b"",
        prev_lsn: int = 0,
    ) -> None:
        self.lsn = lsn
        self.txn_id = txn_id
        self.page_id = page_id
        self.action_type = ActionType(action_type)
        self.undo_data = bytes(undo_data)
        self.redo_data = bytes(redo_data)
        self.prev_lsn = prev_lsn

    def serialize(self) -> bytes:
        """
        Serializes the log record into binary wire format:
        [Header 40B] + [Undo Data] + [Redo Data] + [CRC32 4B]
        """
        header_bytes = struct.pack(
            LOG_HEADER_FORMAT,
            LOG_RECORD_MAGIC,
            self.lsn,
            self.txn_id,
            self.page_id,
            int(self.action_type),
            len(self.undo_data),
            len(self.redo_data),
            self.prev_lsn,
        )
        content_bytes = header_bytes + self.undo_data + self.redo_data
        checksum = zlib.crc32(content_bytes) & 0xFFFFFFFF
        return content_bytes + struct.pack(CRC_FORMAT, checksum)

    @classmethod
    def deserialize(cls, data: bytes) -> LogRecord:
        """
        Deserializes binary bytes into a LogRecord, verifying CRC32 integrity.
        """
        if len(data) < LOG_HEADER_SIZE + CRC_SIZE:
            raise CorruptedLogRecordError(
                f"Data length {len(data)} too short for log record"
            )

        header_bytes = data[:LOG_HEADER_SIZE]
        (
            magic,
            lsn,
            txn_id,
            page_id,
            action_code,
            undo_len,
            redo_len,
            prev_lsn,
        ) = struct.unpack(LOG_HEADER_FORMAT, header_bytes)

        if magic != LOG_RECORD_MAGIC:
            raise CorruptedLogRecordError(f"Invalid magic number {hex(magic)}")

        expected_total = LOG_HEADER_SIZE + undo_len + redo_len + CRC_SIZE
        if len(data) < expected_total:
            raise CorruptedLogRecordError(
                f"Expected {expected_total} bytes, got {len(data)}"
            )

        undo_offset = LOG_HEADER_SIZE
        redo_offset = undo_offset + undo_len
        crc_offset = redo_offset + redo_len

        undo_data = data[undo_offset:redo_offset]
        redo_data = data[redo_offset:crc_offset]

        # Verify checksum
        stored_crc = struct.unpack(CRC_FORMAT, data[crc_offset : crc_offset + CRC_SIZE])[0]
        content_bytes = data[:crc_offset]
        computed_crc = zlib.crc32(content_bytes) & 0xFFFFFFFF

        if stored_crc != computed_crc:
            raise CorruptedLogRecordError(
                f"CRC mismatch for LSN {lsn}: expected {stored_crc}, computed {computed_crc}"
            )

        return cls(
            lsn=lsn,
            txn_id=txn_id,
            page_id=page_id,
            action_type=ActionType(action_code),
            undo_data=undo_data,
            redo_data=redo_data,
            prev_lsn=prev_lsn,
        )

    def __repr__(self) -> str:
        return (
            f"LogRecord(lsn={self.lsn}, txn_id={self.txn_id}, page_id={self.page_id}, "
            f"type={self.action_type.name}, undo_len={len(self.undo_data)}, redo_len={len(self.redo_data)})"
        )


class LogManager:
    """
    Manages the sequential appending, durable fsync flushing, and scanning of querix_wal.log.
    Coordinates with the BufferPoolManager to enforce the WAL protocol.
    """

    def __init__(self, wal_path: Union[str, Path]) -> None:
        self.wal_path = Path(wal_path).resolve()
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.wal_path.exists():
            self.wal_path.touch()

        # Open in append-binary mode with reading capability
        self._file = open(self.wal_path, "a+b")
        self._is_closed = False

        # Read existing records to discover last LSN
        existing_records = self.read_all_records()
        if existing_records:
            last_record = existing_records[-1]
            self.next_lsn = last_record.lsn + 1
            self.flushed_lsn = last_record.lsn
        else:
            self.next_lsn = 1
            self.flushed_lsn = 0

    def append_record(
        self,
        txn_id: int,
        page_id: int,
        action_type: Union[ActionType, int],
        undo_data: bytes = b"",
        redo_data: bytes = b"",
        prev_lsn: int = 0,
    ) -> int:
        """
        Appends a new LogRecord to the WAL log buffer.
        Allocates a new unique auto-incrementing LSN.

        :return: Allocated Log Sequence Number (LSN).
        """
        self._assert_open()

        lsn = self.next_lsn
        self.next_lsn += 1

        record = LogRecord(
            lsn=lsn,
            txn_id=txn_id,
            page_id=page_id,
            action_type=action_type,
            undo_data=undo_data,
            redo_data=redo_data,
            prev_lsn=prev_lsn,
        )

        wire_bytes = record.serialize()
        self._file.seek(0, os.SEEK_END)
        self._file.write(wire_bytes)
        self._file.flush()

        return lsn

    def log_begin(self, txn_id: int) -> int:
        """Logs a transaction BEGIN record."""
        return self.append_record(
            txn_id=txn_id,
            page_id=-1,
            action_type=ActionType.BEGIN,
        )

    def log_commit(self, txn_id: int) -> int:
        """
        Logs a transaction COMMIT record and immediately fsyncs the log to disk.
        ACID durability guarantee: transaction is durable once this returns.
        """
        lsn = self.append_record(
            txn_id=txn_id,
            page_id=-1,
            action_type=ActionType.COMMIT,
        )
        self.flush()
        return lsn

    def log_abort(self, txn_id: int) -> int:
        """Logs a transaction ABORT record and flushes to disk."""
        lsn = self.append_record(
            txn_id=txn_id,
            page_id=-1,
            action_type=ActionType.ABORT,
        )
        self.flush()
        return lsn

    def log_checkpoint(self) -> int:
        """Logs a CHECKPOINT record and flushes to disk."""
        lsn = self.append_record(
            txn_id=-1,
            page_id=-1,
            action_type=ActionType.CHECKPOINT,
        )
        self.flush()
        return lsn

    def flush(self) -> None:
        """Forces all unwritten log buffers to be synchronously persisted to disk (fsync)."""
        self._assert_open()
        self._file.flush()
        os.fsync(self._file.fileno())
        self.flushed_lsn = self.next_lsn - 1

    def flush_up_to(self, lsn: int) -> None:
        """
        Ensures that all log records up to and including `lsn` have been flushed via fsync.
        Core WAL protocol enforcement method called by BufferPoolManager before writing dirty pages.
        """
        if self.flushed_lsn < lsn:
            self.flush()

    def read_all_records(self) -> List[LogRecord]:
        """
        Scans querix_wal.log from the beginning to the end, validating and returning all records.
        """
        records: List[LogRecord] = []
        if not self.wal_path.exists():
            return records

        with open(self.wal_path, "rb") as f:
            while True:
                header_bytes = f.read(LOG_HEADER_SIZE)
                if not header_bytes or len(header_bytes) < LOG_HEADER_SIZE:
                    break

                try:
                    (
                        magic,
                        lsn,
                        txn_id,
                        page_id,
                        action_code,
                        undo_len,
                        redo_len,
                        prev_lsn,
                    ) = struct.unpack(LOG_HEADER_FORMAT, header_bytes)

                    if magic != LOG_RECORD_MAGIC:
                        break

                    payload_and_crc_len = undo_len + redo_len + CRC_SIZE
                    rest_bytes = f.read(payload_and_crc_len)
                    if len(rest_bytes) < payload_and_crc_len:
                        break

                    full_record_bytes = header_bytes + rest_bytes
                    record = LogRecord.deserialize(full_record_bytes)
                    records.append(record)

                except CorruptedLogRecordError:
                    break

        return records

    def _assert_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("Cannot operate on a closed LogManager instance.")

    def close(self) -> None:
        """Flushes buffers and closes the WAL log file handle."""
        if not self._is_closed:
            self.flush()
            self._file.close()
            self._is_closed = True

    def __enter__(self) -> LogManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
