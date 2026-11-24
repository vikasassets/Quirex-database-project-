"""
Binary record serialization, framing, and data integrity verification.

Record Wire Format:
+----------------+--------------------+-----------------+-----------------+----------------+------------------+
| CRC32 (4B, !I) | Timestamp (8B, !Q) | Key Len (4B, !I)| Val Len (4B, !i)| Key (K bytes)  | Value (V bytes)  |
+----------------+--------------------+-----------------+-----------------+----------------+------------------+
Total Header Size = 20 bytes.
Val Len = -1 denotes a TOMBSTONE (deleted key).
"""

from __future__ import annotations

import struct
import time
import zlib
from typing import Optional, Tuple


# Header format:
# ! = Network byte order (big-endian)
# I = 4-byte unsigned int (CRC32)
# Q = 8-byte unsigned int (Timestamp in nanoseconds)
# I = 4-byte unsigned int (Key length)
# i = 4-byte signed int (Value length, -1 for tombstone)
HEADER_FORMAT = "!IQIi"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
TOMBSTONE_VAL_LEN = -1


class RecordError(Exception):
    """Base class for record framing errors."""
    pass


class CorruptedRecordError(RecordError):
    """Raised when record checksum fails or record header contains invalid data."""
    pass


class IncompleteRecordError(RecordError):
    """Raised when an unexpected EOF occurs mid-record."""
    pass


def encode_record(key: bytes, value: Optional[bytes], timestamp_ns: Optional[int] = None) -> bytes:
    """
    Serializes a key and value (or tombstone) into a binary framed record with CRC32.

    :param key: Key as raw bytes.
    :param value: Value as raw bytes, or None for a tombstone.
    :param timestamp_ns: Optional custom timestamp in nanoseconds. Defaults to current time.
    :return: Framed bytes ready to be written to disk.
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()

    key_len = len(key)
    if value is None:
        val_len = TOMBSTONE_VAL_LEN
        val_bytes = b""
    else:
        val_len = len(value)
        val_bytes = value

    # Compute CRC32 over (timestamp + key_len + val_len + key + value)
    body_for_crc = struct.pack("!QIi", timestamp_ns, key_len, val_len) + key + val_bytes
    crc = zlib.crc32(body_for_crc) & 0xFFFFFFFF

    header = struct.pack(HEADER_FORMAT, crc, timestamp_ns, key_len, val_len)
    return header + key + val_bytes


def decode_header(header_bytes: bytes) -> Tuple[int, int, int, int]:
    """
    Decodes the 20-byte record header.

    :return: (crc, timestamp_ns, key_len, val_len)
    """
    if len(header_bytes) != HEADER_SIZE:
        raise IncompleteRecordError(f"Expected {HEADER_SIZE} header bytes, got {len(header_bytes)}")
    return struct.unpack(HEADER_FORMAT, header_bytes)


def verify_and_decode_record(data: bytes) -> Tuple[bytes, Optional[bytes], int]:
    """
    Decodes and validates a complete binary record.

    :param data: Complete byte buffer representing a record.
    :return: Tuple of (key, value, timestamp_ns). If tombstone, value is None.
    :raises CorruptedRecordError: If CRC32 mismatch occurs.
    :raises IncompleteRecordError: If data buffer is too short.
    """
    if len(data) < HEADER_SIZE:
        raise IncompleteRecordError(f"Record data smaller than header: {len(data)} bytes")

    crc, timestamp_ns, key_len, val_len = decode_header(data[:HEADER_SIZE])
    expected_payload_len = key_len + (max(0, val_len))
    expected_total_len = HEADER_SIZE + expected_payload_len

    if len(data) < expected_total_len:
        raise IncompleteRecordError(
            f"Expected {expected_total_len} bytes for record, but only got {len(data)}"
        )

    # Reconstruct data to verify CRC
    body_for_crc = struct.pack("!QIi", timestamp_ns, key_len, val_len) + data[HEADER_SIZE:expected_total_len]
    computed_crc = zlib.crc32(body_for_crc) & 0xFFFFFFFF

    if computed_crc != crc:
        raise CorruptedRecordError(
            f"CRC32 mismatch: expected {crc:#010x}, calculated {computed_crc:#010x}"
        )

    payload = data[HEADER_SIZE:expected_total_len]
    key = payload[:key_len]
    value = None if val_len == TOMBSTONE_VAL_LEN else payload[key_len:key_len + val_len]

    return key, value, timestamp_ns
