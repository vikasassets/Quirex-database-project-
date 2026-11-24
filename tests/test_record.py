"""
Unit tests for record encoding, decoding, and checksum integrity.
"""

import unittest
from querix.storage.record import (
    HEADER_SIZE,
    CorruptedRecordError,
    IncompleteRecordError,
    decode_header,
    encode_record,
    verify_and_decode_record,
)


class TestRecordFraming(unittest.TestCase):

    def test_encode_and_decode_valid_record(self):
        key = b"user:1001"
        val = b'{"name": "Alice", "role": "admin"}'
        timestamp = 1698765432100000000

        data = encode_record(key, val, timestamp_ns=timestamp)
        self.assertGreater(len(data), HEADER_SIZE)

        decoded_key, decoded_val, decoded_ts = verify_and_decode_record(data)
        self.assertEqual(decoded_key, key)
        self.assertEqual(decoded_val, val)
        self.assertEqual(decoded_ts, timestamp)

    def test_encode_and_decode_tombstone(self):
        key = b"deleted_key"
        data = encode_record(key, None)

        decoded_key, decoded_val, _ = verify_and_decode_record(data)
        self.assertEqual(decoded_key, key)
        self.assertIsNone(decoded_val)

    def test_corrupted_payload_raises_error(self):
        key = b"secure_key"
        val = b"confidential_data"
        data = bytearray(encode_record(key, val))

        # Corrupt a byte in the payload
        data[-1] ^= 0xFF

        with self.assertRaises(CorruptedRecordError):
            verify_and_decode_record(bytes(data))

    def test_incomplete_header_raises_error(self):
        truncated = b"incomplete_bytes"
        with self.assertRaises(IncompleteRecordError):
            decode_header(truncated)

    def test_incomplete_payload_raises_error(self):
        data = encode_record(b"hello", b"world")
        truncated = data[:-3]
        with self.assertRaises(IncompleteRecordError):
            verify_and_decode_record(truncated)


if __name__ == "__main__":
    unittest.main()
