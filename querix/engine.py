"""
Querix Storage Engine - Main API
Integrates the Storage layer, Multi-Segment Disk Manager, and Disk-Backed B+ Tree Index
into a coherent, ACID-durable Key-Value Engine with online Compaction and ordered range scans.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from querix.buffer.pool_manager import BufferPoolManager
from querix.concurrency.lock_manager import LockManager, LockMode
from querix.concurrency.transaction import Transaction, TransactionState
from querix.index.bplus_tree import BPlusTree
from querix.index.memory_index import IndexEntry
from querix.storage.compaction import CompactionResult, Compactor
from querix.storage.disk import DEFAULT_MAX_SEGMENT_SIZE, DiskStorage
from querix.storage.record import TOMBSTONE_VAL_LEN
from querix.wal.log_manager import ActionType, LogManager
from querix.wal.recovery import RecoveryManager, RecoveryResult


def _to_bytes(data: Union[str, bytes]) -> bytes:
    """Utility helper to coerce string or bytes into raw bytes."""
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise TypeError(f"Key/Value must be str or bytes, got {type(data).__name__}")


class BTreeIndexAdapter:
    """Adapts BPlusTree to provide the index interface required by Compactor."""

    def __init__(self, tree: BPlusTree) -> None:
        self.tree = tree

    def put(self, key: bytes, entry: IndexEntry) -> None:
        ptr = (
            entry.segment_id,
            entry.file_offset,
            entry.record_size,
            entry.key_len,
            entry.val_len,
            entry.timestamp_ns,
        )
        self.tree.insert(key, ptr)

    def get(self, key: bytes) -> Optional[IndexEntry]:
        ptr = self.tree.search(key)
        if ptr is None or ptr[4] == TOMBSTONE_VAL_LEN:
            return None
        return IndexEntry(
            segment_id=ptr[0],
            file_offset=ptr[1],
            record_size=ptr[2],
            key_len=ptr[3],
            val_len=ptr[4],
            timestamp_ns=ptr[5] if len(ptr) > 5 else 0,
        )

    def delete(self, key: bytes) -> bool:
        return self.tree.delete(key)

    def contains(self, key: bytes) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self.tree)

    def keys(self) -> Iterator[bytes]:
        for k, ptr in self.tree.range_scan(None, None):
            if ptr[4] != TOMBSTONE_VAL_LEN:
                yield k

    def items(self) -> Iterator[Tuple[bytes, IndexEntry]]:
        for k, ptr in self.tree.range_scan(None, None):
            if ptr[4] != TOMBSTONE_VAL_LEN:
                entry = IndexEntry(
                    segment_id=ptr[0],
                    file_offset=ptr[1],
                    record_size=ptr[2],
                    key_len=ptr[3],
                    val_len=ptr[4],
                    timestamp_ns=ptr[5] if len(ptr) > 5 else 0,
                )
                yield k, entry

    def clear(self) -> None:
        for k, _ in list(self.tree.range_scan(None, None)):
            self.tree.delete(k)


class QuerixEngine:
    """
    Querix Key-Value Storage Engine.

    Guarantees:
    - Durability: Synchronous fsync writes on append.
    - Data Integrity: CRC32 checksums on every record.
    - Persistence: Automatic multi-segment recovery on restart.
    - Fast Reads & Ordered Scans: Disk-backed B+ Tree index with BufferPool caching.
    - Space Efficiency: Online log compaction purges obsolete versions & tombstones.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        max_segment_size: int = DEFAULT_MAX_SEGMENT_SIZE,
        sync_on_write: bool = True,
        buffer_pool_size: int = 50,
    ) -> None:
        """
        Initializes or opens an existing Querix database directory.

        :param data_dir: Directory containing database segment files.
        :param max_segment_size: Maximum size in bytes before active segment rolls.
        :param sync_on_write: Whether to call fsync on every write operation.
        :param buffer_pool_size: Number of 4KB frames cached in RAM by the BufferPoolManager.
        """
        self.data_dir = Path(data_dir).resolve()
        self.max_segment_size = max_segment_size
        self.sync_on_write = sync_on_write
        self.buffer_pool_size = buffer_pool_size

        self._disk = DiskStorage(
            self.data_dir,
            max_segment_size=max_segment_size,
            sync_on_write=sync_on_write,
        )

        index_db_path = self.data_dir / "index.db"
        if index_db_path.exists():
            try:
                os.remove(index_db_path)
            except Exception:
                pass

        self._bpm = BufferPoolManager(index_db_path, pool_size=buffer_pool_size)
        self._bplus_tree = BPlusTree(self._bpm)
        self._index = BTreeIndexAdapter(self._bplus_tree)
        self._compactor = Compactor(self._disk, self._index)

        # Initialize WAL LogManager and RecoveryManager
        wal_path = self.data_dir / "querix_wal.log"
        self._log_manager = LogManager(wal_path)
        self._bpm.set_log_manager(self._log_manager)
        self._recovery_manager = RecoveryManager(self._log_manager, self._bpm)

        # Initialize Row-Level LockManager for Concurrency & Strict 2PL
        self._lock_manager = LockManager()

        # Execute WAL recovery upon engine startup
        self._recovery_result = self._recovery_manager.recover()

        self._is_closed = False

        # Reconstruct B+ tree index by scanning all segments in chronological order
        self._recover()

    @property
    def lock_manager(self) -> LockManager:
        """Returns the row-level LockManager."""
        self._assert_open()
        return self._lock_manager

    @property
    def log_manager(self) -> LogManager:
        """Returns the Write-Ahead Log Manager."""
        self._assert_open()
        return self._log_manager

    @property
    def recovery_manager(self) -> RecoveryManager:
        """Returns the RecoveryManager."""
        self._assert_open()
        return self._recovery_manager

    @property
    def recovery_result(self) -> RecoveryResult:
        """Returns the RecoveryResult from engine startup."""
        self._assert_open()
        return self._recovery_result

    def recover(self) -> RecoveryResult:
        """Manually triggers WAL recovery."""
        self._assert_open()
        return self._recovery_manager.recover()

    @property
    def bplus_tree(self) -> BPlusTree:
        """Returns the underlying disk-backed B+ Tree index."""
        self._assert_open()
        return self._bplus_tree

    @property
    def buffer_pool_manager(self) -> BufferPoolManager:
        """Returns the underlying BufferPoolManager."""
        self._assert_open()
        return self._bpm

    @property
    def active_segment_id(self) -> int:
        """Returns the ID of the current active segment."""
        self._assert_open()
        return self._disk.active_segment_id

    @property
    def active_segment_path(self) -> Path:
        """Returns the filesystem path to the current active segment file."""
        self._assert_open()
        return self._disk.active_segment_path

    @property
    def total_disk_size(self) -> int:
        """Returns total bytes consumed across all segment files."""
        self._assert_open()
        return self._disk.total_size

    @property
    def segments_count(self) -> int:
        """Returns the number of segment files currently managed."""
        self._assert_open()
        return len(self._disk._segments)

    def _recover(self) -> None:
        """
        Scans all segment files in ascending chronological order,
        reconstructing the B+ Tree index.
        """
        for seg_id, offset, record_size, key, value, ts in self._disk.scan_all_segments():
            if value is None:
                # Tombstone: remove from index if present
                self._bplus_tree.delete(key)
            else:
                entry = (seg_id, offset, record_size, len(key), len(value), ts)
                self._bplus_tree.insert(key, entry)

    def begin_transaction(self) -> Transaction:
        """
        Begins a new ACID transaction under this engine.
        Tracks unique txn_id, registers with WAL, and enforces Strict 2PL.
        """
        self._assert_open()
        txn = Transaction(
            lock_manager=self._lock_manager,
            log_manager=self._log_manager,
            recovery_manager=self._recovery_manager,
            engine=self,
        )
        return txn.begin()

    @contextmanager
    def transaction(
        self,
        for_update: bool = False,
        exclusive_keys: Optional[List[Union[str, bytes]]] = None,
    ) -> Iterator[Transaction]:
        """
        Context manager for executing operations within an ACID transaction.
        Automatically commits on normal block exit, or aborts and rolls back
        on unhandled exceptions.
        """
        self._assert_open()
        txn = self.begin_transaction()
        if exclusive_keys:
            for k in exclusive_keys:
                txn.acquire_exclusive(_to_bytes(k))
        try:
            yield txn
            if txn.state == TransactionState.GROWING:
                txn.commit()
        except Exception:
            if txn.state == TransactionState.GROWING:
                txn.abort()
            raise

    def put(
        self,
        key: Union[str, bytes],
        value: Union[str, bytes],
        txn: Optional[Transaction] = None,
    ) -> None:
        """
        Stores a key-value pair persistently.
        Acquires an Exclusive (X) lock on the key.

        :param key: Key identifier (str or bytes).
        :param value: Value payload (str or bytes).
        :param txn: Optional active transaction context. If None, auto-commits.
        """
        self._assert_open()
        if txn is None:
            with self.transaction() as auto_txn:
                self.put(key, value, txn=auto_txn)
            return

        k_bytes = _to_bytes(key)
        v_bytes = _to_bytes(value)

        # Acquire Exclusive lock under transaction
        self._lock_manager.acquire(txn, k_bytes, LockMode.EXCLUSIVE)

        # Record pre-image savepoint for transaction rollback
        old_entry = self._bplus_tree.search(k_bytes)
        txn.record_savepoint("put", k_bytes, old_entry)

        seg_id, offset, record_size, val_len = self._disk.append(k_bytes, v_bytes)
        entry = (seg_id, offset, record_size, len(k_bytes), val_len, 0)
        self._bplus_tree.insert(k_bytes, entry)

    def get(
        self,
        key: Union[str, bytes],
        verify_checksum: bool = False,
        txn: Optional[Transaction] = None,
        for_update: bool = False,
    ) -> bytes:
        """
        Retrieves the value for a given key as raw bytes using the B+ Tree index.
        Acquires a Shared (S) lock (or Exclusive (X) lock if for_update=True).

        :param key: Key identifier (str or bytes).
        :param verify_checksum: If True, reads full record and verifies CRC32.
                                If False, seeks directly to the value bytes.
        :param txn: Optional active transaction context. If None, auto-commits.
        :param for_update: If True, acquires an Exclusive (X) lock for upcoming mutation.
        :return: Value as raw bytes.
        :raises KeyError: If the key does not exist or was deleted.
        """
        self._assert_open()
        if txn is None:
            with self.transaction() as auto_txn:
                return self.get(
                    key,
                    verify_checksum=verify_checksum,
                    txn=auto_txn,
                    for_update=for_update,
                )

        k_bytes = _to_bytes(key)
        mode = LockMode.EXCLUSIVE if for_update else LockMode.SHARED
        self._lock_manager.acquire(txn, k_bytes, mode)

        entry = self._bplus_tree.search(k_bytes)

        if entry is None or entry[4] == TOMBSTONE_VAL_LEN:
            raise KeyError(f"Key not found: {key!r}")

        seg_id, offset, record_size, key_len, val_len, ts = entry

        if verify_checksum:
            _, value, _ = self._disk.read_record_at(seg_id, offset)
            if value is None:
                raise KeyError(f"Key not found (tombstone): {key!r}")
            return value

        return self._disk.read_value_direct(
            segment_id=seg_id,
            offset=offset,
            key_len=key_len,
            val_len=val_len,
        )

    def get_str(
        self,
        key: Union[str, bytes],
        encoding: str = "utf-8",
        txn: Optional[Transaction] = None,
        for_update: bool = False,
    ) -> str:
        """Convenience wrapper returning value decoded as string."""
        return self.get(key, txn=txn, for_update=for_update).decode(encoding)

    def delete(
        self,
        key: Union[str, bytes],
        txn: Optional[Transaction] = None,
    ) -> None:
        """
        Deletes a key by appending a tombstone record to the active segment
        and removing the key from the B+ Tree index.
        Acquires an Exclusive (X) lock on the key.

        :param key: Key identifier.
        :param txn: Optional active transaction context. If None, auto-commits.
        :raises KeyError: If key does not exist.
        """
        self._assert_open()
        if txn is None:
            with self.transaction() as auto_txn:
                self.delete(key, txn=auto_txn)
            return

        k_bytes = _to_bytes(key)
        self._lock_manager.acquire(txn, k_bytes, LockMode.EXCLUSIVE)

        entry = self._bplus_tree.search(k_bytes)

        if entry is None or entry[4] == TOMBSTONE_VAL_LEN:
            raise KeyError(f"Cannot delete non-existent key: {key!r}")

        txn.record_savepoint("delete", k_bytes, entry)
        self._disk.append(k_bytes, None)
        self._bplus_tree.delete(k_bytes)

    def range_scan(
        self,
        start_key: Optional[Union[str, bytes]] = None,
        end_key: Optional[Union[str, bytes]] = None,
    ) -> List[Tuple[bytes, bytes]]:
        """
        Executes an ordered range scan returning [(key, value), ...]
        for all active keys where start_key <= key <= end_key.

        :param start_key: Lower bound key (inclusive). If None, starts from lowest key.
        :param end_key: Upper bound key (inclusive). If None, scans to highest key.
        :return: List of (key, value) pairs in ascending key order.
        """
        self._assert_open()
        s_bytes = _to_bytes(start_key) if start_key is not None else None
        e_bytes = _to_bytes(end_key) if end_key is not None else None

        results: List[Tuple[bytes, bytes]] = []
        for k, entry in self._bplus_tree.range_scan(s_bytes, e_bytes):
            val_len = entry[4]
            if val_len == TOMBSTONE_VAL_LEN:
                continue
            val = self._disk.read_value_direct(
                segment_id=entry[0],
                offset=entry[1],
                key_len=entry[3],
                val_len=entry[4],
            )
            results.append((k, val))
        return results

    def range(
        self,
        start_key: Optional[Union[str, bytes]] = None,
        end_key: Optional[Union[str, bytes]] = None,
    ) -> List[Tuple[bytes, bytes]]:
        """Convenience alias for range_scan."""
        return self.range_scan(start_key, end_key)

    def compact(self, roll_active_if_needed: bool = True) -> CompactionResult:
        """
        Runs compaction to reclaim disk space from frozen segments by purging
        obsolete record versions and tombstones.

        :param roll_active_if_needed: If True, rolls the active segment first so all
                                      preceding writes can be compacted immediately.
        :return: CompactionResult with statistics.
        """
        self._assert_open()
        return self._compactor.compact(roll_active_if_needed=roll_active_if_needed)

    def __contains__(self, key: Union[str, bytes]) -> bool:
        """Checks if key exists in the database."""
        self._assert_open()
        k_bytes = _to_bytes(key)
        entry = self._bplus_tree.search(k_bytes)
        return entry is not None and entry[4] != TOMBSTONE_VAL_LEN

    def __len__(self) -> int:
        """Returns total count of active keys in database."""
        self._assert_open()
        return len(self._bplus_tree)

    def keys(self) -> Iterator[bytes]:
        """Iterates over all active keys in ascending order."""
        self._assert_open()
        for k, entry in self._bplus_tree.range_scan(None, None):
            if entry[4] != TOMBSTONE_VAL_LEN:
                yield k

    def _assert_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("Cannot operate on a closed QuerixEngine instance.")

    def close(self) -> None:
        """Flushes buffers and safely closes all open segment files, WAL, and index buffer pool."""
        if not self._is_closed:
            self._bpm.close()
            self._log_manager.close()
            self._disk.close()
            self._is_closed = True

    def __enter__(self) -> QuerixEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
