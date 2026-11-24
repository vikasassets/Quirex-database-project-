"""
ARIES-Style Recovery Manager for Querix Storage Engine.
Executes the three-phase recovery algorithm upon startup:
1. Analysis Phase: Discovers active (loser) and committed (winner) transactions.
2. Redo Phase: Replays logged changes forward from checkpoint to reconstruct crash state.
3. Undo Phase: Rolls back uncommitted changes backward, restoring data consistency.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Set

from querix.buffer.pool_manager import BufferPoolManager
from querix.storage.page import HEADER_FORMAT, PAGE_SIZE, SlottedPage
from querix.wal.log_manager import ActionType, LogManager, LogRecord


@dataclass(frozen=True)
class RecoveryResult:
    """Statistics and outcome of the recovery cycle."""
    records_scanned: int
    txns_active: List[int]       # Loser transactions rolled back during Undo
    txns_committed: List[int]    # Winner transactions preserved during Redo
    pages_redone: int
    pages_undone: int


def _apply_page_state(page: SlottedPage, state_bytes: bytes) -> None:
    """Restores a SlottedPage's complete 4096-byte memory state."""
    if len(state_bytes) != PAGE_SIZE:
        raise ValueError(f"Expected {PAGE_SIZE} bytes for page, got {len(state_bytes)}")
    page._data = bytearray(state_bytes)
    (
        page.page_id,
        page.lower_bound,
        page.upper_bound,
        page.num_slots,
    ) = struct.unpack_from(HEADER_FORMAT, page._data, 0)


class RecoveryManager:
    """
    Coordinates crash recovery for Querix using the Write-Ahead Log.
    """

    def __init__(
        self,
        log_manager: LogManager,
        buffer_pool_manager: BufferPoolManager,
    ) -> None:
        self.log_manager = log_manager
        self.bpm = buffer_pool_manager

    def recover(self) -> RecoveryResult:
        """
        Executes the three-phase ARIES crash recovery algorithm:
        1. Analysis: Identifies active uncommitted transactions.
        2. Redo: Repeats history forward to restore memory state at the crash point.
        3. Undo: Scans backward to revert changes of uncommitted transactions.

        :return: RecoveryResult with detailed statistics.
        """
        records = self.log_manager.read_all_records()

        # =========================================================================
        # PHASE 1: ANALYSIS
        # =========================================================================
        active_txns: Dict[int, int] = {}       # txn_id -> last_lsn
        committed_txns: Set[int] = set()       # set of committed txn_ids
        dirty_pages: Dict[int, int] = {}        # page_id -> rec_lsn (earliest unwritten change)
        checkpoint_lsn = 0

        for record in records:
            if record.action_type == ActionType.BEGIN:
                active_txns[record.txn_id] = record.lsn
            elif record.action_type in (ActionType.INSERT, ActionType.UPDATE, ActionType.DELETE):
                active_txns[record.txn_id] = record.lsn
                if record.page_id >= 0 and record.page_id not in dirty_pages:
                    dirty_pages[record.page_id] = record.lsn
            elif record.action_type == ActionType.COMMIT:
                committed_txns.add(record.txn_id)
                active_txns.pop(record.txn_id, None)
            elif record.action_type == ActionType.ABORT:
                active_txns.pop(record.txn_id, None)
            elif record.action_type == ActionType.CHECKPOINT:
                checkpoint_lsn = record.lsn

        # Loser transactions: active at the time of the crash without a COMMIT
        loser_txns = set(active_txns.keys())
        redo_start_lsn = min(dirty_pages.values()) if dirty_pages else (checkpoint_lsn if checkpoint_lsn > 0 else 1)

        # =========================================================================
        # PHASE 2: REDO ("Repeat History")
        # =========================================================================
        pages_redone = 0
        for record in records:
            if record.lsn < redo_start_lsn:
                continue

            if record.action_type in (ActionType.INSERT, ActionType.UPDATE, ActionType.DELETE):
                if record.page_id < 0 or not record.redo_data:
                    continue

                page = self.bpm.fetch_page(record.page_id)
                try:
                    if len(record.redo_data) == PAGE_SIZE:
                        # Full page image redo
                        _apply_page_state(page, record.redo_data)
                    elif record.action_type == ActionType.INSERT:
                        # Slot-level insert
                        page.insert_record(record.redo_data)

                    page.page_lsn = record.lsn
                    pages_redone += 1
                finally:
                    self.bpm.unpin_page(record.page_id, is_dirty=True)

        # =========================================================================
        # PHASE 3: UNDO ("Roll Back Losers")
        # =========================================================================
        pages_undone = 0
        if loser_txns:
            # Scan backward from the end of the log
            for record in reversed(records):
                if record.txn_id in loser_txns and record.action_type in (
                    ActionType.INSERT, ActionType.UPDATE, ActionType.DELETE
                ):
                    if record.page_id < 0 or not record.undo_data:
                        continue

                    page = self.bpm.fetch_page(record.page_id)
                    try:
                        if len(record.undo_data) == PAGE_SIZE:
                            # Full page image undo
                            _apply_page_state(page, record.undo_data)

                        page.page_lsn = record.lsn
                        pages_undone += 1
                    finally:
                        self.bpm.unpin_page(record.page_id, is_dirty=True)

        # Flush all recovered pages to disk so state is durable
        self.bpm.flush_all_pages()

        return RecoveryResult(
            records_scanned=len(records),
            txns_active=sorted(list(loser_txns)),
            txns_committed=sorted(list(committed_txns)),
            pages_redone=pages_redone,
            pages_undone=pages_undone,
        )

    def rollback_transaction(self, txn_id: int) -> int:
        """
        Executes the WAL Undo phase specifically for a single transaction.
        Scans querix_wal.log backwards and applies undo_data to restore the
        affected pages to their pre-transaction states.

        :param txn_id: The identifier of the transaction being rolled back.
        :return: Count of pages undone.
        """
        records = self.log_manager.read_all_records()
        pages_undone = 0

        for record in reversed(records):
            if record.txn_id == txn_id and record.action_type in (
                ActionType.INSERT, ActionType.UPDATE, ActionType.DELETE
            ):
                if record.page_id < 0 or not record.undo_data:
                    continue

                page = self.bpm.fetch_page(record.page_id)
                try:
                    if len(record.undo_data) == PAGE_SIZE:
                        _apply_page_state(page, record.undo_data)
                    page.page_lsn = record.lsn
                    pages_undone += 1
                finally:
                    self.bpm.unpin_page(record.page_id, is_dirty=True)

        return pages_undone

