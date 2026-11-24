"""
Buffer Pool Manager for Querix Storage Engine.
Caches fixed-size 4KB SlottedPages in RAM, manages frame allocation and eviction via LRU Replacer,
tracks dirty and pin metadata, and coordinates disk I/O.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from querix.storage.disk import DiskManager
from querix.storage.page import PAGE_SIZE, SlottedPage


class BufferPoolError(Exception):
    """Base class for all BufferPool exceptions."""
    pass


class BufferPoolFullError(BufferPoolError, RuntimeError):
    """Raised when all buffer pool frames are pinned and no page can be evicted."""
    pass


class PageNotFoundError(BufferPoolError, KeyError):
    """Raised when accessing a page that does not exist in buffer pool or on disk."""
    pass


class Frame:
    """
    Represents an individual in-memory frame slot in the Buffer Pool.
    Holds a SlottedPage object along with caching metadata (pin_count, is_dirty).
    """

    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.page: Optional[SlottedPage] = None
        self.page_id: Optional[int] = None
        self.pin_count: int = 0
        self.is_dirty: bool = False
        self.page_lsn: int = 0

    def reset(self) -> None:
        """Resets the frame to an empty, unpinned, non-dirty state."""
        self.page = None
        self.page_id = None
        self.pin_count = 0
        self.is_dirty = False
        self.page_lsn = 0

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute lookups to the underlying SlottedPage if present."""
        if self.page is not None and hasattr(self.page, name):
            return getattr(self.page, name)
        raise AttributeError(f"'Frame' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return (
            f"Frame(id={self.frame_id}, page_id={self.page_id}, "
            f"pin_count={self.pin_count}, is_dirty={self.is_dirty})"
        )


class LRUReplacer:
    """
    Least Recently Used (LRU) replacer tracking unpinned frames eligible for eviction.
    Uses an OrderedDict to achieve O(1) time complexity for pin, unpin, victim, and remove operations.
    """

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        self._entries: OrderedDict[int, bool] = OrderedDict()

    def victim(self) -> Optional[int]:
        """
        Selects and removes the victim frame (the least recently unpinned/used frame).

        :return: Frame ID of the victim, or None if no frames are currently eligible for eviction.
        """
        if not self._entries:
            return None
        # Pop the oldest unpinned frame from the head of the OrderedDict
        frame_id, _ = self._entries.popitem(last=False)
        return frame_id

    def pin(self, frame_id: int) -> None:
        """
        Marks a frame as pinned, removing it from eviction consideration.

        :param frame_id: Frame ID to pin.
        """
        self._entries.pop(frame_id, None)

    def unpin(self, frame_id: int) -> None:
        """
        Marks a frame as unpinned, making it eligible for eviction.
        Appends or moves the frame to the MRU (most recently used) end.

        :param frame_id: Frame ID to unpin.
        """
        if frame_id in self._entries:
            self._entries.move_to_end(frame_id)
        else:
            self._entries[frame_id] = True

    def remove(self, frame_id: int) -> None:
        """
        Explicitly removes a frame from the replacer (e.g. upon page deletion).

        :param frame_id: Frame ID to remove.
        """
        self._entries.pop(frame_id, None)

    def __len__(self) -> int:
        """Returns the number of unpinned frames eligible for eviction."""
        return len(self._entries)

    @property
    def size(self) -> int:
        """Returns the number of unpinned frames eligible for eviction."""
        return len(self._entries)

    def __contains__(self, frame_id: int) -> bool:
        """Checks if a frame is currently in the replacer (unpinned and eligible)."""
        return frame_id in self._entries


class BufferPoolManager:
    """
    Buffer Pool Manager for the Querix Storage Engine.
    Manages fixed pool_size frames in memory, handles SlottedPage fetching, allocation,
    unpinning with dirty tracking, eviction via LRUReplacer, and synchronous disk flushing.
    """

    def __init__(
        self,
        disk_manager_or_pool_size: Any = None,
        pool_size_or_disk_manager: Any = None,
        *,
        disk_manager: Optional[Any] = None,
        pool_size: Optional[int] = None,
    ) -> None:
        """
        Initializes the BufferPoolManager with flexible arguments.

        Can be called as:
        - BufferPoolManager(disk_manager, pool_size=10)
        - BufferPoolManager(pool_size=10, disk_manager=dm)
        - BufferPoolManager(10)
        - BufferPoolManager("my_db.db", 10)
        - BufferPoolManager()
        """
        resolved_dm: Any = None
        resolved_size: Optional[int] = None

        if disk_manager is not None:
            resolved_dm = disk_manager
            resolved_size = pool_size if pool_size is not None else (
                disk_manager_or_pool_size if isinstance(disk_manager_or_pool_size, int) else None
            )
        elif isinstance(disk_manager_or_pool_size, int):
            resolved_size = disk_manager_or_pool_size
            resolved_dm = pool_size_or_disk_manager
        else:
            resolved_dm = disk_manager_or_pool_size
            if isinstance(pool_size_or_disk_manager, int):
                resolved_size = pool_size_or_disk_manager
            elif pool_size is not None:
                resolved_size = pool_size

        if resolved_size is None or not isinstance(resolved_size, int):
            resolved_size = pool_size if pool_size is not None else 10

        self._pool_size = resolved_size
        self._temp_file: Optional[Path] = None

        # Resolve or instantiate DiskManager
        if resolved_dm is None:
            fd, temp_path = tempfile.mkstemp(prefix="querix_bpm_", suffix=".db")
            os.close(fd)
            self._temp_file = Path(temp_path)
            self.disk_manager = DiskManager(self._temp_file)
        elif isinstance(resolved_dm, (str, Path)):
            self.disk_manager = DiskManager(resolved_dm)
        else:
            self.disk_manager = resolved_dm

        self.replacer = LRUReplacer(self._pool_size)
        self.frames: List[Frame] = [Frame(frame_id=i) for i in range(self._pool_size)]
        self.page_table: Dict[int, int] = {}  # page_id -> frame_id
        self._free_list: List[int] = list(range(self._pool_size))
        self.log_manager: Optional[Any] = None
        self._is_closed = False

    def set_log_manager(self, log_manager: Any) -> None:
        """Configures the LogManager used to enforce the WAL protocol on page flushes."""
        self.log_manager = log_manager

    def _ensure_wal_flushed_for_frame(self, frame: Frame) -> None:
        """
        Enforces the Write-Ahead Log (WAL) protocol:
        Before writing a dirty page to disk, all log records up to the page's page_LSN
        must be flushed and fsynced to the WAL log file.
        """
        if self.log_manager is None:
            return
        page_lsn = getattr(frame.page, "page_lsn", getattr(frame, "page_lsn", 0))
        if page_lsn > 0:
            self.log_manager.flush_up_to(page_lsn)

    @property
    def pool_size(self) -> int:
        """Returns the fixed number of frames managed by the buffer pool."""
        return self._pool_size

    def _get_available_frame(self) -> Optional[int]:
        """
        Finds an available frame index.
        Checks the free list first; if empty, evicts an unpinned frame using the LRU replacer.
        If the victim frame contains a dirty page, writes it to disk before repurposing.

        :return: Frame ID of available frame, or None if all frames are pinned.
        """
        # 1. Use an empty frame from the free list if available
        if self._free_list:
            return self._free_list.pop(0)

        # 2. Free list is empty: ask LRU replacer for an unpinned victim frame
        victim_frame_id = self.replacer.victim()
        if victim_frame_id is None:
            return None

        # 3. Evict the victim frame
        victim_frame = self.frames[victim_frame_id]
        if victim_frame.page_id is not None:
            # If dirty, write back to disk before eviction
            if victim_frame.is_dirty and victim_frame.page is not None:
                self._ensure_wal_flushed_for_frame(victim_frame)
                self.disk_manager.write_page(
                    victim_frame.page_id, victim_frame.page.serialize()
                )
                victim_frame.is_dirty = False
            # Remove mapping from page table
            self.page_table.pop(victim_frame.page_id, None)

        # Reset frame metadata
        victim_frame.reset()
        return victim_frame_id

    def fetch_page(self, page_id: int) -> SlottedPage:
        """
        Fetches the requested page into memory.

        If the page is already in the buffer pool, increments its pin count,
        removes it from the LRU replacer, and returns it.

        If not in the pool, finds an available frame (evicting an unpinned LRU frame
        and writing it to disk if dirty), reads the 4KB page from disk, loads it
        into the frame with pin_count=1, updates the page table, and returns it.

        :param page_id: Target page identifier to fetch.
        :return: SlottedPage instance loaded in memory.
        :raises BufferPoolFullError: If all frames in the buffer pool are currently pinned.
        :raises IOError: If reading the page from disk fails.
        """
        self._assert_open()

        # Cache hit
        if page_id in self.page_table:
            frame_id = self.page_table[page_id]
            frame = self.frames[frame_id]
            frame.pin_count += 1
            self.replacer.pin(frame_id)
            if frame.page is None:
                raise PageNotFoundError(f"Frame {frame_id} has page_id {page_id} but page is None")
            return frame.page

        # Cache miss: find an available frame
        frame_id = self._get_available_frame()
        if frame_id is None:
            raise BufferPoolFullError(
                f"Cannot fetch page {page_id}: buffer pool is full and all {self._pool_size} frames are pinned."
            )

        try:
            page_bytes = self.disk_manager.read_page(page_id)
            page = SlottedPage.deserialize(page_bytes)
        except Exception:
            # Reclaim frame to free list on read failure to prevent leakage
            self._free_list.append(frame_id)
            raise

        frame = self.frames[frame_id]
        frame.page = page
        frame.page_id = page_id
        frame.pin_count = 1
        frame.is_dirty = False

        self.page_table[page_id] = frame_id
        self.replacer.pin(frame_id)

        return page

    def unpin_page(self, page_id: int, is_dirty: bool = False) -> bool:
        """
        Decrements the pin count of a page in the buffer pool.

        If the pin count reaches 0, the page becomes eligible for LRU eviction.
        Updates the frame's dirty flag if is_dirty is True.

        :param page_id: Page identifier to unpin.
        :param is_dirty: True if the caller modified the page while holding it.
        :return: True if the page was unpinned successfully, False if not found or already unpinned.
        """
        self._assert_open()

        if page_id not in self.page_table:
            return False

        frame_id = self.page_table[page_id]
        frame = self.frames[frame_id]

        if frame.pin_count <= 0:
            return False

        # Mark dirty if modified; never reset an already dirty flag to False on unpin
        if is_dirty:
            frame.is_dirty = True

        frame.pin_count -= 1

        if frame.pin_count == 0:
            self.replacer.unpin(frame_id)

        return True

    def new_page(self) -> SlottedPage:
        """
        Allocates a brand new 4KB page on disk, loads it into an available frame
        in the buffer pool, sets pin_count=1, updates the page table, and returns it.

        :return: Newly allocated SlottedPage instance.
        :raises BufferPoolFullError: If all frames in the buffer pool are currently pinned.
        """
        self._assert_open()

        frame_id = self._get_available_frame()
        if frame_id is None:
            raise BufferPoolFullError(
                f"Cannot allocate new page: buffer pool is full and all {self._pool_size} frames are pinned."
            )

        try:
            page_id = self.disk_manager.allocate_page()
            page = SlottedPage(page_id=page_id)
        except Exception:
            self._free_list.append(frame_id)
            raise

        frame = self.frames[frame_id]
        frame.page = page
        frame.page_id = page_id
        frame.pin_count = 1
        frame.is_dirty = False

        self.page_table[page_id] = frame_id
        self.replacer.pin(frame_id)

        return page

    def flush_page(self, page_id: int) -> bool:
        """
        Forcibly writes a specific page from the buffer pool back to disk,
        regardless of its pin count, and clears its is_dirty flag.

        :param page_id: Page identifier to flush.
        :return: True if the page was found and flushed, False otherwise.
        """
        self._assert_open()

        if page_id not in self.page_table:
            return False

        frame_id = self.page_table[page_id]
        frame = self.frames[frame_id]

        if frame.page is not None:
            self._ensure_wal_flushed_for_frame(frame)
            data = frame.page.serialize()
            self.disk_manager.write_page(page_id, data)
            frame.is_dirty = False

        return True

    def flush_all_pages(self) -> None:
        """
        Forcibly writes all pages currently cached in the buffer pool to disk,
        clearing the is_dirty flag on every frame.
        """
        self._assert_open()
        for page_id in list(self.page_table.keys()):
            self.flush_page(page_id)

    def delete_page(self, page_id: int) -> bool:
        """
        Deletes a page from the buffer pool if it is unpinned.

        :param page_id: Target page identifier to remove.
        :return: True if deleted or not in pool, False if page is currently pinned.
        """
        self._assert_open()

        if page_id not in self.page_table:
            return True

        frame_id = self.page_table[page_id]
        frame = self.frames[frame_id]

        if frame.pin_count > 0:
            return False

        self.replacer.remove(frame_id)
        self.page_table.pop(page_id, None)
        frame.reset()
        self._free_list.append(frame_id)
        return True

    def get_pin_count(self, page_id: int) -> int:
        """Returns the current pin count for a page, or 0 if not currently cached."""
        frame_id = self.page_table.get(page_id)
        if frame_id is None:
            return 0
        return self.frames[frame_id].pin_count

    def is_dirty(self, page_id: int) -> bool:
        """Returns True if the specified page is loaded and marked dirty."""
        frame_id = self.page_table.get(page_id)
        if frame_id is None:
            return False
        return self.frames[frame_id].is_dirty

    def get_frame(self, frame_id: int) -> Frame:
        """Returns the Frame object at index frame_id."""
        if frame_id < 0 or frame_id >= self._pool_size:
            raise IndexError(f"Frame ID {frame_id} out of bounds (pool_size={self._pool_size})")
        return self.frames[frame_id]

    def get_page_frame(self, page_id: int) -> Optional[Frame]:
        """Returns the Frame object holding page_id, or None if not cached."""
        frame_id = self.page_table.get(page_id)
        if frame_id is None:
            return None
        return self.frames[frame_id]

    def _assert_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("Cannot operate on a closed BufferPoolManager.")

    def close(self) -> None:
        """
        Flushes all cached dirty pages to disk, safely closes the underlying disk manager,
        and cleans up temporary resources.
        """
        if not self._is_closed:
            try:
                self.flush_all_pages()
            except Exception:
                pass

            if hasattr(self.disk_manager, "close"):
                self.disk_manager.close()

            if self._temp_file is not None and self._temp_file.exists():
                try:
                    os.remove(self._temp_file)
                except Exception:
                    pass

            self._is_closed = True

    def __enter__(self) -> BufferPoolManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __len__(self) -> int:
        """Returns the number of pages currently cached in memory."""
        return len(self.page_table)

    def __contains__(self, page_id: int) -> bool:
        """Checks if a page_id is currently cached in the buffer pool."""
        return page_id in self.page_table
