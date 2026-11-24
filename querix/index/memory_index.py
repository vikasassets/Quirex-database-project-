"""
In-Memory Key Directory (KeyDir) Index.
Maps keys to their corresponding file location metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple


@dataclass(frozen=True)
class IndexEntry:
    """
    Metadata pointer to a record location on disk across segment files.
    """
    segment_id: int
    file_offset: int
    record_size: int
    key_len: int
    val_len: int
    timestamp_ns: int



class MemoryIndex:
    """
    In-memory Hash Index mapping byte keys to disk IndexEntry pointers.
    Provides O(1) key lookups, insertions, and deletions.
    """

    def __init__(self) -> None:
        self._entries: Dict[bytes, IndexEntry] = {}

    def put(self, key: bytes, entry: IndexEntry) -> None:
        """Adds or updates an entry in the index."""
        self._entries[key] = entry

    def get(self, key: bytes) -> Optional[IndexEntry]:
        """Returns the entry pointer for the given key, or None if not found."""
        return self._entries.get(key)

    def delete(self, key: bytes) -> bool:
        """
        Removes a key from the index.
        :return: True if the key was present, False otherwise.
        """
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def contains(self, key: bytes) -> bool:
        """Checks if key is present in the index."""
        return key in self._entries

    def __len__(self) -> int:
        """Returns the number of active keys indexed."""
        return len(self._entries)

    def keys(self) -> Iterator[bytes]:
        """Returns an iterator over all indexed keys."""
        return iter(self._entries.keys())

    def items(self) -> Iterator[Tuple[bytes, IndexEntry]]:
        """Returns an iterator over key and IndexEntry pairs."""
        return iter(self._entries.items())

    def clear(self) -> None:
        """Clears all entries from memory."""
        self._entries.clear()
