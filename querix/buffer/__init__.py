"""
Querix Buffer Subsystem.
Provides in-memory caching and management for 4KB SlottedPages.
"""

from querix.buffer.pool_manager import (
    BufferPoolError,
    BufferPoolFullError,
    BufferPoolManager,
    Frame,
    LRUReplacer,
    PageNotFoundError,
)

__all__ = [
    "BufferPoolManager",
    "LRUReplacer",
    "Frame",
    "BufferPoolError",
    "BufferPoolFullError",
    "PageNotFoundError",
]
