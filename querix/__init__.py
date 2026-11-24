"""
Querix Storage Engine
An advanced indexed key-value storage engine built from scratch in Python.
"""

from querix.buffer import BufferPoolManager
from querix.concurrency import (
    DeadlockError,
    LockManager,
    LockMode,
    Strict2PLError,
    Transaction,
    TransactionAbortError,
    TransactionState,
)
from querix.engine import QuerixEngine
from querix.index.bplus_tree import BPlusTree
from querix.wal import ActionType, LogManager, RecoveryManager

__all__ = [
    "QuerixEngine",
    "BufferPoolManager",
    "BPlusTree",
    "LogManager",
    "RecoveryManager",
    "ActionType",
    "LockManager",
    "LockMode",
    "Transaction",
    "TransactionState",
    "DeadlockError",
    "TransactionAbortError",
    "Strict2PLError",
]
__version__ = "0.1.0"
