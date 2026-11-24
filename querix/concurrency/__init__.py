"""
Concurrency Control and Transaction Management for Querix Storage Engine.
Implements Strict Two-Phase Locking (Strict 2PL), Row-Level Lock Manager,
Wait-Die Deadlock Prevention, and ACID Transactions.
"""

from querix.concurrency.lock_manager import (
    DeadlockError,
    InvalidTransactionStateError,
    LockManager,
    LockMode,
    Strict2PLError,
    TransactionAbortError,
)
from querix.concurrency.transaction import (
    Transaction,
    TransactionState,
)

__all__ = [
    "LockMode",
    "LockManager",
    "Transaction",
    "TransactionState",
    "TransactionAbortError",
    "DeadlockError",
    "Strict2PLError",
    "InvalidTransactionStateError",
]
