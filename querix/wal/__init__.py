"""
Querix Write-Ahead Log (WAL) & Recovery Subsystem.
Implements write-ahead logging, the WAL protocol, and ARIES-style crash recovery.
"""

from querix.wal.log_manager import (
    ActionType,
    CorruptedLogRecordError,
    LogManager,
    LogRecord,
)
from querix.wal.recovery import RecoveryManager, RecoveryResult

__all__ = [
    "ActionType",
    "LogRecord",
    "LogManager",
    "CorruptedLogRecordError",
    "RecoveryManager",
    "RecoveryResult",
]
