# QUERIX: Advanced Indexed Key-Value Storage Engine

QUERIX is a disk-backed, ACID-compliant indexed key-value storage engine implemented from first principles in Python with zero third-party database dependencies.

## Architecture & Roadmap

```
+---------------------------------------------------------+
|                  QuerixEngine (API Layer)               |
+---------------------------------------------------------+
             |                                  |
             v                                  v
+------------------------+          +-------------------------+
|  Index Subsystem       |          |  Storage Subsystem      |
|  - KeyDir (Milestone 1)|          |  - Binary Wire Framing  |
|  - B+ Tree (M5)        |          |  - Append Log / Slotted |
+------------------------+          |  - CRC32 Checksums      |
                                    |  - fsync Durability     |
                                    +-------------------------+
```

### Development Roadmap
- [x] **Milestone 1**: Minimal Append-Only Log Storage (Bitcask pattern) with CRC32 framing and restart durability.
- [x] **Milestone 2**: File Segmentation & Log Compaction (Multi-segment files, size thresholds, dead space reclamation, tombstone purging).
- [x] **Milestone 3**: Slotted Page Architecture (Fixed 4KB block layout, meet-in-the-middle slot array, defragmentation on deletion).
- [x] **Milestone 4**: Buffer Pool Manager (Frame management, LRU eviction, dirty page flushing).
- [x] **Milestone 5**: Disk-Backed B+ Tree Index (Branching, page-based node serialization, ordered range scans).
- [x] **Milestone 6**: Write-Ahead Logging (WAL) & ARIES Crash Recovery (Atomicity, REDO/UNDO logs).
- [x] **Milestone 7**: Concurrency Control & Transactions (Strict 2PL, Row-Level LockManager, Wait-Die Deadlock Prevention, Multi-Threaded ACID Isolation).

---

## Running the Unit Tests

QUERIX uses Python's built-in `unittest` module:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Quick Start

```python
from querix import QuerixEngine

# Open database
with QuerixEngine("data/store.db") as db:
    # PUT
    db.put("user:42", '{"name": "Alice", "role": "admin"}')

    # GET
    val = db.get_str("user:42")
    print(val)  # {"name": "Alice", "role": "admin"}

    # DELETE
    db.delete("user:42")
```
