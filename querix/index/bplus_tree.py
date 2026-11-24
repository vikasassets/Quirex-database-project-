"""
Disk-Backed B+ Tree Index for Querix Storage Engine.
Interfaces directly with BufferPoolManager to store and retrieve fixed-size 4KB SlottedPages.
Supports variable-length byte keys, arbitrary record pointers, node splitting (leaf and internal),
point lookups, and fast ordered range scans via leaf page chaining.
"""

from __future__ import annotations

import bisect
import json
import struct
from typing import Any, Iterator, List, Optional, Tuple, Union

from querix.buffer.pool_manager import BufferPoolManager
from querix.storage.page import HEADER_SIZE, PAGE_SIZE, SLOT_SIZE, SlottedPage

# Node type identifiers
NODE_TYPE_INTERNAL = 1
NODE_TYPE_LEAF = 2


class BPlusTreeError(Exception):
    """Base class for B+ Tree errors."""
    pass


class LeafNode:
    """
    Leaf Node of a B+ Tree.
    Stored within a 4KB SlottedPage.
    Holds sorted keys, record pointer tuples, and next_page_id forming a linked list.
    """

    def __init__(
        self,
        page_id: int,
        keys: Optional[List[bytes]] = None,
        values: Optional[List[tuple]] = None,
        next_page_id: int = -1,
    ) -> None:
        self.page_id = page_id
        self.keys: List[bytes] = keys if keys is not None else []
        self.values: List[tuple] = values if values is not None else []
        self.next_page_id: int = next_page_id

    def write_to_page(self, page: SlottedPage) -> None:
        """Serializes leaf node state into a 4096-byte SlottedPage."""
        page._data = bytearray(PAGE_SIZE)
        page.page_id = self.page_id
        page.lower_bound = HEADER_SIZE
        page.upper_bound = PAGE_SIZE
        page.num_slots = 0
        page._sync_header()

        # Slot 0: Node Header (node_type, next_page_id)
        header_bytes = struct.pack("!Bi", NODE_TYPE_LEAF, self.next_page_id if self.next_page_id is not None else -1)
        page.insert_record(header_bytes)

        # Slots 1..N: Key-Value entries
        for k, v in zip(self.keys, self.values):
            val_bytes = json.dumps(v).encode("utf-8")
            entry = struct.pack("!H", len(k)) + k + struct.pack("!H", len(val_bytes)) + val_bytes
            page.insert_record(entry)

    @classmethod
    def from_page(cls, page: SlottedPage) -> LeafNode:
        """Reconstructs LeafNode from a SlottedPage."""
        if page.num_slots == 0:
            return cls(page_id=page.page_id, keys=[], values=[], next_page_id=-1)

        header = page.get_record(0)
        node_type, next_pid = struct.unpack("!Bi", header[:5])
        if node_type != NODE_TYPE_LEAF:
            raise ValueError(f"Expected leaf node ({NODE_TYPE_LEAF}), got {node_type}")

        keys: List[bytes] = []
        values: List[tuple] = []
        for s in range(1, page.num_slots):
            if not page.is_valid_slot(s):
                continue
            entry = page.get_record(s)
            k_len = struct.unpack_from("!H", entry, 0)[0]
            k = entry[2 : 2 + k_len]
            v_len = struct.unpack_from("!H", entry, 2 + k_len)[0]
            v_bytes = entry[4 + k_len : 4 + k_len + v_len]
            v = tuple(json.loads(v_bytes.decode("utf-8")))
            keys.append(k)
            values.append(v)

        return cls(page_id=page.page_id, keys=keys, values=values, next_page_id=next_pid)


class InternalNode:
    """
    Internal (Routing) Node of a B+ Tree.
    Stored within a 4KB SlottedPage.
    Holds M routing keys and M+1 child page_id pointers.
    """

    def __init__(
        self,
        page_id: int,
        keys: Optional[List[bytes]] = None,
        children: Optional[List[int]] = None,
    ) -> None:
        self.page_id = page_id
        self.keys: List[bytes] = keys if keys is not None else []
        self.children: List[int] = children if children is not None else []

    def write_to_page(self, page: SlottedPage) -> None:
        """Serializes internal node state into a 4096-byte SlottedPage."""
        page._data = bytearray(PAGE_SIZE)
        page.page_id = self.page_id
        page.lower_bound = HEADER_SIZE
        page.upper_bound = PAGE_SIZE
        page.num_slots = 0
        page._sync_header()

        # Slot 0: Node Header (node_type, first_child_id)
        first_child = self.children[0] if self.children else -1
        header_bytes = struct.pack("!Bi", NODE_TYPE_INTERNAL, first_child)
        page.insert_record(header_bytes)

        # Slots 1..M: Routing entries (child_id, key)
        for i in range(len(self.keys)):
            k = self.keys[i]
            child_id = self.children[i + 1]
            entry = struct.pack("!iH", child_id, len(k)) + k
            page.insert_record(entry)

    @classmethod
    def from_page(cls, page: SlottedPage) -> InternalNode:
        """Reconstructs InternalNode from a SlottedPage."""
        header = page.get_record(0)
        node_type, first_child = struct.unpack("!Bi", header[:5])
        if node_type != NODE_TYPE_INTERNAL:
            raise ValueError(f"Expected internal node ({NODE_TYPE_INTERNAL}), got {node_type}")

        children: List[int] = [first_child]
        keys: List[bytes] = []
        for s in range(1, page.num_slots):
            if not page.is_valid_slot(s):
                continue
            entry = page.get_record(s)
            child_id, k_len = struct.unpack_from("!iH", entry, 0)
            k = entry[6 : 6 + k_len]
            keys.append(k)
            children.append(child_id)

        return cls(page_id=page.page_id, keys=keys, children=children)


def read_node_from_page(page: SlottedPage) -> Union[LeafNode, InternalNode]:
    """Helper to deserialize any B+ Tree node from a SlottedPage."""
    if page.num_slots == 0:
        return LeafNode(page_id=page.page_id, keys=[], values=[], next_page_id=-1)
    header = page.get_record(0)
    node_type = header[0]
    if node_type == NODE_TYPE_LEAF:
        return LeafNode.from_page(page)
    elif node_type == NODE_TYPE_INTERNAL:
        return InternalNode.from_page(page)
    raise ValueError(f"Unknown B+ tree node type {node_type} in page {page.page_id}")


class BPlusTree:
    """
    Disk-Backed B+ Tree Index.
    Integrates directly with BufferPoolManager to manage 4KB SlottedPages.
    Ensures strict pinning/unpinning of pages during traversal and node splits to avoid memory leaks.
    """

    def __init__(
        self,
        bpm: BufferPoolManager,
        root_page_id: int = 0,
        max_keys: Optional[int] = None,
    ) -> None:
        """
        Initializes or opens a B+ Tree index.

        :param bpm: BufferPoolManager managing page I/O.
        :param root_page_id: Page ID of the root node (default 0).
        :param max_keys: Optional hard limit on keys per node (useful for testing small trees).
        """
        self.bpm = bpm
        self.root_page_id = root_page_id
        self.max_keys = max_keys

        # If disk manager is brand new and root page does not exist yet, allocate page 0
        if root_page_id not in self.bpm and (
            hasattr(self.bpm.disk_manager, "num_pages") and self.bpm.disk_manager.num_pages == 0
        ):
            root_page = self.bpm.new_page()
            root_node = LeafNode(page_id=root_page.page_id, keys=[], values=[], next_page_id=-1)
            root_node.write_to_page(root_page)
            self.bpm.unpin_page(root_page.page_id, is_dirty=True)

    def _can_fit_leaf(self, keys: List[bytes], values: List[tuple]) -> bool:
        """Checks if a set of leaf entries fits in a single 4096-byte SlottedPage."""
        if self.max_keys is not None and len(keys) > self.max_keys:
            return False

        # Calculate exact byte consumption in SlottedPage:
        # Header (10B) + Slot 0 (4B slot + 5B header = 9B)
        needed = HEADER_SIZE + (SLOT_SIZE + 5)
        for k, v in zip(keys, values):
            val_bytes = json.dumps(v).encode("utf-8")
            # Slot overhead (4B) + 2B k_len + len(k) + 2B v_len + len(val_bytes)
            needed += SLOT_SIZE + 4 + len(k) + len(val_bytes)
        return needed <= PAGE_SIZE

    def _can_fit_internal(self, keys: List[bytes], children: List[int]) -> bool:
        """Checks if a set of internal routing entries fits in a 4096-byte SlottedPage."""
        if self.max_keys is not None and len(keys) > self.max_keys:
            return False

        needed = HEADER_SIZE + (SLOT_SIZE + 5)
        for k in keys:
            # Slot overhead (4B) + 4B child_id + 2B k_len + len(k)
            needed += SLOT_SIZE + 6 + len(k)
        return needed <= PAGE_SIZE

    def _find_leaf_page(self, key: bytes) -> Tuple[int, SlottedPage, LeafNode, List[int]]:
        """
        Traverses from the root node down to the target leaf page for key.
        Unpins all intermediate internal nodes along the path.
        Returns (leaf_page_id, leaf_page, leaf_node, ancestors).
        NOTE: leaf_page is returned PINNED; caller is responsible for unpinning it.
        """
        ancestors: List[int] = []
        curr_id = self.root_page_id
        page = self.bpm.fetch_page(curr_id)
        node = read_node_from_page(page)

        while isinstance(node, InternalNode):
            ancestors.append(curr_id)
            idx = bisect.bisect_right(node.keys, key)
            child_id = node.children[idx]

            # Fetch child page before unpinning parent
            child_page = self.bpm.fetch_page(child_id)
            self.bpm.unpin_page(curr_id, is_dirty=False)

            curr_id = child_id
            page = child_page
            node = read_node_from_page(page)

        if not isinstance(node, LeafNode):
            self.bpm.unpin_page(curr_id, is_dirty=False)
            raise BPlusTreeError(f"Expected leaf node at page {curr_id}")

        return curr_id, page, node, ancestors

    def search(self, key: bytes) -> Optional[tuple]:
        """
        Traverses the B+ Tree from root to leaf to retrieve the record pointer for key.
        Strictly unpins all fetched pages during traversal.

        :param key: Target key to look up.
        :return: Record pointer tuple if found, None otherwise.
        """
        leaf_id, leaf_page, leaf_node, _ = self._find_leaf_page(key)
        try:
            idx = bisect.bisect_left(leaf_node.keys, key)
            if idx < len(leaf_node.keys) and leaf_node.keys[idx] == key:
                return leaf_node.values[idx]
            return None
        finally:
            self.bpm.unpin_page(leaf_id, is_dirty=False)

    def insert(self, key: bytes, record_pointer: tuple) -> None:
        """
        Inserts a key and its corresponding record pointer into the B+ Tree.
        Handles node splitting (both leaf and internal) when a node exceeds 4KB capacity.

        :param key: Binary key to insert.
        :param record_pointer: Tuple pointer to the disk record (e.g. (page_id, slot_id) or segment metadata).
        """
        leaf_id, leaf_page, leaf_node, ancestors = self._find_leaf_page(key)

        # Check if key already exists in this leaf: if so, update in-place
        idx = bisect.bisect_left(leaf_node.keys, key)
        if idx < len(leaf_node.keys) and leaf_node.keys[idx] == key:
            leaf_node.values[idx] = record_pointer
            leaf_node.write_to_page(leaf_page)
            self.bpm.unpin_page(leaf_id, is_dirty=True)
            return

        # Insert new key/value in sorted order
        new_keys = leaf_node.keys[:idx] + [key] + leaf_node.keys[idx:]
        new_values = leaf_node.values[:idx] + [record_pointer] + leaf_node.values[idx:]

        # Case 1: Fits without splitting
        if self._can_fit_leaf(new_keys, new_values):
            leaf_node.keys = new_keys
            leaf_node.values = new_values
            leaf_node.write_to_page(leaf_page)
            self.bpm.unpin_page(leaf_id, is_dirty=True)
            return

        # Case 2: Leaf node exceeds capacity -> Split!
        mid = len(new_keys) // 2
        left_keys = new_keys[:mid]
        left_values = new_values[:mid]
        right_keys = new_keys[mid:]
        right_values = new_values[mid:]
        routing_key = right_keys[0]

        # Case 2A: Leaf is the ROOT (Page 0)
        if not ancestors:
            # Allocate two new pages to become the left and right children
            left_page = self.bpm.new_page()
            right_page = self.bpm.new_page()
            left_id = left_page.page_id
            right_id = right_page.page_id

            # Write left leaf
            left_leaf = LeafNode(page_id=left_id, keys=left_keys, values=left_values, next_page_id=right_id)
            left_leaf.write_to_page(left_page)
            self.bpm.unpin_page(left_id, is_dirty=True)

            # Write right leaf
            right_leaf = LeafNode(page_id=right_id, keys=right_keys, values=right_values, next_page_id=leaf_node.next_page_id)
            right_leaf.write_to_page(right_page)
            self.bpm.unpin_page(right_id, is_dirty=True)

            # Root (Page 0) becomes an internal node pointing to left and right
            new_root = InternalNode(page_id=self.root_page_id, keys=[routing_key], children=[left_id, right_id])
            new_root.write_to_page(leaf_page)
            self.bpm.unpin_page(self.root_page_id, is_dirty=True)
            return

        # Case 2B: Leaf is NOT the root
        right_page = self.bpm.new_page()
        right_id = right_page.page_id

        # Write right leaf
        right_leaf = LeafNode(page_id=right_id, keys=right_keys, values=right_values, next_page_id=leaf_node.next_page_id)
        right_leaf.write_to_page(right_page)
        self.bpm.unpin_page(right_id, is_dirty=True)

        # Update left leaf
        leaf_node.keys = left_keys
        leaf_node.values = left_values
        leaf_node.next_page_id = right_id
        leaf_node.write_to_page(leaf_page)
        self.bpm.unpin_page(leaf_id, is_dirty=True)

        # Push routing key up to parent internal node
        self._insert_into_parent(ancestors, routing_key, right_id)

    def _insert_into_parent(self, ancestors: List[int], key: bytes, child_page_id: int) -> None:
        """Recursively inserts routing key and child pointer up the internal ancestor chain."""
        parent_id = ancestors.pop()
        parent_page = self.bpm.fetch_page(parent_id)
        parent_node = InternalNode.from_page(parent_page)

        idx = bisect.bisect_right(parent_node.keys, key)
        new_keys = parent_node.keys[:idx] + [key] + parent_node.keys[idx:]
        new_children = parent_node.children[:idx + 1] + [child_page_id] + parent_node.children[idx + 1:]

        # Case 1: Fits in parent internal node without splitting
        if self._can_fit_internal(new_keys, new_children):
            parent_node.keys = new_keys
            parent_node.children = new_children
            parent_node.write_to_page(parent_page)
            self.bpm.unpin_page(parent_id, is_dirty=True)
            return

        # Case 2: Internal node split!
        mid = len(new_keys) // 2
        promoted_key = new_keys[mid]
        left_keys = new_keys[:mid]
        left_children = new_children[:mid + 1]
        right_keys = new_keys[mid + 1:]
        right_children = new_children[mid + 1:]

        # Case 2A: Parent is ROOT
        if not ancestors:
            left_page = self.bpm.new_page()
            right_page = self.bpm.new_page()
            left_id = left_page.page_id
            right_id = right_page.page_id

            left_internal = InternalNode(left_id, left_keys, left_children)
            left_internal.write_to_page(left_page)
            self.bpm.unpin_page(left_id, is_dirty=True)

            right_internal = InternalNode(right_id, right_keys, right_children)
            right_internal.write_to_page(right_page)
            self.bpm.unpin_page(right_id, is_dirty=True)

            # Page 0 becomes new internal root
            new_root = InternalNode(self.root_page_id, [promoted_key], [left_id, right_id])
            new_root.write_to_page(parent_page)
            self.bpm.unpin_page(self.root_page_id, is_dirty=True)
            return

        # Case 2B: Parent is NOT root
        right_page = self.bpm.new_page()
        right_id = right_page.page_id

        right_internal = InternalNode(right_id, right_keys, right_children)
        right_internal.write_to_page(right_page)
        self.bpm.unpin_page(right_id, is_dirty=True)

        parent_node.keys = left_keys
        parent_node.children = left_children
        parent_node.write_to_page(parent_page)
        self.bpm.unpin_page(parent_id, is_dirty=True)

        # Recurse up
        self._insert_into_parent(ancestors, promoted_key, right_id)

    def delete(self, key: bytes) -> bool:
        """
        Removes a key from the B+ Tree.

        :param key: Target key to delete.
        :return: True if found and deleted, False otherwise.
        """
        leaf_id, leaf_page, leaf_node, _ = self._find_leaf_page(key)
        idx = bisect.bisect_left(leaf_node.keys, key)
        if idx < len(leaf_node.keys) and leaf_node.keys[idx] == key:
            leaf_node.keys.pop(idx)
            leaf_node.values.pop(idx)
            leaf_node.write_to_page(leaf_page)
            self.bpm.unpin_page(leaf_id, is_dirty=True)
            return True

        self.bpm.unpin_page(leaf_id, is_dirty=False)
        return False

    def _get_leftmost_leaf_id(self) -> int:
        """Traverses the leftmost child pointers from the root to locate the first leaf page."""
        curr_id = self.root_page_id
        page = self.bpm.fetch_page(curr_id)
        node = read_node_from_page(page)

        while isinstance(node, InternalNode):
            first_child = node.children[0]
            child_page = self.bpm.fetch_page(first_child)
            self.bpm.unpin_page(curr_id, is_dirty=False)

            curr_id = first_child
            page = child_page
            node = read_node_from_page(page)

        self.bpm.unpin_page(curr_id, is_dirty=False)
        return curr_id

    def range_scan(
        self,
        start_key: Optional[bytes] = None,
        end_key: Optional[bytes] = None,
    ) -> List[Tuple[bytes, tuple]]:
        """
        Performs an ordered range scan returning [(key, record_pointer), ...]
        for all entries where start_key <= key <= end_key.
        Follows the next_page_id leaf pointers, strictly unpinning each leaf page before
        moving to the next.

        :param start_key: Lower bound key (inclusive). If None, starts from the lowest key.
        :param end_key: Upper bound key (inclusive). If None, scans to the highest key.
        :return: List of (key, record_pointer) tuples in ascending key order.
        """
        results: List[Tuple[bytes, tuple]] = []

        if start_key is None:
            leaf_id = self._get_leftmost_leaf_id()
            leaf_page = self.bpm.fetch_page(leaf_id)
            leaf_node = LeafNode.from_page(leaf_page)
        else:
            leaf_id, leaf_page, leaf_node, _ = self._find_leaf_page(start_key)

        while leaf_id is not None and leaf_id >= 0:
            should_stop = False
            for k, v in zip(leaf_node.keys, leaf_node.values):
                if start_key is not None and k < start_key:
                    continue
                if end_key is not None and k > end_key:
                    should_stop = True
                    break
                results.append((k, v))

            next_id = leaf_node.next_page_id
            self.bpm.unpin_page(leaf_id, is_dirty=False)

            if should_stop or next_id is None or next_id < 0:
                break

            leaf_id = next_id
            leaf_page = self.bpm.fetch_page(leaf_id)
            leaf_node = LeafNode.from_page(leaf_page)

        return results

    def __contains__(self, key: bytes) -> bool:
        """Checks if a key exists in the index."""
        return self.search(key) is not None

    def __len__(self) -> int:
        """Returns total count of active keys stored in the B+ Tree."""
        count = 0
        leaf_id = self._get_leftmost_leaf_id()
        while leaf_id is not None and leaf_id >= 0:
            page = self.bpm.fetch_page(leaf_id)
            leaf = LeafNode.from_page(page)
            count += len(leaf.keys)
            next_id = leaf.next_page_id
            self.bpm.unpin_page(leaf_id, is_dirty=False)
            if next_id is None or next_id < 0:
                break
            leaf_id = next_id
        return count
