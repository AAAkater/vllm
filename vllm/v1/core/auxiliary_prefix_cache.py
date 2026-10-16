# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publication and leases for prefix state outside the token-page pool.

Token-page hashes alone cannot authorize a hit for recurrent auxiliary state.
The scheduler owns replacement; workers acknowledge a checkpoint only after
all state copies finish. This ledger does not own or pin token pages: lookup
must also validate the current scheduler block IDs.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field


def supports_auxiliary_prefix_speculation(method: str | None, maximum_requests: int) -> bool:
    """DSpark owns accepted draft state alongside its single target request."""
    return method is None or (method == "dspark" and maximum_requests == 1)


@dataclass(frozen=True)
class AuxiliaryPrefixDescriptor:
    slot: int
    generation: int
    num_tokens: int
    block_hash: bytes
    block_ids: tuple[tuple[int, ...], ...]

    @property
    def ticket(self) -> tuple[int, int]:
        return self.slot, self.generation


@dataclass(frozen=True)
class AuxiliaryPrefixAck:
    ticket: tuple[int, int]
    rank: int


@dataclass
class AuxiliaryPrefixOperations:
    captures: dict[str, AuxiliaryPrefixDescriptor] = field(default_factory=dict)
    restores: dict[str, AuxiliaryPrefixDescriptor] = field(default_factory=dict)


@dataclass
class _Entry:
    descriptor: AuxiliaryPrefixDescriptor
    ready: bool = False
    leases: set[str] = field(default_factory=set)


class AuxiliaryPrefixCache:
    """One bounded, scheduler-owned checkpoint table shared by all workers."""

    def __init__(self, capacity: int, world_size: int, block_size: int):
        if capacity <= 0 or world_size <= 0 or block_size <= 0:
            raise ValueError("Auxiliary prefix dimensions must be positive")
        self.capacity = capacity
        self.world_size = world_size
        self.block_size = block_size
        self.entries: OrderedDict[int, _Entry] = OrderedDict()
        self.generations = [0] * capacity

    def _entry(self, descriptor: AuxiliaryPrefixDescriptor) -> _Entry:
        entry = self.entries.get(descriptor.slot)
        if entry is None or entry.descriptor != descriptor:
            raise RuntimeError("Stale auxiliary prefix descriptor")
        return entry

    def reserve_capture(
        self,
        num_tokens: int,
        block_hash: bytes,
        block_ids: tuple[tuple[int, ...], ...],
    ) -> AuxiliaryPrefixDescriptor | None:
        """Reserve a worker slot; no hit is possible before acknowledgment.

        A full table with pending copies/leases is a cache miss, not permission
        to overwrite them. The worker replaces the same slot on every rank.
        """
        if num_tokens <= 0 or num_tokens % self.block_size:
            raise ValueError("Checkpoint must end at a complete token page")
        if not isinstance(block_hash, bytes) or not block_hash:
            raise ValueError("Checkpoint requires an immutable block hash")
        pages = num_tokens // self.block_size
        if not block_ids or any(
            len(group) != pages or any(block <= 0 for block in group)
            for group in block_ids
        ):
            raise ValueError("Checkpoint must cover all prefix pages")
        # Duplicates may use different physical pages after concurrent cold
        # requests. Preserve each descriptor's exact source allocation.
        slot = next((i for i in range(self.capacity) if i not in self.entries), None)
        if slot is None:
            slot = next(
                (
                    i
                    for i, entry in self.entries.items()
                    if entry.ready and not entry.leases
                ),
                None,
            )
        if slot is None:
            return None
        self.generations[slot] += 1
        descriptor = AuxiliaryPrefixDescriptor(
            slot, self.generations[slot], num_tokens, block_hash, block_ids
        )
        self.entries[slot] = _Entry(descriptor)
        self.entries.move_to_end(slot)
        return descriptor

    def find(
        self,
        block_hashes: Sequence[bytes],
        max_length: int,
        block_ids: tuple[Sequence[int], ...],
    ) -> AuxiliaryPrefixDescriptor | None:
        """Clamp a real token-page hit to an acknowledged state boundary."""
        matches = []
        for entry in self.entries.values():
            descriptor = entry.descriptor
            pages = descriptor.num_tokens // self.block_size
            if (
                entry.ready
                and descriptor.num_tokens <= max_length
                and pages <= len(block_hashes)
                and block_hashes[pages - 1] == descriptor.block_hash
                and tuple(tuple(group[:pages]) for group in block_ids)
                == descriptor.block_ids
            ):
                matches.append(descriptor)
        if not matches:
            return None
        selected = max(matches, key=lambda descriptor: descriptor.num_tokens)
        self.entries.move_to_end(selected.slot)
        return selected

    def lease(self, request_id: str, descriptor: AuxiliaryPrefixDescriptor) -> None:
        """Protect a scheduled restore until its worker result is consumed."""
        entry = self._entry(descriptor)
        if not entry.ready:
            raise RuntimeError("Cannot lease an unpublished auxiliary prefix")
        if request_id in entry.leases:
            raise RuntimeError("Duplicate auxiliary prefix restore lease")
        entry.leases.add(request_id)

    def complete(
        self,
        operations: AuxiliaryPrefixOperations,
        acknowledgments: Sequence[AuxiliaryPrefixAck],
    ) -> None:
        """Publish only the exact all-rank checkpoint set of this step.

        Validate the whole result before mutating the ledger. A canceled
        request's in-flight restore still retains its lease until this call.
        """
        captures = [self._entry(item) for item in operations.captures.values()]
        restores = {
            request_id: self._entry(item)
            for request_id, item in operations.restores.items()
        }
        expected = {
            (entry.descriptor.ticket, rank)
            for entry in captures
            for rank in range(self.world_size)
        }
        actual = [(ack.ticket, ack.rank) for ack in acknowledgments]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError("Auxiliary prefix acknowledgment is incomplete or stale")
        if any(entry.ready for entry in captures):
            raise RuntimeError("Auxiliary prefix capture was already published")
        if any(
            request_id not in entry.leases for request_id, entry in restores.items()
        ):
            raise RuntimeError("Auxiliary prefix restore has no live lease")
        for entry in captures:
            entry.ready = True
        for request_id, entry in restores.items():
            entry.leases.remove(request_id)

    def reset(self) -> None:
        if not self.can_reset:
            raise RuntimeError("Cannot reset auxiliary prefixes with in-flight owners")
        self.entries.clear()

    @property
    def can_reset(self) -> bool:
        return all(entry.ready and not entry.leases for entry in self.entries.values())
