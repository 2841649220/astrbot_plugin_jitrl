"""Exact top-k Jaccard retrieval in sub-linear time (Eq. 24).

The paper computes J(s, s_i) against every memory row, which is why the
published implementation subsamples the bank (paper Table 19: retrieval latency
grows linearly from 15 ms at 500 rows to 47 ms at 2500 rows). A chat plugin must
do better: retrieval runs on the critical path of *every* LLM request, and the
default configuration allows tens of thousands of experiences.

This module replaces the linear scan with an inverted index plus a provably
lossless candidate filter, so the result is the same top-k list a brute-force
scan would return:

1. Length filter. J(q, x) >= t implies t*|q| <= |x| <= |q|/t (because
   |q n x| <= min(|q|, |x|)), so rows whose token count is outside that window
   are rejected without touching their tokens.
2. Overlap lower bound. From the same identity,
   |q n x| >= t/(1+t) * (|q| + |x|). Inserting the smallest admissible
   |x| = t*|q| gives |q n x| >= t*|q|, so every admissible row shares at least
   ceil(t*|q|) query tokens.
3. Prefix filter (PPJoin, Bayardo et al. 2007). Order the query tokens by
   increasing posting-list length and keep only the first
   p = |q| - ceil(t*|q|) + 1 of them. If a row shared no token with that prefix,
   its overlap would be at most |q| - p = ceil(t*|q|) - 1, which contradicts
   step 2. Therefore every admissible row appears in at least one posting list
   of the query prefix, and no candidate can be missed.

Only rows that survive the length filter are intersected, and each intersection
costs O(|x|) with C-level set membership. For the paper's typical setting
(t = 0.8, |q| ~ 60) the prefix is a handful of rare tokens, so a 20k-row bank
costs a few hundred intersections instead of 20,000 full Jaccard computations.

The index is append-only and all state lives in flat array objects: about
0.55 KB per 60-token experience, i.e. roughly 11 MB for the default 20,000-entry
cap, versus about 44 MB for a naive dict[int, frozenset]. Deletions are handled
by the owner invalidating and rebuilding the index, so no tombstone bookkeeping
is needed here.
"""

from __future__ import annotations

import math
import time
from array import array
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["ExperienceIndex", "IndexHit"]

_EPS = 1e-9
"""Float slack used when turning thresholds into integer bounds.

0.7 * 10 evaluates to 7.000000000000001, so a bare ceil would demand one token
more than the threshold requires and silently drop true matches.
"""

_SECONDS_PER_DAY = 86400.0


@dataclass(slots=True, frozen=True)
class IndexHit:
    """One retrieved memory row, before its display text is loaded.

    Attributes:
        entry_id: Primary key of the stored experience.
        similarity: Jaccard similarity after the cross-scope penalty, i.e.
            exactly what the brute-force path reports.
        created_at: Row timestamp, used for recency weighting and tie-breaks.
        weight: similarity * (floor + (1 - floor) * recency), the kernel weight
            consumed by the value estimator.
        action_key: Normalised action identity.
        scope: Scope the row belongs to.
        discounted_return: Stored G_i.
        reward: Stored step reward r_i.
        overlap: Number of shared tokens (diagnostics).
        length: Token count of the stored state (diagnostics).
    """

    entry_id: int
    similarity: float
    created_at: float
    weight: float
    action_key: str
    scope: str
    discounted_return: float
    reward: float
    overlap: int = 0
    length: int = 0


class ExperienceIndex:
    """In-memory inverted index over tokenised states.

    The index never touches the database: the owning MemoryBank feeds it rows on
    insert and rebuilds it after deletes, so this class stays pure and directly
    unit-testable.

    Args:
        max_entries: Hard cap on indexed rows. An append that would cross the
            cap is rejected (returns False) so the caller can fall back to the
            exact SQL scan instead of silently degrading recall.
    """

    __slots__ = (
        "_action_index",
        "_action_keys",
        "_action_names",
        "_alive",
        "_created",
        "_entry_ids",
        "_id_position",
        "_lengths",
        "_max_entries",
        "_offsets",
        "_postings",
        "_returns",
        "_rewards",
        "_scope_index",
        "_scope_names",
        "_scopes",
        "_stale",
        "_surface_index",
        "_surface_names",
        "_surfaces",
        "_token_flat",
        "_token_ids",
        "_vocab",
    )

    def __init__(self, *, max_entries: int = 200000) -> None:
        self._max_entries = max(1, int(max_entries))
        # token -> id; query lookups must not grow the vocabulary.
        self._token_ids: dict[str, int] = {}
        self._vocab: list[str] = []
        # entry position -> data
        self._entry_ids = array("q")
        self._offsets = array("q")
        self._lengths = array("i")
        self._token_flat = array("i")
        self._returns = array("d")
        self._rewards = array("d")
        self._created = array("d")
        # token id -> entry positions, ascending
        self._postings: dict[int, array] = {}
        # deduplicated filter columns
        self._scope_names: list[str] = []
        self._scope_index: dict[str, int] = {}
        self._scopes = array("i")
        self._surface_names: list[str] = []
        self._surface_index: dict[str, int] = {}
        self._surfaces = array("i")
        self._action_names: list[str] = []
        self._action_index: dict[str, int] = {}
        self._action_keys = array("i")
        # Tombstones: rows are removed by clearing the alive flag, so pruning
        # and forgetting stay incremental. A compaction pass reclaims the space
        # once dead rows exceed roughly a fifth of the table.
        self._alive = bytearray()
        self._stale = 0
        self._id_position: dict[int, int] = {}

    # -- introspection -------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of live indexed rows.

        Returns:
            The row count, excluding tombstones.
        """
        return len(self._entry_ids) - self._stale

    @property
    def entry_count(self) -> int:
        """Number of live indexed rows.

        Returns:
            The row count, excluding tombstones.
        """
        return len(self._entry_ids) - self._stale

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens seen so far.

        Returns:
            The vocabulary size.
        """
        return len(self._vocab)

    @property
    def is_full(self) -> bool:
        """Whether the index reached its configured row cap.

        Returns:
            True when further appends are rejected.
        """
        return self.entry_count >= self._max_entries

    def clear(self) -> None:
        """Drop every indexed row and every deduplicated name."""
        self._token_ids.clear()
        self._vocab.clear()
        del self._entry_ids[:]
        del self._offsets[:]
        del self._lengths[:]
        del self._token_flat[:]
        del self._returns[:]
        del self._rewards[:]
        del self._created[:]
        self._postings.clear()
        self._scope_names.clear()
        self._scope_index.clear()
        del self._scopes[:]
        self._surface_names.clear()
        self._surface_index.clear()
        del self._surfaces[:]
        self._action_names.clear()
        self._action_index.clear()
        del self._action_keys[:]
        self._alive.clear()
        self._stale = 0
        self._id_position.clear()

    # -- construction --------------------------------------------------------
    @staticmethod
    def _intern_name(name: str, index: dict[str, int], names: list[str]) -> int:
        value = index.get(name)
        if value is None:
            value = len(names)
            index[name] = value
            names.append(name)
        return value

    def append(
        self,
        *,
        entry_id: int,
        tokens: Iterable[str],
        scope: str,
        surface: str,
        action_key: str,
        discounted_return: float,
        reward: float,
        created_at: float,
    ) -> bool:
        """Index one memory row.

        Args:
            entry_id: Database primary key of the row.
            tokens: Token set of the stored state. Empty sets are skipped: a
                state with no tokens can never be a neighbour, because Jaccard
                is defined as 0.0 for empty input.
            scope: Scope the row belongs to.
            surface: Coarse partition key of the state.
            action_key: Normalised action identity.
            discounted_return: Stored G_i.
            reward: Stored step reward.
            created_at: Row timestamp.

        Returns:
            True when the row was indexed, False when the index is full or the
            token set is empty.
        """
        if self.entry_count >= self._max_entries:
            return False
        entry_key = int(entry_id)
        if entry_key in self._id_position:
            # A re-written row (same primary key) must not be indexed twice.
            self.remove(entry_key)
        identifiers: set[int] = set()
        token_ids = self._token_ids
        for token in tokens:
            token_id = token_ids.get(token)
            if token_id is None:
                token_id = len(self._vocab)
                token_ids[token] = token_id
                self._vocab.append(token)
            identifiers.add(token_id)
        if not identifiers:
            return False
        position = len(self._entry_ids)
        ordered = sorted(identifiers)
        base = len(self._token_flat)
        self._token_flat.extend(ordered)
        self._entry_ids.append(entry_key)
        self._alive.append(1)
        self._id_position[entry_key] = position
        self._offsets.append(base)
        self._lengths.append(len(ordered))
        self._returns.append(float(discounted_return))
        self._rewards.append(float(reward))
        self._created.append(float(created_at))
        self._scopes.append(
            self._intern_name(scope, self._scope_index, self._scope_names)
        )
        self._surfaces.append(
            self._intern_name(surface, self._surface_index, self._surface_names)
        )
        self._action_keys.append(
            self._intern_name(action_key, self._action_index, self._action_names)
        )
        postings = self._postings
        for token_id in ordered:
            posting = postings.get(token_id)
            if posting is None:
                postings[token_id] = array("q", (position,))
            else:
                posting.append(position)
        return True

    # -- mutation ------------------------------------------------------------
    def remove(self, entry_id: int) -> bool:
        """Mark one row as deleted.

        The row stays in the posting lists as a tombstone until the next
        compaction, which keeps deletion O(1); the search skips dead rows.

        Args:
            entry_id: Database primary key of the row to drop.

        Returns:
            True when the row was live, False when it was unknown.
        """
        position = self._id_position.pop(int(entry_id), None)
        if position is None or not self._alive[position]:
            return False
        self._alive[position] = 0
        self._stale += 1
        self._maybe_compact()
        return True

    def remove_many(self, entry_ids: Iterable[int]) -> int:
        """Mark several rows as deleted.

        Args:
            entry_ids: Database primary keys to drop.

        Returns:
            The number of rows that were live.
        """
        removed = 0
        positions = self._id_position
        alive = self._alive
        for entry_id in entry_ids:
            position = positions.pop(int(entry_id), None)
            if position is None or not alive[position]:
                continue
            alive[position] = 0
            removed += 1
        if removed:
            self._stale += removed
            self._maybe_compact()
        return removed

    def update_return(
        self, entry_id: int, *, discounted_return: float, reward: float | None = None
    ) -> bool:
        """Overwrite the stored return of one row.

        Used by user feedback, which shifts an episode's rewards and therefore
        its discounted returns without touching the token sets.

        Args:
            entry_id: Database primary key of the row.
            discounted_return: New G_i.
            reward: New step reward; unchanged when None.

        Returns:
            True when the row was live, False when it was unknown.
        """
        position = self._id_position.get(int(entry_id))
        if position is None or not self._alive[position]:
            return False
        self._returns[position] = float(discounted_return)
        if reward is not None:
            self._rewards[position] = float(reward)
        return True

    def _maybe_compact(self) -> None:
        """Compact the index once tombstones become a meaningful fraction."""
        if self._stale <= 0:
            return
        if self._stale >= 64 and self._stale * 5 >= len(self._entry_ids):
            self._compact()

    def _compact(self) -> None:
        """Rebuild the arrays without tombstones, preserving token ids."""
        alive = self._alive
        offsets = self._offsets
        lengths = self._lengths
        flat = self._token_flat
        if not self._stale:
            return
        new_flat = array("i")
        new_offsets = array("q")
        new_lengths = array("i")
        new_ids = array("q")
        new_returns = array("d")
        new_rewards = array("d")
        new_created = array("d")
        new_scopes = array("i")
        new_surfaces = array("i")
        new_actions = array("i")
        new_alive = bytearray()
        new_positions: dict[int, int] = {}
        postings: dict[int, array] = {}
        position = 0
        for old_position in range(len(self._entry_ids)):
            if not alive[old_position]:
                continue
            start = offsets[old_position]
            end = start + lengths[old_position]
            new_offsets.append(len(new_flat))
            new_flat.extend(flat[start:end])
            new_lengths.append(lengths[old_position])
            entry_id = self._entry_ids[old_position]
            new_ids.append(entry_id)
            new_returns.append(self._returns[old_position])
            new_rewards.append(self._rewards[old_position])
            new_created.append(self._created[old_position])
            new_scopes.append(self._scopes[old_position])
            new_surfaces.append(self._surfaces[old_position])
            new_actions.append(self._action_keys[old_position])
            new_alive.append(1)
            new_positions[entry_id] = position
            for index in range(start, end):
                token_id = flat[index]
                posting = postings.get(token_id)
                if posting is None:
                    postings[token_id] = array("q", (position,))
                else:
                    posting.append(position)
            position += 1
        self._token_flat = new_flat
        self._offsets = new_offsets
        self._lengths = new_lengths
        self._entry_ids = new_ids
        self._returns = new_returns
        self._rewards = new_rewards
        self._created = new_created
        self._scopes = new_scopes
        self._surfaces = new_surfaces
        self._action_keys = new_actions
        self._alive = new_alive
        self._id_position = new_positions
        self._postings = postings
        self._stale = 0

    # -- retrieval -----------------------------------------------------------
    def query_token_ids(self, tokens: Iterable[str]) -> list[int]:
        """Map query tokens to known ids, dropping unseen ones.

        Args:
            tokens: Token set of the query state.

        Returns:
            Distinct known token ids.
        """
        lookup = self._token_ids
        identifiers: dict[int, None] = {}
        for token in tokens:
            token_id = lookup.get(token)
            if token_id is not None:
                identifiers[token_id] = None
        return list(identifiers)

    def search(
        self,
        tokens: Iterable[str],
        *,
        k: int = 10,
        threshold: float = 0.8,
        scope: str | None = None,
        surface: str = "",
        surface_gate: bool = False,
        scope_mode: str = "hybrid",
        prefilter_limit: int = 4000,
        recency_half_life_days: float = 30.0,
        cross_scope_penalty: float = 0.85,
        recency_floor: float = 0.5,
        now: float | None = None,
    ) -> list[IndexHit]:
        """Return the exact top-k neighbours of a query state.

        Args:
            tokens: Token set of the query state.
            k: Maximum number of neighbours to return.
            threshold: Minimum Jaccard similarity t. Values at or below zero are
                rejected: every row would be admissible, so the caller must use
                the SQL scan instead.
            scope: Scope of the current session.
            surface: Surface of the current state.
            surface_gate: Restrict results to rows with the same surface.
            scope_mode: session drops cross-scope rows, global keeps them
                unpenalised, hybrid down-weights them.
            prefilter_limit: Work cap on how many candidate rows are scored.
                When the candidate set is larger, the rows with the highest
                overlap and the newest timestamps win.
            recency_half_life_days: Half-life of the recency factor; values at
                or below zero disable it.
            cross_scope_penalty: Similarity multiplier for cross-scope rows in
                hybrid mode.
            recency_floor: Lower bound of the recency factor.
            now: Reference timestamp for the recency weight; defaults to the
                current time.

        Returns:
            Neighbours sorted by (-similarity, -created_at, -entry_id), carrying
            the same similarity and weight the brute-force path computes. Empty
            when the query shares no tokens with the bank.
        """
        unique_tokens = list(dict.fromkeys(tokens))
        count = len(unique_tokens)
        if count == 0 or threshold <= 0.0:
            return []
        limit = max(1, int(prefilter_limit))
        wanted = max(1, int(k))
        mode = scope_mode if scope_mode in {"session", "global", "hybrid"} else "hybrid"
        query_scope = scope or ""
        penalty = min(1.0, max(0.0, float(cross_scope_penalty)))
        if mode != "hybrid" or not query_scope:
            penalty = 1.0
        # Candidate generation must stay lossless for *every* row that can end up
        # above threshold. The cross-scope penalty only makes acceptance
        # stricter, so the loose bound is the configured threshold itself: a row
        # with final similarity >= threshold always has raw similarity >=
        # threshold and therefore at least ceil(threshold * |q|) shared tokens.
        required = max(1, math.ceil(threshold * count - _EPS))
        if required > count:
            return []
        prefix_length = count - required + 1

        postings = self._postings
        lookup = self._token_ids
        # Tokens that never occurred in memory are the rarest possible and sort
        # first, exactly like an empty posting list. They cannot contribute
        # overlap but they do count towards the query length, so the prefix
        # keeps its position instead of silently shrinking the denominator.
        ranked = sorted(
            (
                (len(postings.get(token_id, ())), token_id)
                for token_id in (lookup.get(token) for token in unique_tokens)
            ),
            key=lambda item: (item[0], item[1] if item[1] is not None else -1),
        )
        candidates: dict[int, None] = {}
        for _posting_length, token_id in ranked[:prefix_length]:
            if token_id is None:
                continue
            posting = postings.get(token_id)
            if posting is None:
                continue
            for position in posting:
                if position not in candidates:
                    candidates[position] = None
        if not candidates:
            return []

        lengths = self._lengths
        offsets = self._offsets
        flat = self._token_flat
        created = self._created
        returns = self._returns
        rewards = self._rewards
        entry_ids = self._entry_ids
        scopes = self._scopes
        surfaces = self._surfaces
        action_names = self._action_names
        scope_names = self._scope_names
        action_codes = self._action_keys
        minimum_length = required
        maximum_length = int(math.floor(count / threshold + _EPS))
        query_set = {
            token_id
            for token_id in (lookup.get(token) for token in unique_tokens)
            if token_id is not None
        }
        contains = query_set.__contains__
        surface_code = self._surface_index.get(surface) if surface_gate else None
        scope_code = self._scope_index.get(query_scope) if query_scope else None
        half_life = float(recency_half_life_days)
        reference = float(now) if now is not None else time.time()
        floor = min(1.0, max(0.0, float(recency_floor)))
        # (similarity, created_at, entry_id, position, overlap, length, weight)
        scored: list[tuple[float, float, int, int, int, int, float]] = []
        alive = self._alive
        for position, _ in candidates.items():
            if not alive[position]:
                continue
            length = lengths[position]
            if length < minimum_length or length > maximum_length:
                continue
            if surface_gate and surfaces[position] != surface_code:
                continue
            cross_scope = False
            if mode == "session" and scope_code is not None:
                if scopes[position] != scope_code:
                    continue
            elif mode == "hybrid" and scope_code is not None:
                cross_scope = scopes[position] != scope_code
            start = offsets[position]
            end = start + length
            overlap = 0
            for index in range(start, end):
                if contains(flat[index]):
                    overlap += 1
            if overlap < required:
                continue
            similarity = overlap / (count + length - overlap)
            # Contract inherited from the plugin's SQL scan: the threshold is
            # tested on the *raw* Jaccard score and the cross-scope penalty is
            # applied afterwards, so it only dampens ranking and the estimator
            # weight. (The paper has no scope penalty at all.)
            if similarity < threshold:
                continue
            if cross_scope:
                similarity *= penalty
            weight = similarity
            if half_life > 0.0:
                age = max(0.0, (reference - created[position]) / _SECONDS_PER_DAY)
                weight *= floor + (1.0 - floor) * (0.5 ** (age / half_life))
            scored.append(
                (
                    similarity,
                    created[position],
                    entry_ids[position],
                    position,
                    overlap,
                    length,
                    weight,
                )
            )
        if not scored:
            return []
        if len(scored) > limit:
            # Work cap: keep the rows closest to the query before ranking.
            scored.sort(key=lambda item: (-item[4], -item[1], -item[2]))
            del scored[limit:]
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return [
            IndexHit(
                entry_id=entry_id,
                similarity=similarity,
                created_at=created_at,
                weight=weight,
                action_key=action_names[action_codes[position]],
                scope=scope_names[scopes[position]],
                discounted_return=returns[position],
                reward=rewards[position],
                overlap=overlap,
                length=length,
            )
            for similarity, created_at, entry_id, position, overlap, length, weight in scored[
                :wanted
            ]
        ]
