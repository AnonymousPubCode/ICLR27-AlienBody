"""Pre-registered A/B block membership for a formal set (D25 / 8.5 L347).

The plan calls for the formal set to be split into two blocks of equal size
("随机分成A/B各100", scaled to 200/200 at n=400) *before* any formal episode
runs, with each block carrying the same stratum shape as the whole set
(A/B 30/40/30 per the 200-env schedule, 60/80/60 at 400).  Block membership is a
deterministic function of the set's manifest and a recorded seed, so it can be
re-derived by anyone holding the frozen set — which is what §8.6.1's
"每个cell核对N、唯一ID、A/B成员" asks the launch check to verify.

The assignment is within stratum, never across: a shuffle that ignored strata
could hand block A an easier mix than block B, which would turn a replication
check into a confound.

This module is also what turns a manifest into the runner's dataset entries:
``config_entries`` emits one entry per block with an explicit ``env_ids`` list,
which `run_nextstep.resolve_dataset` already understands (the dataset hash is
then taken over exactly the selection, so each cell's dataset hash pins its
block).
"""
from __future__ import annotations

import random

BLOCK_RULE = ("within each stratum, the accepted members are shuffled with "
              "random.Random(block_seed) and split into equal consecutive "
              "slices, one per block, in the order given")


def assign_blocks(members: list[tuple[str, str]], block_names: list[str],
                  seed: int) -> dict[str, list[str]]:
    """Split ``members`` (``(env_id, stratum)`` pairs, acceptance order) into blocks.

    Every stratum's member count must divide evenly by ``len(block_names)`` —
    an uneven split would silently give the blocks different stratum shapes, so
    it is an error rather than a rounding decision.
    """
    if len(block_names) < 2:
        raise ValueError("a block split needs at least two block names")
    if len(set(block_names)) != len(block_names):
        raise ValueError(f"duplicate block names: {block_names}")

    ids = [env_id for env_id, _stratum in members]
    if len(set(ids)) != len(ids):
        raise ValueError("member ids are not unique; a block split would "
                         "produce overlapping blocks")

    by_stratum: dict[str, list[str]] = {}
    for env_id, stratum in members:
        by_stratum.setdefault(stratum, []).append(env_id)

    rng = random.Random(seed)
    blocks: dict[str, list[str]] = {name: [] for name in block_names}
    for stratum, ids in by_stratum.items():
        if len(ids) % len(block_names):
            raise ValueError(
                f"stratum {stratum!r} has {len(ids)} members, which does not "
                f"divide into {len(block_names)} equal blocks")
        rng.shuffle(ids)
        per_block = len(ids) // len(block_names)
        for index, name in enumerate(block_names):
            blocks[name].extend(ids[index * per_block:(index + 1) * per_block])
    return blocks


def block_counts(blocks: dict[str, list[str]], members: list[tuple[str, str]]) -> dict:
    """Per-block per-stratum counts — the audit that the shapes are identical."""
    stratum_of = {env_id: stratum for env_id, stratum in members}
    return {
        name: {stratum: sum(1 for env_id in ids if stratum_of[env_id] == stratum)
               for stratum in sorted(set(stratum_of.values()))}
        for name, ids in blocks.items()
    }


def config_entries(blocks: dict[str, list[str]], path: str, task: str,
                   total: int, note: str) -> dict:
    """The `datasets` entries the runner selects blocks through.

    The key deliberately avoids the ``{task}_formal`` prefix: that prefix is
    reserved for the one *unscoped* formal entry (``run_nextstep.formal_dataset_key``
    requires exactly one), and a block is selected explicitly with
    ``--dataset-key``.
    """
    return {
        f"{task}_block_{name}": {
            "path": path,
            "env_ids": list(env_ids),
            "task": task,
            "partition": "formal",
            "block": name,
            "n": len(env_ids),
            "note": f"{note} ({len(env_ids)} of {total} formal environments; "
                    f"membership frozen in the set manifest)",
        }
        for name, env_ids in blocks.items()
    }
