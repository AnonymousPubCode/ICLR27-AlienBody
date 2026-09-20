"""Episode records for the NEXTSTEP J-protocol (TODO_NEXTSTEP.md §8.3).

This module owns the *record shape and the file discipline* — nothing about
models or arms.  Two rules from the plan are enforced here and nowhere else:

* **No appending.**  A writer opens its target file exclusively; a partial run
  is resumed into a NEW file (never appended to the old one), so a crashed run
  can always be told apart from a finished one.
* **Unknown is ``None``, never 0.**  Token fields that a provider does not
  report are ``null`` in the record, so a downstream sum can never quietly
  read a missing number as zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

PROTOCOL_ID = "nextstep-interfaces-v0"

# Closed enumeration of episode endings (DECISIONS D11).  ``success`` is set
# only by the task adapter from the real environment state.
TERMINATION_REASONS = (
    "success",
    "action_budget_exhausted",
    "plan_invalid",
    "plan_over_cap",
    "plan_exhausted",
    "parse_failed",
    "model_call_cap",
    "context_failure",
    "transport_failure",
    "oracle_exhausted_terminal",
    "internal_error",
)

# Failure labels a J-arm may attach to a real-action attempt.  They are
# response-level: they spend one real step (DECISIONS D15) but are never
# silently converted into action 0.
INVALID_ATTEMPT_KINDS = (
    "not_json",
    "unterminated_json",
    "fenced_json",
    "draft_over_cap",
    "action_out_of_range",
    "query_out_of_range",
    "code_not_fenced",
    "plan_not_list",
    "plan_empty",
    "plan_over_cap",
    "plan_action_out_of_range",
    "other",
)


# ── hashing helpers ────────────────────────────────────────────────────────

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_json_canonical(obj: Any) -> str:
    """Hash of the canonical JSON encoding (sorted keys, no spaces)."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def dataset_sha256(paths: Iterable[str | Path], keys: Optional[Iterable[str]] = None) -> str:
    """Hash a frozen dataset: sha256 over sorted (key, sha256(file bytes)).

    ``keys`` lets the caller use the suite's env ids as the sort key instead of
    the file names; the default is the path's ``.name``.
    """
    items = []
    for i, p in enumerate(paths):
        p = Path(p)
        key = list(keys)[i] if keys is not None else p.name
        items.append((str(key), sha256_file(p)))
    items.sort()
    h = hashlib.sha256()
    for key, digest in items:
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── record pieces ──────────────────────────────────────────────────────────

@dataclass
class Event:
    """One entry of the event table every J0/J1/J3L round re-renders.

    ``state_before`` / ``state_after`` are the rendered public states; the two
    sha256 fields make "the same initial state renders identically across
    arms" checkable without re-rendering.  ``status`` is ``ok`` for a real
    action that the environment applied and ``invalid`` for an attempt that
    spent a step without an action (DECISIONS D15).
    """
    index: int
    status: str
    action: Optional[int]
    state_before: str
    state_after: str
    state_before_sha256: str
    state_after_sha256: str
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Query:
    """One oracle interaction (J3L / J4).

    ``key`` is the canonical query tuple, ``answer`` the canonical answer,
    ``count`` how many times this exact query has now been asked, and
    ``served`` whether it consumed oracle budget.  J3L re-asks are answered
    again and still cost budget (DECISIONS D6).
    """
    index: int
    key: list
    answer: Optional[list]
    status: str
    count: int
    served: bool
    round: int = 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ModelCall:
    """One model request, including the retry chain (records every attempt)."""
    index: int
    round: int
    request_sha256: str
    response_sha256: Optional[str]
    response_text: Optional[str]
    finish_reason: Optional[str]
    attempts: int
    transport_error: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    cached_tokens: Optional[int]
    wall_seconds: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeRecord:
    """The §8.3 episode schema.  Field names are the contract."""
    protocol_id: str
    dataset_sha256: str
    env_id: str
    model_id: str
    provider: str
    inference_seed: int
    arm: str
    config_sha256: str
    code_sha256_at_launch: str
    prompt_sha256: Optional[str]
    started_at: str
    success: bool
    termination_reason: str
    real_action_attempts: int
    valid_real_actions: int
    oracle_attempts: int
    oracle_served: int
    distinct_queries: int
    model_calls: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    cached_tokens: Optional[int]
    wall_seconds: float
    parse_failures: int
    transport_failures: int
    code_errors: int
    events: list = field(default_factory=list)
    queries: list = field(default_factory=list)
    raw_responses: list = field(default_factory=list)
    attempt_id: str = ""
    # J-protocol additions that the plan's accounting needs explicitly
    partition: str = "dev"
    block_id: str = ""
    code_submissions: int = 0
    invalid_attempt_kinds: dict = field(default_factory=dict)
    termination_note: str = ""
    arm_budgets: dict = field(default_factory=dict)
    state_sha256_initial: Optional[str] = None

    def to_json(self) -> dict:
        return asdict(self)


# ── writer with the no-append / resume discipline ──────────────────────────

class EpisodeWriter:
    """Writes ``episodes__<arm>__<model>__<partition>__<block>.jsonl``.

    The file is created exclusively: if it exists, the run is refused unless it
    is a resume, in which case a *new* file (``.resumeN``) is created and
    already-complete episodes with a matching ``config_sha256`` are skipped.
    A torn last line is never treated as a completed episode.
    """

    def __init__(self, out_dir: str | Path, arm: str, model_id: str,
                 partition: str, block_id: str, resume: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = f"episodes__{arm}__{_slug(model_id)}__{partition}__{block_id or 'all'}.jsonl"
        self.path = self.out_dir / base
        self.prior_path: Optional[Path] = None
        self._skipped: set[tuple[str, str, str]] = set()
        self._skipped_records: list[dict] = []
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    f"refusing to append to an existing episode file: {self.path}. "
                    "A partial run must be resumed into a NEW file (--resume) or "
                    "written to a new --output-dir."
                )
            self.prior_path = self.path
            n = 1
            while self.path.exists():
                self.path = self.out_dir / f"{base[:-6]}.resume{n}.jsonl"
                n += 1
        self._fh = open(self.path, "x", encoding="utf-8")
        if self.prior_path is not None:
            self._load_prior()

    def resume_chain(self) -> list[dict]:
        """The files this run's records are spread across, oldest first.

        A resumed run writes only the missing episodes, so the complete data of
        a cell lives in more than one file; a manifest that named only the new
        one would point at a partial record set.  The chain is what the
        analysis has to read, in this order.
        """
        chain = []
        if self.prior_path is not None:
            chain.append({"file": self.prior_path.name,
                          "sha256": sha256_file(self.prior_path),
                          "episodes": len(self._skipped_records)})
        return chain

    def _load_prior(self) -> None:
        """Index complete episodes of the prior file; ignore a torn tail."""
        text = self.prior_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A half-written line: everything after it is suspect.
                break
            key = (rec.get("env_id"), rec.get("arm"), str(rec.get("inference_seed")))
            self._skipped.add(key)
            self._skipped_records.append(rec)

    def already_done(self, env_id: str, arm: str, inference_seed: int,
                     config_sha256: str) -> Optional[dict]:
        key = (env_id, arm, str(inference_seed))
        if key not in self._skipped:
            return None
        for rec in self._skipped_records:
            if (rec.get("env_id"), rec.get("arm"),
                    str(rec.get("inference_seed"))) == key:
                if rec.get("config_sha256") == config_sha256:
                    return rec
                return None
        return None

    def write(self, record: EpisodeRecord) -> None:
        self._fh.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text)


# ── summary recomputed from events (8.3 acceptance) ────────────────────────

def summarize_records(records: list[dict]) -> dict:
    """Recompute a run summary from stored episodes only.

    The plan's acceptance asks that the summary be recomputable from the event
    log; this function reads nothing but the records it is given and is the
    one used by both the runner's manifest and the tests.
    """
    n = len(records)
    successes = sum(1 for r in records if r.get("success"))
    terminations: dict[str, int] = {}
    invalid_kinds: dict[str, int] = {}
    for r in records:
        reason = r.get("termination_reason", "internal_error")
        terminations[reason] = terminations.get(reason, 0) + 1
        for k, v in (r.get("invalid_attempt_kinds") or {}).items():
            invalid_kinds[k] = invalid_kinds.get(k, 0) + int(v)

    def _sum(field_name: str) -> int:
        return sum(int(r.get(field_name) or 0) for r in records)

    def _sum_optional(field_name: str) -> Optional[int]:
        vals = [r.get(field_name) for r in records]
        if any(v is None for v in vals):
            return None      # unknown, not zero
        return sum(int(v) for v in vals)

    return {
        "n_episodes": n,
        "successes": successes,
        "success_rate": (successes / n) if n else None,
        "termination_reasons": terminations,
        "invalid_attempt_kinds": invalid_kinds,
        "real_action_attempts": _sum("real_action_attempts"),
        "valid_real_actions": _sum("valid_real_actions"),
        "oracle_attempts": _sum("oracle_attempts"),
        "oracle_served": _sum("oracle_served"),
        "model_calls": _sum("model_calls"),
        "parse_failures": _sum("parse_failures"),
        "transport_failures": _sum("transport_failures"),
        "code_errors": _sum("code_errors"),
        "code_submissions": _sum("code_submissions"),
        "input_tokens": _sum_optional("input_tokens"),
        "output_tokens": _sum_optional("output_tokens"),
        "reasoning_tokens": _sum_optional("reasoning_tokens"),
        "cached_tokens": _sum_optional("cached_tokens"),
        "wall_seconds": round(sum(float(r.get("wall_seconds") or 0.0) for r in records), 3),
        "distinct_queries": _sum("distinct_queries"),
    }


def resummarize_from_events(records: list[dict]) -> dict:
    """Independent recomputation of the success/attempt counts from events.

    Used by the tests to prove the stored accounting is derivable from the
    event log alone (8.3 acceptance).  ``success`` here means "the last event
    reached the goal", which the adapter set on the state; the runner stores
    that verdict explicitly in ``termination_reason``.
    """
    out = []
    for r in records:
        events = r.get("events") or []
        valid = sum(1 for e in events if e.get("status") == "ok")
        attempts = len(events)
        out.append({
            "env_id": r.get("env_id"),
            "arm": r.get("arm"),
            "real_action_attempts": attempts,
            "valid_real_actions": valid,
            "success": r.get("termination_reason") == "success",
            "valid_real_actions_match": valid == int(r.get("valid_real_actions") or 0),
            "attempts_match": attempts == int(r.get("real_action_attempts") or 0),
        })
    return {"n": len(out), "episodes": out}
