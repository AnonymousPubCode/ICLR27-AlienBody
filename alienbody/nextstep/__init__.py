"""NEXTSTEP S1: the J-protocol interfaces (TODO_NEXTSTEP.md §1.2, §8.3).

New module, new scripts, new outputs.  It imports the frozen v2 primitives
read-only (oracle, strict parsers, sandbox) and never edits them, so every
published v3/v4 artifact keeps its exact provenance (DECISIONS D1/D2).
"""
from alienbody.nextstep.records import (  # noqa: F401
    PROTOCOL_ID,
    TERMINATION_REASONS,
    EpisodeRecord,
    EpisodeWriter,
    summarize_records,
)
