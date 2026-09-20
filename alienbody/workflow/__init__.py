"""AlienWorkflow: a pure-discrete, CPU-only tool-selection workflow benchmark.

Frozen spec ``workflow-dev-v1``.  Eight boolean device flags (256 states), six
tools, a partial-assignment goal, and a step budget.  Nothing here uses the
network, a GPU, or global RNG state.

Submodules
    ``env``          bitmask transition, step budget, goal check
    ``observation``  canonical public rendering + public/hidden boundary
    ``reference``    independent per-flag-list transition, used to cross-check
    ``baselines``    deterministic BFS over the exact transition
    ``generator``    stratified, isomorphism-deduplicated dev-set generation
"""
from alienbody.workflow.baselines import (
    DEFAULT_STRATA,
    shortest_plan,
    shortest_plan_length,
    stratum_for_length,
)
from alienbody.workflow.env import (
    Action,
    WorkflowEnv,
    WorkflowInstance,
    action_problems,
    is_success,
    oracle_next_state,
)
from alienbody.workflow.generator import (
    GenConfig,
    GenerationResult,
    IsomorphismRegistry,
    canonical_key,
    canonical_key_of,
    generate,
    is_isomorphic,
)
from alienbody.workflow.observation import (
    HIDDEN_FIELDS,
    PUBLIC_FIELDS,
    format_state,
    render_observation,
)
from alienbody.workflow.reference import reference_next_state

__all__ = [
    "DEFAULT_STRATA",
    "GenConfig",
    "GenerationResult",
    "HIDDEN_FIELDS",
    "IsomorphismRegistry",
    "PUBLIC_FIELDS",
    "WorkflowEnv",
    "WorkflowInstance",
    "Action",
    "action_problems",
    "canonical_key",
    "canonical_key_of",
    "format_state",
    "generate",
    "is_isomorphic",
    "is_success",
    "oracle_next_state",
    "reference_next_state",
    "render_observation",
    "shortest_plan",
    "shortest_plan_length",
    "stratum_for_length",
]
