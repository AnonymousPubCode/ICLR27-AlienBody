# NEXTSTEP protocols — what is in this release

| File | What it is |
|---|---|
| `PROTOCOL_NEXTSTEP_INTERFACES.md` | The J-protocol interface spec: the four ladder arms, the state/history contract, budgets, transport policy. |
| `PROTOCOL_WORKFLOW.md` | The AlienWorkflow flagship task spec (bit-state schema, instance generation, verification). |
| `prereg_b_env_ids.json` | The environment ids frozen for the pre-registered B analysis. |

## What is *not* here

The run configurations that these specs are instantiated with —
`config.frozen.json`, `config.dev.json`, `config_prereg_c_cap256.json` — and the
launcher `scripts/run_nextstep.py` are not part of the release. They record the
bookkeeping of our own cluster runs (working directories, gateway endpoints,
per-run code/prompt hashes, queue tags), which is neither needed to read the
protocol nor meaningful on another machine. The two specs above define the
protocol those configs instantiate; the environment sets the runs consumed are
in `data/frozen_suites/`, and the adapters, records, and block assignment live
in `alienbody/nextstep/`.

The two runner-configuration assertions in `tests/test_nextstep_stubs.py` (the
transport-backoff and temperature policies) are kept as executable
documentation and skip when the launcher is absent.
