# config — this agent's settings and overlay

`config.yaml` here is the agent's overlay on the platform defaults: its
identity and host, its declared features, its own versions, model,
memory paths, security phrases, eval plugin, and the `customer_support_agent`
section read by `CustomerSupportAgentSettings`.

## Vocabulary

- **overlay** — this file, merged over `agentic_commons/config/config.yaml` (named by `AGENTIC_OVERLAYS`)
- **features** — this agent's harness posture — declared here, in full, as `CLAUDE.md` requires

## Public interface

- `CustomerSupportAgentSettings`, `get_agent_settings()`, `reload_agent_settings()` (`settings.py`)
- Nested: `IntentSettings`, `IntentRouterSettings`, `FollowupSettings`, `PlanStepSettings`, `CsaEvalSettings`

## Configuration

Section `customer_support_agent:` — `grounding_facts`, `max_tool_iterations`, `action_classes`, `intents`,
`plans`, `intent_router`, `followup`, `history_turns`, `deliberation_markers`, user-code format,
`demo_user_code`, `policy_path`, `console_reviewer`, `dummy_data_generator_path`, and `evals`
(dataset, fixture seed and template, workspace, rubric, quality metrics, Promptfoo config,
grounding roles, reply patterns, prompt-leak n-gram). Env prefix `CUSTOMER_SUPPORT_AGENT__`.

## Seams

- Importing the package sets `AGENTIC_OVERLAYS` to this file unless the environment already names one.

## Collaborators

- Read by every module of this agent; checked by `tests/unit/test_config_surface.py`.

## Episodes

E01 onward.

## Tests

Run with `./venv/bin/pytest tests/unit/test_config.py tests/unit/test_config_surface.py -q` (or `make test` for everything).
