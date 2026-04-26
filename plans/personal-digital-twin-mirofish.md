# Personal Digital Twin + MiroFish - Implementation Plan

## Goal
Create a first-class personal digital twin that Hermes can use as the principal for autonomous behavior:

- Hermes remains the main executive agent.
- A new `twin` memory provider models the user's identity, preferences, state, and long-lived policy.
- MiroFish provides counterfactual simulation and stakeholder/world modeling for high-stakes decisions.
- Existing Hermes approval and tool-guard rails remain the final gate before real-world actuation.

This is not a "replace Hermes with MiroFish" plan. It is a "give Hermes a durable self-model and a simulation cortex" plan.

## Architecture Decision
### Path A (chosen)
Use three cooperating components:

1. `plugins/memory/twin/` - the personal digital twin memory provider
2. `tools/mirofish_tool.py` - native Hermes toolset that talks to a running MiroFish backend over HTTP
3. `plugins/twin-governor/` - a general plugin that uses existing `pre_llm_call`, `pre_tool_call`, and `post_tool_call` hooks to enforce autonomy policy

### Not Path B
Do not embed MiroFish as Hermes' primary conversation loop.

Reasons:

- MiroFish is a separate simulation stack with its own backend, frontend, memory, and report lifecycle.
- Hermes already has the main loop, tool registry, approvals, delegation, and scheduling.
- The user's identity model should evolve slowly and conservatively. A high-variance simulation stack is the wrong place to store that source of truth.

### Not Path C
Do not use a context engine as the main twin substrate.

Reasons:

- Hermes context engines are about compression and expansion, not long-term personal modeling.
- The memory provider interface already supports the lifecycle we need: `system_prompt_block`, `prefetch`, `sync_turn`, `on_session_end`, and `on_delegation`.

### Sidecar vs MCP
Use a native Hermes HTTP toolset first, not an MCP wrapper.

Reasons:

- MiroFish already exposes a Flask API under `/api/graph/*`, `/api/simulation/*`, and `/api/report/*`.
- Hermes already has a clean native tool path in `tools/*.py` plus `toolsets.py`.
- An MCP wrapper can be added later if we want cross-agent reuse outside Hermes.

## System Model
Treat the system as four layers:

- `Hermes executive`: decides, plans, asks, delegates, and executes tools
- `Twin provider`: represents who the user is, what they care about, and what Hermes may do on their behalf
- `MiroFish simulator`: predicts how people, projects, and social systems may react
- `Approval boundary`: blocks or asks before externalized or dangerous actions

In short:

- Hermes answers "what should I do now?"
- The twin answers "what kind of action is aligned with this person?"
- MiroFish answers "if we do that, what likely happens next?"
- Approvals answer "are we allowed to cross the boundary into action?"

## Core Design Rule
Simulated outcomes must never directly rewrite the user's core identity.

Allowed:

- simulations informing hypotheses about stakeholders
- simulations informing candidate plans
- simulations being stored as "predicted outcome" records

Not allowed:

- simulations automatically changing constitution, red lines, values, or durable preferences
- simulations being treated as observed reality
- simulations silently widening Hermes' delegation authority

This rule prevents identity drift and self-reinforcing hallucinated behavior loops.

## Twin Data Model
The twin should explicitly separate stable identity from volatile state and from uncertain world beliefs.

### Layer 1 - Constitution
Slowest-moving, highest-trust layer.

Contains:

- values
- red lines
- communication boundaries
- delegation scope
- autonomy mode
- classes of action that are always blocked, always ask, or can auto-run

Update rules:

- explicit user confirmation only
- no simulation writes
- no automatic inference from a single turn

### Layer 2 - Durable Profile
Moderately stable user model.

Contains:

- preferences
- habits
- working style
- decision style
- tone preferences
- relationship preferences
- recurring goals

Update rules:

- explicit statements are high confidence
- repeated observed behavior can promote a hypothesis into profile state
- contradictory evidence lowers confidence rather than overwriting immediately

### Layer 3 - Current State
Fast-moving, decaying context.

Contains:

- active priorities
- deadlines
- current projects
- stress/energy indicators
- travel/availability
- budget sensitivity
- current interpersonal context

Update rules:

- may be inferred from recent turns
- decays automatically by age
- should not be mistaken for identity

### Layer 4 - World Model
Beliefs about other people, organizations, and ongoing situations.

Contains:

- entities: people, teams, companies, projects, communities
- relationship edges
- beliefs about incentives and likely reactions
- confidence scores
- freshness timestamps
- provenance: explicit user statement, observed reality, or simulation hypothesis

Update rules:

- observed outcomes can strengthen beliefs
- simulation outputs create hypotheses, not facts
- freshness matters; stale world-model beliefs must not remain unchallenged forever

### Layer 5 - Calibration and Outcomes
Tracks whether the twin is getting better or worse.

Contains:

- predicted action outcomes
- actual observed outcomes
- deltas between prediction and reality
- corrections from the user
- false assumptions that should be demoted or removed

Update rules:

- every high-stakes action should have an outcome record if reality becomes known
- calibration stats should feed confidence, not rewrite identity directly

## Suggested Storage Layout
Use provider-owned storage under `HERMES_HOME`, not hardcoded home paths.

Suggested layout:

- `$HERMES_HOME/twin.json` - provider config and static policy
- `$HERMES_HOME/twin/twin.db` - SQLite store for profile, state, entities, predictions, outcomes
- `$HERMES_HOME/twin/snapshots/` - periodic exported snapshots for debugging and rollback
- `$HERMES_HOME/twin/reports/` - cached MiroFish summaries and decision memos

Suggested tables:

- `constitution_items`
- `profile_items`
- `state_items`
- `entities`
- `entity_beliefs`
- `action_policies`
- `simulation_runs`
- `predictions`
- `observed_outcomes`
- `corrections`

The provider should keep a small exported "active twin brief" in memory for fast recall, and use SQLite for queryable history and calibration.

## Twin Record Shape
The provider should normalize all facts into records with explicit trust metadata.

Example shape:

```json
{
  "scope": "profile",
  "key": "communication_style.directness",
  "value": "medium",
  "confidence": 0.84,
  "source": "explicit_user_statement",
  "observed_at": "2026-04-22T18:20:00Z",
  "last_confirmed_at": "2026-04-22T18:20:00Z",
  "expires_at": null,
  "provenance_ref": "session:abc123:turn:14"
}
```

Every record needs:

- scope
- value
- confidence
- provenance
- timestamps
- optional expiry

Without this, the twin will drift into an undifferentiated memory blob.

## Twin Memory Provider Behavior
Implement `plugins/memory/twin/__init__.py` as a standard Hermes memory provider.

### `system_prompt_block()`
Keep this static and short.

Include:

- twin mode is active
- the model must distinguish constitution, profile, state, and world hypotheses
- simulation outputs are not reality
- user identity changes require confirmation or strong repeated evidence

Do not include volatile personal data here if it can instead be injected via `prefetch()`. Hermes keeps prompt caching healthier when dynamic context stays out of the system prompt.

### `prefetch(query)`
Return a compact "twin brief" relevant to the current turn.

This brief should usually include:

- top constitution items relevant to the task
- active priorities and constraints
- relevant stakeholders/entities
- current autonomy policy for this action class
- open hypotheses with confidence
- links to prior outcomes if similar actions were attempted before

This is the right place to inject dynamic identity and world context because Hermes already injects provider prefetch into the user message at API-call time.

### `queue_prefetch(query)`
Pre-compute the next likely twin brief in the background.

Targets:

- related entities
- similar prior actions
- pending deadlines
- pending simulations that may complete before the next turn

### `sync_turn(user_content, assistant_content)`
Extract candidate updates after every turn.

Allowed writes:

- current-state changes
- confidence adjustments
- new entity mentions as low-confidence hypotheses
- candidate profile facts in a pending-review state

Disallowed direct writes:

- constitution rewrites
- delegation-scope expansion
- turning simulation outputs into observed facts

### `on_session_end(messages)`
Run a slower consolidation pass:

- merge repeated low-confidence observations
- summarize recent state transitions
- surface unresolved identity contradictions for review
- export a snapshot

### `on_delegation(task, result, ...)`
Record what Hermes delegated and what came back.

This matters because autonomy quality depends on learning from delegated work, not just direct chat turns.

### Provider Tools
Expose a small, safe tool surface:

- `twin_profile` - return the active twin brief or one section of it
- `twin_entities` - inspect relevant people/projects and their confidence-tagged beliefs
- `twin_feedback` - record user correction or observed outcome
- `twin_policy` - classify whether an action class is `allow`, `simulate`, `ask`, or `deny`
- `twin_note` - write an explicitly confirmed fact into the right layer

Avoid large write surfaces. The twin should be hard to poison accidentally.

## MiroFish Toolset
Implement a new native Hermes tool file: `tools/mirofish_tool.py`.

The first version should be a thin HTTP client over a running MiroFish backend.

### Configuration
Environment variables:

- `MIROFISH_BASE_URL`
- `MIROFISH_API_KEY` if the deployment adds auth later
- `MIROFISH_TIMEOUT`

Optional provider config in `twin.json`:

```json
{
  "mirofish": {
    "enabled": true,
    "base_url": "http://localhost:5001",
    "default_project": "personal-twin",
    "auto_seed_entities": true
  }
}
```

### Initial Tool Surface
Wrap the highest-value endpoints first:

- `mirofish_build_graph`
  - Build or update the graph from seed materials and structured context
- `mirofish_prepare_simulation`
  - Create a simulation scaffold from graph + scenario
- `mirofish_run_simulation`
  - Start the simulation
- `mirofish_simulation_status`
  - Poll readiness or run state
- `mirofish_generate_report`
  - Ask MiroFish for a report after simulation
- `mirofish_get_report`
  - Fetch the report body and metadata
- `mirofish_interview_agents`
  - Query selected agents after a simulation
- `mirofish_close_env`
  - Release simulation environments

These tool names are intentionally higher-level than the raw HTTP endpoints. Hermes should work with decision primitives, not backend route trivia.

### Result Contract
Every tool should return structured JSON with:

- `success`
- `project_id`
- `graph_id`
- `simulation_id`
- `report_id`
- `summary`
- `artifacts`
- `status`
- `next_recommended_step`

The model should not need to parse backend logs to use the tools effectively.

## Twin Governor Plugin
The memory provider gives Hermes identity and recall. It does not hard-enforce action policy. Use a general plugin for that.

Create `plugins/twin-governor/` and register hooks:

- `pre_llm_call`
- `pre_tool_call`
- `post_tool_call`

### `pre_llm_call`
Inject a small decision-policy block when the current request looks like it may lead to external action.

Examples:

- "This looks like a message to another human"
- "Simulation is required before external negotiation"
- "The twin marks this as ask-before-send"

This keeps the policy visible without rewriting the system prompt.

### `pre_tool_call`
Block tool executions that violate twin policy.

Return a standard Hermes block directive:

```json
{
  "action": "block",
  "message": "Twin policy requires simulation and user approval before sending this message."
}
```

Initial action classes:

- `external_message`
- `calendar_change`
- `public_post`
- `purchase_or_spend`
- `account_or_credentials_change`
- `long_running_external_automation`

Initial tool mapping:

- `send_message` -> `external_message`
- email/chat platform send tools -> `external_message`
- automation/webhook creation commands -> `long_running_external_automation`
- selected browser actions that submit forms -> later phase
- `terminal` remains governed primarily by existing dangerous-command approvals; do not make the first version parse arbitrary shell intent

### `post_tool_call`
Log the attempted action and its result into the twin store:

- action class
- whether a simulation was consulted
- whether the user approved it
- immediate tool result

This is the bridge between autonomy decisions and later calibration.

## Autonomy Policy
The twin should classify candidate actions into four buckets:

- `allow`
- `simulate`
- `ask`
- `deny`

### Suggested defaults
`allow`

- local workspace reasoning
- file reads
- code navigation
- low-stakes draft generation
- internal summaries

`simulate`

- ambiguous human messaging
- negotiation
- prioritization across competing stakeholders
- career and relationship strategy
- project announcements

`ask`

- sending messages externally
- scheduling or rescheduling with another person
- publishing or posting
- granting access or permissions
- any action with a meaningful reputation cost

`deny`

- legal claims
- medical claims
- transfers of money
- destructive account changes
- anything outside the user's declared delegation scope

### Simulation Trigger Heuristic
The governor should require simulation when any of the following are true:

- more than one stakeholder is implicated
- outcome depends on how other people will react
- stakes are medium or high
- Hermes is uncertain which option best aligns with the user
- there is delayed or second-order impact
- a similar past action had a poor calibrated outcome

## Personal Twin Config Layout
To minimize core Hermes changes, keep the provider's native config in `HERMES_HOME`.

Suggested `config.yaml`:

```yaml
memory:
  provider: twin
```

Suggested `$HERMES_HOME/twin.json`:

```json
{
  "enabled": true,
  "workspace": "personal",
  "autonomy_mode": "conservative",
  "constitution": {
    "red_lines": [
      "Do not make commitments on my behalf without approval",
      "Do not represent certainty about my intent when confidence is low"
    ]
  },
  "policy": {
    "default_external_action": "ask",
    "simulate_before_external_message": true,
    "simulate_before_public_post": true,
    "deny_money_movement": true
  },
  "learning": {
    "promote_profile_after_repetitions": 3,
    "max_auto_profile_confidence": 0.75,
    "state_ttl_hours": 168
  },
  "mirofish": {
    "enabled": true,
    "base_url": "http://localhost:5001",
    "default_project": "personal-twin"
  }
}
```

This lets us ship the feature mostly as plugins and tools without introducing large new top-level config branches.

## User Experience
Phase 1 should not try to make the system fully invisible. Add a small but explicit UX.

### Suggested slash commands
Implemented via plugin command registration:

- `/twin status`
- `/twin show`
- `/twin correct <fact or policy>`
- `/twin simulate <goal>`
- `/twin policy`

Optional CLI subcommands later:

- `hermes twin setup`
- `hermes twin status`
- `hermes twin export`
- `hermes twin calibrate`

### First-run behavior
The setup flow should collect:

- user name or preferred identity label
- delegation scope
- explicit red lines
- communication preferences
- whether MiroFish is enabled and where it runs

That first-run constitution matters more than any later inference.

## Learning Loop
The system should learn through a closed loop:

1. Hermes forms candidate actions.
2. The twin classifies the action and required safeguards.
3. MiroFish simulates when required.
4. Hermes chooses an action or drafts options.
5. The user approves or edits if policy requires it.
6. Hermes executes.
7. Real outcomes are observed and written back as calibration records.

This is where emergent behavior becomes useful instead of chaotic.

## Drift Prevention
This section is mandatory. Without it the twin will degrade.

### Hard rules
- Constitution updates require explicit user confirmation.
- Simulation output never writes directly to constitution or durable profile.
- Observed reality and simulated hypotheses must have different provenance labels.
- Current-state facts must expire automatically.
- Contradictions reduce confidence before they overwrite.
- External actions must remain behind existing Hermes approvals unless explicitly downgraded by policy.

### Soft rules
- Prefer additive hypotheses over destructive overwrite.
- Keep confidence visible to the model.
- Require repeated evidence before promoting a profile belief.
- Prefer "I am not sure yet" over false certainty.

### Correction path
The user needs a fast override path:

- `/twin correct ...`
- `twin_feedback(kind="correction", ...)`

Corrections should immediately demote or remove the offending belief and record the correction source.

## Licensing and Distribution Note
Hermes is MIT. MiroFish declares AGPL-3.0.

That means:

- a direct bundled integration needs legal review before merge/distribution
- a user-configured sidecar deployment is operationally cleaner, but still needs review if we distribute tight integration code
- do not assume "it is a separate HTTP service" fully resolves license obligations

This is not legal advice. It is a warning that the implementation plan must include a license review checkpoint before shipping.

## Files to Create
### New files
1. `plugins/memory/twin/__init__.py`
2. `plugins/memory/twin/README.md`
3. `plugins/memory/twin/plugin.yaml`
4. `plugins/twin-governor/__init__.py`
5. `plugins/twin-governor/plugin.yaml`
6. `tools/mirofish_tool.py`
7. `tests/twin_plugin/test_provider.py`
8. `tests/plugins/test_twin_governor.py`
9. `tests/tools/test_mirofish_tool.py`

### Optional later files
10. `plugins/twin-governor/README.md`
11. `tests/integration/test_twin_simulate_before_send.py`

## Files to Modify
1. `toolsets.py`
   - add `mirofish` toolset
2. `plugins/memory/__init__.py`
   - no code change required if plugin follows the normal directory pattern
3. `hermes_cli/memory_setup.py`
   - likely no core change required if the provider exposes `get_config_schema()` and `save_config()`
4. `README.md` or release notes
   - document the feature once it exists

Core loop changes should be avoided unless testing proves a genuine gap.

## Phased Rollout
### Phase 0 - Design and legal review
- finalize schema
- decide whether the MiroFish integration can be distributed
- confirm sidecar deployment model

### Phase 1 - Twin provider only
- implement `plugins/memory/twin/`
- no MiroFish yet
- no hard action blocking yet
- verify the twin brief, correction flow, and low-drift updates

Success criteria:

- Hermes can recall constitution, profile, and state cleanly
- corrections work
- prompt caching remains healthy because dynamic context stays in `prefetch()`

### Phase 2 - MiroFish toolset
- add `tools/mirofish_tool.py`
- enable manual simulation via direct tool use or `/twin simulate`

Success criteria:

- Hermes can build a scenario, run a simulation, and summarize the report
- failures degrade gracefully to "ask the user" instead of wedging the turn

### Phase 3 - Twin governor
- add hook-based policy enforcement
- require simulation before selected action classes
- block or ask before selected external tools

Success criteria:

- policy is actually enforced, not just hinted in prompts
- blocked actions return clear reasons
- the system records whether simulation was consulted

### Phase 4 - Calibration and autonomy expansion
- compare predicted vs observed outcomes
- tune confidence thresholds
- selectively widen auto-allowed action classes if the system proves reliable

Success criteria:

- measurable reduction in bad external actions
- stable identity over time
- improved simulation usefulness on repeated decision classes

## Testing
Always use `scripts/run_tests.sh`, not raw `pytest`.

Test categories:

- provider unit tests
  - constitution write protection
  - state decay
  - confidence demotion on contradiction
- governor tests
  - tool blocking by action class
  - simulation-required policy
  - approval-required policy
- MiroFish client tests
  - endpoint mapping
  - timeout and error handling
  - structured result normalization
- integration tests
  - twin prefetch injection does not mutate persisted messages
  - simulation outputs are stored as hypotheses, not observed facts
  - `send_message` is blocked when policy says simulate or ask first

## Recommended First Implementation Step
Build the `twin` memory provider before touching MiroFish.

Why:

- it establishes the source of truth for personal identity and delegation policy
- it provides value even without simulation
- it exposes the right policy surface for later MiroFish invocation

Once the twin provider is stable, add the MiroFish toolset, then add the governor that makes simulation mandatory for selected action classes.
