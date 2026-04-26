# Twin Memory Provider

Local personal digital twin memory provider for Hermes.

The provider separates user modeling into stable identity, durable profile, current state, world-model beliefs, and calibration records. Phase 1 is intentionally conservative: it supports explicit notes, corrections, policy lookup, entity inspection, and compact context recall without trying to infer or rewrite the user's identity automatically.

## Setup

```bash
hermes config set memory.provider twin
```

Optional native config lives at `$HERMES_HOME/twin.json`:

```json
{
  "enabled": true,
  "workspace": "personal",
  "autonomy_mode": "conservative",
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
  }
}
```

## Tools

- `twin_profile` - inspect the active twin brief or one layer.
- `twin_entities` - inspect people, projects, organizations, and confidence-tagged beliefs.
- `twin_feedback` - record corrections, confirmations, and observed outcomes.
- `twin_policy` - classify an action as `allow`, `simulate`, `ask`, or `deny`.
- `twin_note` - store an explicitly confirmed fact or policy.

## Drift Rules

- Constitution writes require `confirmed=true`.
- Simulation output cannot write to constitution or durable profile.
- Current-state facts expire by default.
- Corrections demote or correct matching beliefs instead of silently overwriting history.
- Simulation hypotheses and observed reality use different provenance labels.
