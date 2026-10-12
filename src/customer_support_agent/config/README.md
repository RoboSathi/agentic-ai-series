# config — this agent's overlay and settings

`config.yaml` is the agent's overlay on the platform defaults: identity and host, its declared features (all nine), models, paths under `db/` and `data/`, security phrases, the eval plugin, and the `customer_support_agent` section. `profiles/` holds named postures (`bert_intent`, `no_phrase_rules`).

- **Entry points:** `CustomerSupportAgentSettings`, `get_agent_settings`, `reload_agent_settings`
- **Configuration:** this folder
- **Design:** [LLD §B1](../../../design_docs/LLD.md#b1-customer-support-agent)
