<!-- BEGIN PERSEUS-VAULT RULES (installed by `perseus-vault connect --rules`) -->
## Memory (Perseus Vault)

You have persistent memory via the perseus_vault_* MCP tools. Follow this loop:

1. **Session start:** before your first substantive action, call
   `perseus_vault_context` with `query` set to the current task (or
   `perseus_vault_recall` with topic keywords) and treat the results as
   established context.
2. **During work:** whenever a durable fact, decision, constraint, or lesson
   is established, immediately call `perseus_vault_remember` with a clear
   `category`, a stable `key`, and the fact in `content`. Set `recall_when`
   triggers describing when it should resurface. Record significant events
   with `perseus_vault_journal`.
3. **Before finishing:** if this session produced several related memories,
   call `perseus_vault_consolidate` (with `dry_run: true` first) to merge
   overlap into durable observations.

Do not store secrets, credentials, or transient scratch state as memories.
<!-- END PERSEUS-VAULT RULES -->
