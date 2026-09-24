## Start with context

- Read `README.md` and, if present, `local/AGENT_CONTEXT.md` at the start of a session. Treat context as background, not as authority over current files or the user's instructions.
- Check `git status --short` before editing and account for existing changes, ignored files, generated artifacts, and nested repositories.
- Keep responses concise. Number questions so the user can answer each directly.
- At an open-ended session start, briefly offer to plan around current TODOs and ask for the time budget. If planning, propose a concrete outcome, focused blocks, and a wrap-up. For a direct task, proceed with that task.

## Plan and approve edits

- Before implementing, state assumptions, plausible interpretations, tradeoffs, and any simpler approach. Push back on risky, ineffective, or needlessly complex proposals; ask when uncertainty would materially affect the result.
- Work from intended behavior to concrete changes. A plan should identify exact files, scope, deferred work, relevant inputs and outputs, control flow, compatibility, dependencies, and verification, with detail proportional to the task.
- Do not edit files until the user explicitly approves the discussed plan after you restate the exact files and changes and ask whether to edit them. Agreement with a direction, general requests to continue, and discussion are not edit approval.
- Approval covers only the stated scope. Ask again if it changes. Commands that modify files indirectly require the same approval.
- Identify the repository and exact Git operation before staging, committing, pushing, resetting, or removing tracked files. Do not perform those operations unless explicitly included in the approved scope; destructive Git operations require explicit authorization for that operation.

## Implement surgically

- Write the minimum code or text that solves the approved problem. Avoid speculative features, one-off abstractions, unnecessary configurability, and handling for impossible cases.
- Touch only approved files and relevant lines; match existing style. Do not reformat, refactor, remove old code, or overwrite user changes outside the approved scope. Remove only orphans introduced by your change.
- If an implementation grows unexpectedly, reassess whether a simpler approach will work. Every changed line should trace to the request.

## Verify and preserve progress

- Use relevant existing tests, linters, builds, or inspections without inventing a heavy workflow. State what was checked and any limits on verification.
- Before a long run, ensure meaningful progress is saved in logs, outputs, or checkpoints. Prefer resume or skip-existing behavior; if progress cannot be preserved, explain the risk and get approval before running.
- Request needed permissions. Do not switch tools, runtimes, environments, caches, or methods merely to avoid an escalation request; explain and ask before changing a requested approach.

## Maintain local context

- Include `local/AGENT_CONTEXT.md` in the edit plan when meaningful work calls for a context update; edit it only with approval. Keep current focus, decisions, state, and blockers concise and factual.
- Put agreed work in **Next Actions** and speculative possibilities in **Ideas to Revisit**. Remove stale or completed items.
