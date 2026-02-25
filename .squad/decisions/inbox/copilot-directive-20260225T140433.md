### 2026-02-25T14:04:33: User directives — branch strategy overhaul
**By:** FjoNef (via Copilot)
**What:**
- **Main branch must not contain any .squad/ files.** Squad state is team tooling, not product code.
- **All new branches must be created from dev, not main.** Main is reserved for releases; dev is the integration branch.
**Why:** User directive — enforce clean main, dev-based branching workflow.
