### 2026-02-25T14:04:33: Branching policy overhaul
**By:** FjoNef (via Copilot)  
**What:**
- `main` is now releases-only. No .squad/ files on main. No direct feature work on main.
- `dev` is the integration branch. All feature branches are created from and merged to `dev`.
- `.squad/` is excluded from `main` via .gitignore. Squad state lives on dev and feature branches.
- Branch naming: `<type>/<agent>-<short-description>` branching from `dev`.
- Scribe branches: `chore/scribe-*` branching from `dev`.
**Why:** User directive — clean main for releases, dev as integration, squad tooling off main.
