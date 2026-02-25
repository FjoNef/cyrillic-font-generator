# Session Log: Branching Overhaul — 2026-02-25T14:04:33

## Summary
Aramaki and Batou implemented branching policy overhaul. Removed .squad/ files from main branch via .gitignore, established dev as integration branch, created PR #2 targeting dev. Updated ceremonies.md to reflect new workflow.

## Changes
- **Branch strategy:** Main is releases-only; dev is integration branch. All feature work on feature branches from dev.
- **Squad files:** Excluded .squad/ from main via .gitignore. Squad state lives on dev and feature branches.
- **PR workflow:** All new work via PR to dev (not main). Saito reviews before merge.
- **Ceremonies:** Updated .squad/ceremonies.md to reflect dev-based branching.

## Decisions Merged
- Branching policy overhaul (dev-based workflow, main releases-only)
- PR #1 review approval by Aramaki

## Outcome
Clean separation: main for releases, dev for integration, squad tooling off main. Ready for feature branch work.
