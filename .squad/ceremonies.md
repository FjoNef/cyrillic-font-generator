# Ceremonies

> Team meetings that happen before or after work. Each squad configures their own.

## Design Review

| Field | Value |
|-------|-------|
| **Trigger** | auto |
| **When** | before |
| **Condition** | multi-agent task involving 2+ agents modifying shared systems |
| **Facilitator** | lead |
| **Participants** | all-relevant |
| **Time budget** | focused |
| **Enabled** | ✅ yes |

**Agenda:**
1. Review the task and requirements
2. Agree on interfaces and contracts between components
3. Identify risks and edge cases
4. Assign action items

---

## Pull Request

| Field | Value |
|-------|-------|
| **Trigger** | auto |
| **When** | after |
| **Condition** | any agent completes an iteration of work |
| **Facilitator** | agent who did the work |
| **Participants** | Saito (review), Aramaki (merge approval for architecture changes) |
| **Time budget** | focused |
| **Enabled** | ✅ yes |

**Rules:**
1. Work is committed on a feature branch — **never directly on `main` or `dev`**.
2. Branch naming: `<type>/<agent>-<short-description>` (e.g. `feat/togusa-inference-pipeline`).
3. Open a PR to `dev` at the end of every iteration. `main` is releases-only.
4. PR description must include: what changed, why, any open issues.
5. Saito reviews for quality before merge where the change touches testable logic.

---

## Retrospective

| Field | Value |
|-------|-------|
| **Trigger** | auto |
| **When** | after |
| **Condition** | build failure, test failure, or reviewer rejection |
| **Facilitator** | lead |
| **Participants** | all-involved |
| **Time budget** | focused |
| **Enabled** | ✅ yes |

**Agenda:**
1. What happened? (facts only)
2. Root cause analysis
3. What should change?
4. Action items for next iteration
