### 2026-02-25T153728: CI must run tests, not just build
**By:** Batou (fix per Aramaki review)  
**What:** squad-ci.yml and squad-preview.yml now include `npx vitest run` after frontend build. Backend uses `dotnet test`.  
**Why:** Aramaki flagged missing test step as blocking in PR #5 review. CI workflows were building successfully but not validating correctness via test execution.
