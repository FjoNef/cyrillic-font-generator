# Routing

## Rules

| Signal | Agent | Reason |
|--------|-------|--------|
| Architecture, tech decisions, scope, "what should we use" | Aramaki | Lead owns architecture |
| Web UI, frontend components, browser rendering, CSS, UX | Togusa | Frontend domain |
| .NET, backend APIs, server-side logic, data pipeline | Batou | Backend domain |
| AI model, training, ONNX, inference, font generation logic | Major | AI/ML domain |
| Tests, quality, edge cases, browser compatibility | Saito | Testing domain |
| Logs, decisions, session notes | Scribe | Always Scribe |
| Work queue, backlog, issue tracking | Ralph | Always Ralph |
| Multi-domain ("build the page", "add feature X") | Aramaki + relevant specialists | Parallel fan-out |
| Performance (client-side inference speed) | Major + Togusa | Shared domain |
| Font data, Unicode, Cyrillic character sets | Major (primary), Batou (data serving) | Domain split |
