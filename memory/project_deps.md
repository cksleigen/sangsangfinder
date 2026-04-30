---
name: venv311 dependency issues
description: Known missing/broken packages in the .venv311 environment
type: project
---

`google-generativeai` was missing from requirements.txt and not installed in `.venv311`. Added `google-generativeai>=0.8.0` to requirements.txt and installed v0.8.6.

`chromadb 0.6.3` has a telemetry bug (`capture() takes 1 positional argument but 3 were given`) that fires on every PersistentClient init. Fixed by setting the chromadb.telemetry loggers to CRITICAL at the top of app.py (before any chromadb import).

**Why:** google-generativeai was never added to requirements.txt, so fresh venv installs miss it. chromadb bug is upstream; upgrading is risky due to API changes.

**How to apply:** If the user reports terminal noise on startup or Gemini not working, these are the two known culprits.
