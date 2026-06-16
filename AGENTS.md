# AGENTS.md

## Project Snapshot

This repository trains and runs a few-shot Yu-Gi-Oh card recognition pipeline built around MobileOne embeddings, metric-learning losses, FAISS retrieval, and YOLO-based detection. Keep changes focused on the script or module that actually owns the behavior.

## Entry Points

- [train.py](train.py) is the main training script.
- [eval.py](eval.py) runs embedding extraction and retrieval evaluation.
- [app.py](app.py) is the end-to-end demo / inference path.
- [data_preprocess.py](data_preprocess.py) holds card perspective correction and artwork extraction helpers.
- [README.md](README.md) is the only top-level setup note and should be linked, not duplicated.

## Working Conventions

- Install dependencies with `pip install -r requirements.txt`.
- The code assumes ImageFolder-style datasets and default 56x56 image inputs in several paths; verify tensor shapes and normalization when touching data or model code.
- Preserve `KMP_DUPLICATE_LIB_OK=TRUE` in entry points unless you are deliberately changing the native library stack.
- Checkpoints and retrieval artifacts are written under `finetuned_models/` and related local paths; avoid hard-coding new paths unless a script already owns them.
- Prefer small, script-level edits over broad refactors because most behavior lives directly in the entry points and helper modules.

## Validation Guidance

- There does not appear to be a dedicated automated test suite in the repo.
- For changes, prefer the narrowest useful check: run the touched script with a small input, or do a focused import / syntax check for the module you edited.

## Notes For Agents

- Link to existing docs instead of restating them.
- If recurring friction shows up during future work, consider `/chronicle improve` to refine these instructions over time.