# Copilot Instructions: Yakutlar

Purpose: Help AI coding agents work productively in this repo. Keep changes minimal, focused, and consistent with existing patterns.

## Architecture
- Single-file Gradio app: `app.py` builds the UI with two tabs: "Translate" (main) and "Compare".
- Translation backends via CTranslate2:
  - MADLAD/T5 (`T5_MODEL_DIR`, `T5_TOKENIZER_DIR`) with language tags like `<2sah>`
  - NLLB (`NLLB_MODEL_DIR`, `NLLB_HF_MODEL`) using `target_prefix` lang ids (e.g., `kaz_Cyrl`)
  - Optional Marian EN→mul models for tag-based languages (e.g., `>>chv<<`).
- Runtime caches for tokenizers/translators: `_t5_cache`, `_nllb_cache`, `_marian_cache`, `_comet_cache`.
- Feature highlights: auto-backend routing, dictionary overrides (`overrides.json`), Russian pivot path, reranking (roundtrip CharF1 or COMET-QE), Cyrillic ASCII-tail cleanup, Uyghur NGA transliteration.

## Key Files/Paths
- `app.py`: All UI and logic, including helpers (`_load_*`, rerankers, translit).
- `requirements.txt`: Minimal deps. Optional COMET/Torch commented out.
- `models/`: Expected local folders for CT2 models (not in repo). See constants at top of `app.py`.
- `overrides.json`: Optional local file for dictionary overrides (keyed by target).

## Developer Workflows
- Run locally:
  - `pip install -r requirements.txt`
  - `python app.py` → serves on `http://localhost:7860`
- Codespaces/mobile friendly:
  - App auto-enables Gradio share in Codespaces or when `GRADIO_SHARE=1`
  - Optional basic auth via `GRADIO_AUTH="user:pass"`
- Model availability:
  - Place CT2 models/tokenizers under `models/` matching defaults:
    - `models/nllb200-1.3b-ct2-int8`, `facebook/nllb-200-distilled-600M`
    - `models/madlad400-3b-mt-ct2-int8`, `models/madlad400-3b-mt`
    - Optional Marian: `models/opus-mt-en-mul-ct2-int8`, `models/opus-mt-en-mul`

## Conventions & Patterns
- Inputs/Outputs: Core translate fn returns `(out_text, translit, alternatives)` strings; always keep triple.
- Targets and routing:
  - `ALL_TARGETS` is derived from `T5_TURKIC` and `NLLB_TURKIC` with ordering: T5-only, both, NLLB-only.
  - Backend "auto" selects NLLB/MADLAD/Marian based on target and tag availability.
  - Chuvash forces Marian for single-word edge cases; also global early Marian path if tag is supported.
- Reranking:
  - `rerank_mode` supports `auto|none|roundtrip|cometqe`.
  - Auto: short (≤2 words) → none; long → COMET-QE if loadable else roundtrip; medium → roundtrip.
- Decoding knobs per target: `get_defaults_for_target` and `get_flag_defaults_for_target` feed UI defaults per `target` change.
- Script conformity: `_prefer_cyrillic_hyps` stable-reorders to prefer non-ASCII for Cyrillic targets.
- Cleanup: `strip_ascii_tail` applied by default to Cyrillic targets or when user sets checkbox.
- Uyghur: if target is Uyghur (Arabic), optional transliteration via `uyghur_arabic_to_latin_nga` appears in second output box.
- Dictionary overrides: When `dict_mode` is on and input has ≤2 words, lookup `overrides.json` sections keyed by target aliases (e.g., `chuvash`, `cv`, `chv`, `chv_cyrl`).

## Integration Points
- COMET-QE: set `comet_model` (default `Unbabel/wmt22-cometkiwi-da`). Requires installing comet + torch if actually used.
- Two-stage pivot: optional EN→RU (NLLB) n-best to target (T5) with reranking.
- Roundtrip scoring backend can be T5 or Marian big models (`rt_marian_*`).

## Making Changes Safely
- Keep API stable: do not change Gradio component signatures or return shapes.
- Avoid renaming constants/keys used across the UI (e.g., `ALL_TARGETS`, `T5_TURKIC` keys).
- Preserve caches and model-loading patterns (`_load_*`) to avoid repeated downloads.
- When adding a new language:
  - Add to the appropriate map(s): `T5_TURKIC` and/or `NLLB_TURKIC` with correct tag/id.
  - If Cyrillic-heavy, consider adding to `CYRILLIC_STRIP_DEFAULT`.
  - Provide decoding defaults in `DEFAULTS` and flag defaults via `get_flag_defaults_for_target` if needed.
- When adding a new backend or scoring method, plug into the existing rerank switch and follow the existing `translate_kwargs` structure.

## Quick Examples
- Add a T5-only language:
  - Update `T5_TURKIC["example"] = "<2ex>"`; `DEFAULTS["example"] = (3, 1.1, 1.05)`; optionally adjust `MADLAD_DEFAULT` membership.
- Add COMET usage at runtime:
  - Install deps, then set UI `rerank_mode = cometqe` and `comet_model` path or id.

## Non-Goals
- Do not introduce heavy frameworks or restructure into packages unless requested. The single-file UI is intentional for portability and Codespaces use.
