import gradio as gr
import ctranslate2
from transformers import AutoTokenizer
import json, os, re
import platform
import subprocess
import multiprocessing

# Apple Silicon auto device and threads configuration
def _detect_apple_silicon_and_configure_threads():
    """Detect Apple Silicon and set sensible thread defaults if not already configured."""
    if platform.system() == "Darwin":  # macOS
        try:
            # Check for Apple Silicon (arm64 architecture)
            arch_result = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True)
            if "arm64" in arch_result.stdout.strip():
                # Apple Silicon detected, determine performance core count
                core_count = None
                
                # Try sysctl for performance cores first
                try:
                    result = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], 
                                          capture_output=True, text=True, check=True)
                    core_count = int(result.stdout.strip())
                except (subprocess.CalledProcessError, ValueError):
                    # Fallback to total logical CPU count
                    try:
                        result = subprocess.run(["sysctl", "-n", "hw.ncpu"], 
                                              capture_output=True, text=True, check=True)
                        core_count = int(result.stdout.strip())
                    except (subprocess.CalledProcessError, ValueError):
                        # Final fallback to multiprocessing
                        core_count = multiprocessing.cpu_count()
                
                # Set thread environment variables if not already set
                if core_count and core_count > 0:
                    if "CT2_NUM_THREADS" not in os.environ:
                        os.environ["CT2_NUM_THREADS"] = str(core_count)
                    if "OMP_NUM_THREADS" not in os.environ:
                        os.environ["OMP_NUM_THREADS"] = str(core_count)
        except Exception:
            # Silently fail if detection doesn't work
            pass

# Configure Apple Silicon settings at module load
_detect_apple_silicon_and_configure_threads()

# Module-level defaults for CTranslate2 device and compute type
CT2_DEVICE = os.environ.get("CT2_DEVICE", "auto")
CT2_COMPUTE = os.environ.get("CT2_COMPUTE", "int8")

# Models and tokenizers
NLLB_MODEL_DIR = "models/nllb200-1.3b-ct2-int8"
NLLB_HF_MODEL = "facebook/nllb-200-distilled-600M"
T5_MODEL_DIR = "models/madlad400-3b-mt-ct2-int8"
T5_TOKENIZER_DIR = "models/madlad400-3b-mt"

# Language maps
NLLB_TURKIC = {
    "azerbaijani (north)": "azj_Latn",
    "azerbaijani (south)": "azb_Arab",
    "bashkir": "bak_Cyrl",
    "crimean tatar": "crh_Latn",
    "kazakh": "kaz_Cyrl",
    "kyrgyz": "kir_Cyrl",
    "tatar": "tat_Cyrl",
    "turkmen": "tuk_Latn",
    "uzbek (latin)": "uzn_Latn",
    "uyghur (arabic)": "uig_Arab",
}

T5_TURKIC = {
    "chuvash": "<2cv>",
    "yakut (sakha)": "<2sah>",
    "karachay-balkar": "<2krc>",
    "gagauz": "<2gag>",
    "nogai": "<2nog>",
    "tuvinian": "<2tyv>",
    "karakalpak": "<2kaa>",
    "southern altai": "<2alt>",
    # Breitere MADLAD-Unterstützung (ISO-639-1 Tags)
    "bashkir": "<2ba>",
    "turkmen": "<2tk>",
    "tatar": "<2tt>",
    "kazakh": "<2kk>",
    "kyrgyz": "<2ky>",
    "uzbek (latin)": "<2uz>",
    "crimean tatar": "<2crh>",
}

# Build unique target list (avoid duplicates like "turkmen" present in both maps)
_t5_only = [k for k in T5_TURKIC.keys() if k not in NLLB_TURKIC]
_nllb_only = [k for k in NLLB_TURKIC.keys() if k not in T5_TURKIC]
_both = [k for k in T5_TURKIC.keys() if k in NLLB_TURKIC]
ALL_TARGETS = _t5_only + _both + _nllb_only

# Routing lists for auto backend
NLLB_DEFAULT = {
    "azerbaijani (north)", "azerbaijani (south)", "bashkir", "crimean tatar",
    "kazakh", "kyrgyz", "tatar", "turkmen", "uzbek (latin)", "uyghur (arabic)"
}
MADLAD_DEFAULT = {
    "chuvash", "yakut (sakha)", "karachay-balkar", "gagauz", "nogai",
    "tuvinian", "karakalpak", "southern altai"
}

CYRILLIC_STRIP_DEFAULT = {
    "chuvash", "karachay-balkar", "nogai", "tuvinian", "bashkir",
    "kazakh", "kyrgyz", "tatar", "crimean tatar"
}

# Heuristic per-language defaults for decoding knobs (no_repeat, rep_pen, len_pen)
DEFAULTS = {
    # T5 group
    "chuvash": (3, 1.1, 1.05),
    "yakut (sakha)": (3, 1.1, 1.05),
    "karachay-balkar": (3, 1.2, 1.0),
    "gagauz": (3, 1.1, 1.0),
    "nogai": (4, 1.2, 1.05),
    "tuvinian": (3, 1.1, 1.05),
    "karakalpak": (3, 1.1, 1.0),
    "southern altai": (3, 1.15, 1.05),
    # NLLB group (generic)
    "azerbaijani (north)": (3, 1.1, 1.0),
    "azerbaijani (south)": (3, 1.1, 1.0),
    "bashkir": (3, 1.1, 1.0),
    "crimean tatar": (3, 1.1, 1.0),
    "kazakh": (3, 1.1, 1.0),
    "kyrgyz": (3, 1.1, 1.0),
    "tatar": (3, 1.1, 1.0),
    "turkmen": (3, 1.1, 1.0),
    "uzbek (latin)": (3, 1.1, 1.0),
    "uyghur (arabic)": (3, 1.1, 1.0),
}

def get_defaults_for_target(target_label: str):
    return DEFAULTS.get(target_label.lower(), (3, 1.1, 1.0))

def get_flag_defaults_for_target(target_label: str):
    t = target_label.lower()
    strip_default = t in CYRILLIC_STRIP_DEFAULT
    # Enable dictionary overrides by default for low-resource MADLAD targets like Chuvash/Yakut
    dict_default = t in {"chuvash", "yakut (sakha)"}
    return strip_default, dict_default

# Caches
_nllb_cache = {}
_t5_cache = {}
_marian_cache = {}
_marian_enmul_supported = None
_comet_cache = {}  # Cache COMET models to avoid reloading


def _char_f1(a: str, b: str) -> float:
    """Compute character-level F1 score between two strings."""
    if not a or not b:
        return 0.0
    import collections
    ca, cb = collections.Counter(a), collections.Counter(b)
    inter = sum((ca & cb).values())
    p = inter / max(1, len(b))
    r = inter / max(1, len(a))
    return (2 * p * r / (p + r)) if (p > 0 and r > 0) else 0.0


def _get_comet_model(model_id_or_path: str):
    """Get COMET model from cache or load it."""
    if model_id_or_path in _comet_cache:
        return _comet_cache[model_id_or_path]
    
    try:
        import os
        from comet import download_model, load_from_checkpoint
        
        # Prefer local path if it exists
        if os.path.exists(model_id_or_path):
            mpath = model_id_or_path
        else:
            mpath = download_model(model_id_or_path)
        
        comet_model = load_from_checkpoint(mpath)
        _comet_cache[model_id_or_path] = comet_model
        return comet_model
    except Exception as e:
        print(f"[UI] Failed to load COMET model {model_id_or_path}: {e}", flush=True)
        return None


def _comet_predict_scores(comet_model, src_text: str, hyps: list):
    """Predict COMET scores for a list of hypotheses."""
    if not comet_model or not hyps:
        return None
    
    try:
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
        data = [{"src": src_text, "mt": h} for h in hyps]
        ret = comet_model.predict(data, batch_size=8, gpus=gpus, num_workers=1)
        scores = ret.get("scores") if isinstance(ret, dict) else getattr(ret, "scores", None)
        return scores
    except Exception as e:
        print(f"[UI] COMET prediction failed: {e}", flush=True)
        return None


def _compute_roundtrip_charf1(text: str, hyps: list, roundtrip_backend: str, rt_marian_model_dir: str, rt_marian_tokenizer_dir: str):
    """Compute roundtrip CharF1 scores for a list of hypotheses."""
    if not hyps:
        return []
    
    scored = []
    
    if roundtrip_backend == "marian_big" and rt_marian_model_dir and rt_marian_tokenizer_dir:
        try:
            tok_b, tr_b = _load_marian(rt_marian_model_dir, rt_marian_tokenizer_dir)
            for h in hyps:
                ids_b = tok_b.encode(h)
                toks_b = tok_b.convert_ids_to_tokens(ids_b)
                res_b = tr_b.translate_batch([toks_b], beam_size=8)
                back = tok_b.decode(tok_b.convert_tokens_to_ids(res_b[0].hypotheses[0]), skip_special_tokens=True)
                scored.append((_char_f1(text, back), h))
        except Exception as e:
            print(f"[UI] Marian roundtrip failed: {e}", flush=True)
            return [(0.0, h) for h in hyps]
    else:
        # Use T5 for roundtrip
        try:
            tok_t5, tr_t5 = _load_t5()
            en_tag = "<2en>"
            for h in hyps:
                try:
                    _ = tok_t5.convert_tokens_to_ids([en_tag])[0]
                    pre = f"{en_tag} {h}"
                except Exception:
                    pre = f"Translate to English: {h}"
                ids_b = tok_t5.encode(pre)
                toks_b = tok_t5.convert_ids_to_tokens(ids_b)
                res_b = tr_t5.translate_batch([toks_b], beam_size=8)
                back = tok_t5.decode(tok_t5.convert_tokens_to_ids(res_b[0].hypotheses[0]), skip_special_tokens=True)
                scored.append((_char_f1(text, back), h))
        except Exception as e:
            print(f"[UI] T5 roundtrip failed: {e}", flush=True)
            return [(0.0, h) for h in hyps]
    
    return scored


def _load_marian(model_dir: str, tokenizer_dir: str):
    key = (model_dir, tokenizer_dir)
    if key not in _marian_cache:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        tr = ctranslate2.Translator(model_dir, device=CT2_DEVICE, compute_type=CT2_COMPUTE)
        _marian_cache[key] = (tok, tr)
    return _marian_cache[key]


def _load_nllb(src_lang: str):
    key = src_lang
    if key not in _nllb_cache:
        tok = AutoTokenizer.from_pretrained(NLLB_HF_MODEL, src_lang=src_lang)
        tr = ctranslate2.Translator(NLLB_MODEL_DIR, device=CT2_DEVICE, compute_type=CT2_COMPUTE)
        _nllb_cache[key] = (tok, tr)
    return _nllb_cache[key]


def _load_t5():
    if "obj" not in _t5_cache:
        tok = AutoTokenizer.from_pretrained(T5_TOKENIZER_DIR)
        tr = ctranslate2.Translator(T5_MODEL_DIR, device=CT2_DEVICE, compute_type=CT2_COMPUTE)
        _t5_cache["obj"] = (tok, tr)
    return _t5_cache["obj"]


def uyghur_arabic_to_latin_nga(text: str) -> str:
    vowels = {
        "ا": "a",
        "ە": "e",
        "ې": "ë",
        "ى": "i",
        "و": "o",
        "ۇ": "u",
        "ۆ": "ö",
        "ۈ": "ü",
    }
    HAMZA = "ئ"
    cons = {
        "ب": "b", "پ": "p", "ت": "t", "ج": "j", "چ": "ch", "ح": "h", "خ": "x", "د": "d",
        "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "غ": "gh", "ف": "f", "ق": "q",
        "ك": "k", "ک": "k", "گ": "g", "ڭ": "ng", "ن": "n", "م": "m", "ل": "l", "ه": "h",
        "ھ": "h", "ء": "ʼ", "ع": "ʼ", "ي": "y", "ۋ": "w",
    }
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == HAMZA and nxt in vowels:
            out.append("ʼ" + vowels[nxt])
            i += 2
            continue
        if ch in vowels:
            out.append(vowels[ch])
            i += 1
            continue
        if ch in cons:
            out.append(cons[ch])
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


OVERRIDES_PATH = "overrides.json"


def load_overrides(path: str):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def strip_ascii_tail(text: str) -> str:
    # Entfernt ein letztes ASCII-Wort (ggf. Bindestrich) am Ende (auch ohne Leerzeichen davor)
    return re.sub(r"\s*[A-Za-z][A-Za-z\-]*$", "", text).strip()


def _prefer_cyrillic_hyps(hyps, target_l):
    """Stable-reorder: prefer hypotheses without ASCII letters for Cyrillic-heavy targets.
    No lexical knowledge, just script conformity.
    """
    CYR_TGT = {"chuvash", "karachay-balkar", "nogai", "tuvinian", "bashkir", "kazakh", "kyrgyz", "tatar"}
    if (target_l or "").lower() not in CYR_TGT:
        return hyps
    def has_ascii(s: str) -> bool:
        return bool(re.search(r"[A-Za-z]", s))
    # Stable partition: keep original order within groups
    cyr = [h for h in hyps if not has_ascii(h)]
    rest = [h for h in hyps if has_ascii(h)]
    return cyr + rest


def _marian_single_word_chuvash(src_text: str) -> str:
    try:
        tok = _marian_cache.get("tok")
        tr = _marian_cache.get("tr")
        if tok is None or tr is None:
            tok = AutoTokenizer.from_pretrained("models/opus-en-trk")
            tr = ctranslate2.Translator("models/opus-en-trk-ct2-int8", device=CT2_DEVICE, compute_type=CT2_COMPUTE)
            _marian_cache["tok"] = tok
            _marian_cache["tr"] = tr
        prep = ">>chv<< " + src_text
        ids = tok.encode(prep)
        toks = tok.convert_ids_to_tokens(ids)
        res = tr.translate_batch([toks], beam_size=8)
        out_tokens = res[0].hypotheses[0]
        return tok.decode(tok.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)
    except Exception:
        return ""


def translate(text: str, target: str, backend: str, beam_size: int, no_repeat_ngram_size: int, repetition_penalty: float, length_penalty: float, dict_mode: bool, strip_ascii: bool, show_uyghur_nga: bool, topk: int, enable_sampling: bool, temperature: float, topp: float, pivot_ru: bool, pivot_ru_backend: str, rerank_mode: str, comet_model: str, marian_model_dir: str, marian_tokenizer_dir: str, roundtrip_backend: str, rt_marian_model_dir: str, rt_marian_tokenizer_dir: str, two_stage_pivot: bool, pivot_ru_nbest: int, allow_madlad_auto: bool):
    # Enforce Marian for Chuvash regardless of selected backend
    if (target or "").lower() == "chuvash":
        backend = "marian"

    # Dynamic routing: prefer Marian en->mul for targets with supported tags
    def _marian_enmul_has_tag(tag: str) -> bool:
        # Prefer local tokenizer dir when available (robustness improvement)
        tokenizer_path = marian_tokenizer_dir if marian_tokenizer_dir else "Helsinki-NLP/opus-mt-en-mul"
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(tokenizer_path)
            tid = tok.convert_tokens_to_ids([tag])[0]
            unk = getattr(tok, "unk_token_id", None)
            return (unk is None) or (tid != unk)
        except Exception:
            # Fallback to online if local fails
            if tokenizer_path != "Helsinki-NLP/opus-mt-en-mul":
                try:
                    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
                    tid = tok.convert_tokens_to_ids([tag])[0]
                    unk = getattr(tok, "unk_token_id", None)
                    return (unk is None) or (tid != unk)
                except Exception:
                    pass
            return False

    tgt_l = (target or "").lower()
    TAGS = {
        "chuvash": ">>chv<<",
        "cv": ">>chv<<",
        "tyv": ">>tyv<<",
        "tuvinian": ">>tyv<<",
        "krc": ">>krc<<",
        "gag": ">>gag<<",
        "nog": ">>nog<<",
        "kaa": ">>kaa<<",
        "alt": ">>alt<<",
    }
    tag = TAGS.get(tgt_l)
    if backend in {"auto", "t5", "nllb"} and tag and _marian_enmul_has_tag(tag):
        backend = "marian"
    text = (text or "").strip()
    if not text:
        return "", "", ""  # Fix: return 3 values to match UI outputs

    target_l = target.lower()

    # Auto rerank logic: resolve "auto" mode based on input characteristics
    _rr_mode = rerank_mode
    if not rerank_mode or rerank_mode.lower() == "auto":
        try:
            short_input = len((text or "").split()) <= 2
            token_count = len((text or "").split())
            text_len = len(text or "")
        except Exception:
            short_input = True
            token_count = 0
            text_len = 0
        
        if short_input:
            # Very short input: do not rerank, prefer dictionary/override logic
            _rr_mode = "none"
        elif token_count >= 5 or text_len >= 20:
            # Longer input: try COMET-QE first, fallback to roundtrip
            comet_model_name = comet_model or "Unbabel/wmt22-cometkiwi-da"
            comet = _get_comet_model(comet_model_name)
            _rr_mode = "cometqe" if comet else "roundtrip"
        else:
            # Medium input: fallback to roundtrip
            _rr_mode = "roundtrip"

    # For very short inputs to Chuvash, raise beam and disable sampling
    try:
        short_input = len((text or "").split()) <= 2
    except Exception:
        short_input = False
    if target_l in {"chuvash", "cv", "chv"} and short_input:
        beam_size = max(beam_size, 24)
        enable_sampling = False

    # Optional Dictionary override (≤2 Wörter)
    if dict_mode and len(text.split()) <= 2:
        ov = load_overrides(OVERRIDES_PATH)
        tk = target_l
        # einfache Synonyme für Ziel
        if tk in {"chuvash", "cv", "chv"}:
            keys = ["chuvash", "cv", "chv", "chv_cyrl"]
        elif tk in {"yakut", "sakha"}:
            keys = ["yakut", "sakha", "sah", "sah_cyrl"]
        else:
            keys = [tk]
        src_key = text.strip().lower()
        for k in keys:
            if k in ov and src_key in ov[k]:
                out_text = ov[k][src_key]
                return out_text, "", ""

    # Backend selection with strict Auto routing
    use_t5 = False
    auto_route_message = ""
    
    if backend == "t5":
        use_t5 = True
    elif backend == "nllb":
        use_t5 = False
    elif backend == "marian":
        # Explicit Marian selection - will be handled later
        pass  
    else:  # auto
        # Strict Auto routing order: NLLB > Marian > MADLAD (with toggle)
        target_l_lower = target_l.lower()
        
        # 1. Check if target is supported by NLLB
        if target_l_lower in NLLB_TURKIC:
            use_t5 = False
        else:
            # 2. Check if Marian EN→MUL tag exists for target and model/tokenizer are configured
            marian_available = False
            if marian_model_dir and marian_tokenizer_dir:
                # Check for supported Marian tags
                tag_map = {
                    "chuvash": ">>chv<<", "cv": ">>chv<<",
                    "tyv": ">>tyv<<", "tuvinian": ">>tyv<<",
                    "krc": ">>krc<<", "gag": ">>gag<<", "nog": ">>nog<<",
                    "kaa": ">>kaa<<", "alt": ">>alt<<",
                }
                tag = tag_map.get(target_l_lower)
                if tag:
                    # Check if tag exists in tokenizer
                    def _marian_enmul_has_tag(tag: str) -> bool:
                        try:
                            from transformers import AutoTokenizer
                            tok = AutoTokenizer.from_pretrained(marian_tokenizer_dir)
                            tid = tok.convert_tokens_to_ids([tag])[0]
                            unk = getattr(tok, "unk_token_id", None)
                            return (unk is None) or (tid != unk)
                        except Exception:
                            return False
                    
                    if _marian_enmul_has_tag(tag):
                        marian_available = True
            
            if marian_available:
                # Route to Marian (will set backend = "marian" later)
                pass
            else:
                # 3. Check MADLAD/T5 with toggle
                if target_l_lower in T5_TURKIC:
                    if allow_madlad_auto:
                        use_t5 = True
                    else:
                        # MADLAD blocked by toggle - return clear message
                        auto_route_message = f"Auto mode: Target '{target}' requires MADLAD/T5 backend. Enable 'Allow MADLAD in Auto' toggle or select T5 backend explicitly."
                        return auto_route_message, "", ""
                else:
                    # Target not supported by any backend
                    auto_route_message = f"Auto mode: Target '{target}' is not supported by available backends."
                    return auto_route_message, "", ""

    # Route to Marian for explicit backend selection, Auto routing decision, or Chuvash hard force
    marian_route = (backend == "marian" or 
                   (backend == "auto" and target_l in {"chuvash"} and marian_model_dir and marian_tokenizer_dir) or
                   (backend == "auto" and 'marian_available' in locals() and marian_available))
    
    if marian_route:
        # Marian EN->Target with language tag (e.g., >>chv<<). Pivot is ignored here (model expects EN source).
        tag_map = {
            "chuvash": ">>chv<<",
            "cv": ">>chv<<",
            "tyv": ">>tyv<<",
            "tuvinian": ">>tyv<<",
            "krc": ">>krc<<",
            "gag": ">>gag<<",
            "nog": ">>nog<<",
            "kaa": ">>kaa<<",
            "alt": ">>alt<<",
        }
        tag = tag_map.get(target_l, None)
        tok_m, tr_m = _load_marian(marian_model_dir, marian_tokenizer_dir)
        prep = f"{tag} {text}" if tag else text
        ids = tok_m.encode(prep)
        toks = tok_m.convert_ids_to_tokens(ids)
        translate_kwargs = dict(
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 0,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            length_penalty=length_penalty if length_penalty else 1.0,
            num_hypotheses=max(1, int(topk)),
            return_scores=(topk and int(topk) > 1),
        )
        res = tr_m.translate_batch([toks], **translate_kwargs)
        hyps = [tok_m.decode(tok_m.convert_tokens_to_ids(h), skip_special_tokens=True) for h in res[0].hypotheses[:max(1,int(topk))]]
        # Rerank if configured using unified Auto logic
        if _rr_mode == "roundtrip" and len(hyps) > 1:
            scored = _compute_roundtrip_charf1(text, hyps, roundtrip_backend, rt_marian_model_dir, rt_marian_tokenizer_dir)
            scored.sort(key=lambda x: x[0], reverse=True)
            hyps = [h for _, h in scored]
        elif _rr_mode == "cometqe" and len(hyps) > 1:
            cm = _get_comet_model(comet_model or "Unbabel/wmt22-cometkiwi-da")
            if cm:
                scores = _comet_predict_scores(cm, text, hyps)
                if scores:
                    order = sorted(range(len(hyps)), key=lambda j: scores[j], reverse=True)
                    hyps = [hyps[j] for j in order]
        # Script conformity tie-breaker for Cyrillic-heavy targets
        hyps = _prefer_cyrillic_hyps(hyps, target_l)
        out_text = hyps[0]
        alts = [f"{i+2}) {h}" for i, h in enumerate(hyps[1:])]
        return out_text, "", "\n".join(alts)

    if use_t5:
        tag = T5_TURKIC[target_l]
        tok, tr = _load_t5()
        # Optional: Russisch-Pivot EN->RU->Ziel
        if pivot_ru:
            if pivot_ru_backend == "nllb":
                # EN->RU via NLLB (more robust pivot)
                src_lang = "eng_Latn"
                tok_n, tr_n = _load_nllb(src_lang)
                ids_en = tok_n.encode(text)
                toks_en = tok_n.convert_ids_to_tokens(ids_en)
                if two_stage_pivot:
                    res_ru = tr_n.translate_batch([toks_en], target_prefix=[["rus_Cyrl"]], beam_size=max(12, beam_size), num_hypotheses=max(1,int(pivot_ru_nbest)), return_scores=True)
                    ru_list = []
                    for tk in res_ru[0].hypotheses:
                        if tk and tk[0] == "rus_Cyrl":
                            tk = tk[1:]
                        ru_list.append(tok_n.decode(tok_n.convert_tokens_to_ids(tk), skip_special_tokens=True))
                    # translate each RU candidate to target with T5 now (skip normal path below)
                    # get target tag
                    from typing import Optional
                    try:
                        ttag = T5_TURKIC[(target or "").lower()]
                    except KeyError:
                        ttag = None
                    # force tag for Sakha
                    if (target or "").lower() in {"yakut (sakha)", "yakut", "sah", "sakha"}:
                        ttag = "<2sah>"
                    preps = [f"{ttag} {r}" if ttag else r for r in ru_list]
                    ids_list = [tok.encode(p) for p in preps]
                    toks_list = [tok.convert_ids_to_tokens(x) for x in ids_list]
                    res_t = tr.translate_batch(toks_list, beam_size=beam_size, num_hypotheses=1, return_scores=True)
                    hyps = [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res_t]
                    # Heuristic: drop Russian-looking outputs for Sakha goal
                    if (target or "").lower() in {"yakut (sakha)", "yakut", "sah", "sakha"}:
                        sah_chars = set("үөһҕҥ")
                        rus_kw = ["бел", "туфл", "ботин", "обув"]
                        def _looks_russian(txt:str)->bool:
                            return any(k in txt.lower() for k in rus_kw) and not any(ch in sah_chars for ch in txt)
                        hyps2 = [h for h in hyps if not _looks_russian(h)]
                        if hyps2:
                            hyps = hyps2
                    # Rerank by selected mode using unified Auto logic
                    if _rr_mode == "cometqe" and len(hyps) > 1:
                        cm = _get_comet_model(comet_model or "Unbabel/wmt22-cometkiwi-da")
                        if cm:
                            scores = _comet_predict_scores(cm, text, hyps)
                            if scores:
                                order = sorted(range(len(hyps)), key=lambda j: scores[j], reverse=True)
                                hyps = [hyps[j] for j in order]
                    elif _rr_mode == "roundtrip" and len(hyps) > 1:
                        scored = _compute_roundtrip_charf1(text, hyps, roundtrip_backend, rt_marian_model_dir, rt_marian_tokenizer_dir)
                        if scored:
                            scored.sort(key=lambda x: x[0], reverse=True)
                            hyps = [h for _, h in scored]
                    out_text = hyps[0]
                    alts = [f"{i+2}) {h}" for i, h in enumerate(hyps[1:])]
                    return out_text, "", "\n".join(alts)
                else:
                    res_ru = tr_n.translate_batch([toks_en], target_prefix=[["rus_Cyrl"]], beam_size=max(12, beam_size))
                    ru_tokens = res_ru[0].hypotheses[0]
                    if ru_tokens and ru_tokens[0] == "rus_Cyrl":
                        ru_tokens = ru_tokens[1:]
                    ru_text = tok_n.decode(tok_n.convert_tokens_to_ids(ru_tokens), skip_special_tokens=True)
            else:
                # EN->RU via T5
                ru_tag = "<2ru>"
                try:
                    _ = tok.convert_tokens_to_ids([ru_tag])[0]
                    pre_ru = f"{ru_tag} {text}"
                except Exception:
                    pre_ru = f"Translate to Russian: {text}"
                ids_ru = tok.encode(pre_ru)
                toks_ru = tok.convert_ids_to_tokens(ids_ru)
                res_ru = tr.translate_batch([toks_ru], beam_size=max(12, beam_size))
                ru_text = tok.decode(tok.convert_tokens_to_ids(res_ru[0].hypotheses[0]), skip_special_tokens=True)
            # Zweite Stufe RU->Ziel
            prep = f"{tag} {ru_text}"
        else:
            prep = f"{tag} {text}"
        ids = tok.encode(prep)
        toks = tok.convert_ids_to_tokens(ids)
        translate_kwargs = dict(
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 0,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            length_penalty=length_penalty if length_penalty else 1.0,
            num_hypotheses=max(1, int(topk)),
            return_scores=(topk and int(topk) > 1),
        )
        if enable_sampling:
            translate_kwargs.update(dict(sampling_temperature=temperature or 1.0, sampling_topp=topp or 1.0))
        res = tr.translate_batch([toks], **translate_kwargs)
        # Sammle Hypothesen (Top-k)
        hyps = [tok.decode(tok.convert_tokens_to_ids(h), skip_special_tokens=True) for h in res[0].hypotheses[:max(1,int(topk))]]
        # Reranking using unified Auto logic
        if _rr_mode == "roundtrip" and len(hyps) > 1:
            scored = _compute_roundtrip_charf1(text, hyps, roundtrip_backend, rt_marian_model_dir, rt_marian_tokenizer_dir)
            scored.sort(key=lambda x: x[0], reverse=True)
            hyps = [h for _, h in scored]
        elif _rr_mode == "cometqe" and len(hyps) > 1:
            cm = _get_comet_model(comet_model or "Unbabel/wmt22-cometkiwi-da")
            if cm:
                scores = _comet_predict_scores(cm, text, hyps)
                if scores:
                    order = sorted(range(len(hyps)), key=lambda j: scores[j], reverse=True)
                    hyps = [hyps[j] for j in order]
        # Script conformity tie-breaker for Cyrillic-heavy targets
        hyps = _prefer_cyrillic_hyps(hyps, target_l)
        out_text = hyps[0]
        # Optional Fallback für Einwort Chuvash, falls ASCII anhängt oder zu lang
        if len(text.split()) <= 2 and target_l in {"chuvash","cv","chv"}:
            if re.search(r"[A-Za-z]", out_text) or len(out_text.split()) > 2:
                fb = _marian_single_word_chuvash(text.strip())
                if fb:
                    out_text = fb
        # ASCII cleanup
        if (target_l in CYRILLIC_STRIP_DEFAULT) or strip_ascii:
            out_text = strip_ascii_tail(out_text)
        # Alternatives (restliche Hyps)
        alts = []
        if len(hyps) > 1:
            for j, detok in enumerate(hyps[1:]):
                if strip_ascii and target_l in {"chuvash", "karachay-balkar", "nogai", "tuvinian"}:
                    detok = strip_ascii_tail(detok)
                if detok and detok != out_text:
                    alts.append(f"{j+2}) {detok}")
        return out_text, "", "\n".join(alts)
    else:
        src_lang = "eng_Latn"  # fixed source
        tgt_lang = NLLB_TURKIC[target_l]
        tok, tr = _load_nllb(src_lang)
        ids = tok.encode(text)
        toks = tok.convert_ids_to_tokens(ids)
        translate_kwargs = dict(
            target_prefix=[[tgt_lang]],
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 0,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            length_penalty=length_penalty if length_penalty else 1.0,
            num_hypotheses=max(1, int(topk)),
            return_scores=(topk and int(topk) > 1),
        )
        if enable_sampling:
            translate_kwargs.update(dict(sampling_temperature=temperature or 1.0, sampling_topp=topp or 1.0))
        res = tr.translate_batch([toks], **translate_kwargs)
        # Sammle Hypothesen
        hyps_tok = res[0].hypotheses[:max(1,int(topk))]
        hyps = []
        for tok_seq in hyps_tok:
            if tok_seq and tok_seq[0] == tgt_lang:
                tok_seq = tok_seq[1:]
            hyps.append(tok.decode(tok.convert_tokens_to_ids(tok_seq), skip_special_tokens=True))
        # Reranking using unified Auto logic
        if _rr_mode == "roundtrip" and len(hyps) > 1:
            scored = _compute_roundtrip_charf1(text, hyps, roundtrip_backend, rt_marian_model_dir, rt_marian_tokenizer_dir)
            scored.sort(key=lambda x: x[0], reverse=True)
            hyps = [h for _, h in scored]
        elif _rr_mode == "cometqe" and len(hyps) > 1:
            cm = _get_comet_model(comet_model or "Unbabel/wmt22-cometkiwi-da")
            if cm:
                scores = _comet_predict_scores(cm, text, hyps)
                if scores:
                    order = sorted(range(len(hyps)), key=lambda j: scores[j], reverse=True)
                    hyps = [hyps[j] for j in order]
        # Script conformity tie-breaker for Cyrillic-heavy targets
        hyps = _prefer_cyrillic_hyps(hyps, target_l)
        out_text = hyps[0]
        # ASCII cleanup
        if (target_l in CYRILLIC_STRIP_DEFAULT) or strip_ascii or (tgt_lang.endswith("_Cyrl")):
            out_text = strip_ascii_tail(out_text)
        # Alternatives
        alts = []
        for j, detok in enumerate(hyps[1:]):
            if strip_ascii and (tgt_lang.endswith("_Cyrl") or target_l in {"chuvash", "karachay-balkar", "nogai", "tuvinian"}):
                detok = strip_ascii_tail(detok)
            if detok and detok != out_text:
                alts.append(f"{j+2}) {detok}")
        # Optional NGA transliteration for Uyghur
        if show_uyghur_nga and target_l == "uyghur (arabic)":
            return out_text, uyghur_arabic_to_latin_nga(out_text), "\n".join(alts)
        return out_text, "", "\n".join(alts)


def translate_compare(text: str, target: str, beam_size: int, no_repeat_ngram_size: int, repetition_penalty: float, length_penalty: float, show_uyg_nga: bool):
    text = (text or "").strip()
    if not text:
        return "", "", ""
    target_l = target.lower()

    # MADLAD side
    if target_l in T5_TURKIC:
        tag = T5_TURKIC[target_l]
        tok_t5, tr_t5 = _load_t5()
        prep = f"{tag} {text}"
        ids = tok_t5.encode(prep)
        toks = tok_t5.convert_ids_to_tokens(ids)
        res = tr_t5.translate_batch(
            [toks],
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 0,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            length_penalty=length_penalty if length_penalty else 1.0,
        )
        out_t5 = tok_t5.decode(tok_t5.convert_tokens_to_ids(res[0].hypotheses[0]), skip_special_tokens=True)
    else:
        out_t5 = "Not supported by MADLAD"

    # NLLB side
    if target_l in NLLB_TURKIC:
        src_lang = "eng_Latn"
        tgt_lang = NLLB_TURKIC[target_l]
        tok_n, tr_n = _load_nllb(src_lang)
        ids = tok_n.encode(text)
        toks = tok_n.convert_ids_to_tokens(ids)
        res = tr_n.translate_batch(
            [toks],
            target_prefix=[[tgt_lang]],
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 0,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            length_penalty=length_penalty if length_penalty else 1.0,
        )
        out_tokens = res[0].hypotheses[0]
        if out_tokens and out_tokens[0] == tgt_lang:
            out_tokens = out_tokens[1:]
        out_nllb = tok_n.decode(tok_n.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)
        out_ng = uyghur_arabic_to_latin_nga(out_nllb) if (show_uyg_nga and target_l == "uyghur (arabic)") else ""
    else:
        out_nllb = "Not supported by NLLB"
        out_ng = ""

    return out_t5, out_nllb, out_ng


def build_ui():
    with gr.Blocks(title="Turkic MT (NLLB + MADLAD/T5)") as demo:
        gr.Markdown("# Turkic Machine Translation\nSource: English")
        with gr.Tabs():
            with gr.TabItem("Translate"):
                with gr.Row():
                    target = gr.Dropdown(choices=ALL_TARGETS, value="chuvash", label="Target language")
                    backend = gr.Dropdown(choices=["auto", "t5", "nllb", "marian"], value="auto", label="Backend")
                    beam = gr.Slider(4, 32, value=12, step=1, label="Beam size")
                inp = gr.Textbox(lines=4, placeholder="Enter English text…", label="English source")
                # Initialize defaults based on the default target ("chuvash")
                _n_def, _r_def, _l_def = get_defaults_for_target("chuvash")
                _strip_def, _dict_def = get_flag_defaults_for_target("chuvash")
                with gr.Accordion("Advanced decoding (optional)", open=False):
                    with gr.Row():
                        no_repeat = gr.Slider(0, 7, value=_n_def, step=1, label="no_repeat_ngram_size (0=off)", info="Verbietet Wiederholung gleicher n‑Gramme; 3 ist ein guter Start")
                        rep_pen = gr.Slider(1.0, 1.5, value=_r_def, step=0.05, label="repetition_penalty", info=">1.0 bestraft Wiederholungen; 1.1–1.2 üblich")
                        len_pen = gr.Slider(0.8, 1.5, value=_l_def, step=0.05, label="length_penalty", info=">1.0 erzeugt tendenziell vollständigere/etwas längere Sätze")
                with gr.Row():
                    dict_mode = gr.Checkbox(value=_dict_def, label="Dictionary mode (≤2 words overrides)")
                    strip_ascii = gr.Checkbox(value=_strip_def, label="Strip ASCII tail for Cyrillic targets")
                    show_uyg = gr.Checkbox(value=True, label="Show Uyghur NGA transliteration (when target is Uyghur)")
                btn = gr.Button("Translate")
                out = gr.Textbox(lines=6, label="Translation")
                out_ng = gr.Textbox(lines=4, label="Transliteration (NGA)")
                with gr.Accordion("Alternatives (Top-k)", open=False):
                    with gr.Row():
                        topk = gr.Slider(1, 5, value=1, step=1, label="Top-k hypotheses", info="Zeigt bis zu k beste Hypothesen (Beam)")
                        enable_sampling = gr.Checkbox(value=False, label="Enable sampling (advanced)")
                        temperature = gr.Slider(0.6, 1.4, value=1.0, step=0.05, label="temperature")
                        topp = gr.Slider(0.8, 1.0, value=1.0, step=0.01, label="nucleus top-p")
                    alts_box = gr.Textbox(lines=6, label="Alternatives")
                with gr.Row():
                    pivot_ru = gr.Checkbox(value=False, label="Use Russian pivot")
                    pivot_ru_backend = gr.Dropdown(choices=["nllb", "t5"], value="nllb", label="Pivot RU backend")
                with gr.Row():
                    rerank_mode = gr.Dropdown(choices=["auto", "none", "roundtrip", "cometqe"], value="auto", label="Rerank mode")
                    comet_model = gr.Textbox(value="Unbabel/wmt22-cometkiwi-da", label="COMET model", scale=2)
                with gr.Row():
                    allow_madlad_auto = gr.Checkbox(value=False, label="Allow MADLAD in Auto", info="Enables MADLAD/T5 as fallback in Auto mode when NLLB/Marian unavailable")
                with gr.Accordion("Marian settings (optional)", open=False):
                    with gr.Row():
                        marian_model_dir = gr.Textbox(value="models/opus-mt-en-mul-ct2-int8", label="Marian CT2 model dir", scale=2)
                        marian_tokenizer_dir = gr.Textbox(value="models/opus-mt-en-mul", label="Marian tokenizer dir", scale=2)
                with gr.Accordion("Roundtrip settings", open=False):
                    with gr.Row():
                        roundtrip_backend = gr.Dropdown(choices=["t5", "marian_big"], value="t5", label="Roundtrip backend")
                    with gr.Row():
                        rt_marian_model_dir = gr.Textbox(value="models/opus-mt-mul-en-big-ct2-int8", label="RT Marian CT2 model dir", scale=2)
                        rt_marian_tokenizer_dir = gr.Textbox(value="models/opus-mt-mul-en-big", label="RT Marian tokenizer dir", scale=2)
                with gr.Accordion("Two-stage pivot (experimental)", open=False):
                    with gr.Row():
                        two_stage_pivot = gr.Checkbox(value=False, label="Enable two-stage pivot rerank")
                        pivot_ru_nbest = gr.Slider(1, 10, value=5, step=1, label="Pivot RU N-best")
                btn.click(
                    fn=translate,
                    inputs=[inp, target, backend, beam, no_repeat, rep_pen, len_pen, dict_mode, strip_ascii, show_uyg, topk, enable_sampling, temperature, topp, pivot_ru, pivot_ru_backend, rerank_mode, comet_model, marian_model_dir, marian_tokenizer_dir, roundtrip_backend, rt_marian_model_dir, rt_marian_tokenizer_dir, two_stage_pivot, pivot_ru_nbest, allow_madlad_auto],
                    outputs=[out, out_ng, alts_box],
                )
                def _defaults(t):
                    n, r, l = get_defaults_for_target(t)
                    strip_def, dict_def = get_flag_defaults_for_target(t)
                    return n, r, l, strip_def, dict_def
                target.change(fn=_defaults, inputs=[target], outputs=[no_repeat, rep_pen, len_pen, strip_ascii, dict_mode])

                # Preset helper
                def _presets(t):
                    tl = (t or "").lower()
                    # Defaults
                    beam_v, topk_v, rr_mode, piv_on, piv_backend = 12, 1, "none", False, "nllb"
                    if tl in {"chuvash", "cv"}:
                        beam_v, topk_v, rr_mode, piv_on, piv_backend = 24, 5, "auto", False, "nllb"
                    elif tl in {"tyv", "tuvinian"}:
                        beam_v, topk_v, rr_mode, piv_on, piv_backend = 24, 5, "auto", False, "nllb"
                    elif tl in {"yakut", "sah", "sakha"}:
                        beam_v, topk_v, rr_mode, piv_on, piv_backend = 32, 10, "auto", False, "nllb"  # RU-pivot default OFF
                    return beam_v, topk_v, rr_mode, piv_on, piv_backend
                def _apply_presets(t):
                    b, k, rm, piv, pivb = _presets(t)
                    return b, k, rm, piv, pivb
                target.change(fn=_apply_presets, inputs=[target], outputs=[beam, topk, rerank_mode, pivot_ru, pivot_ru_backend])

            with gr.TabItem("Compare (MADLAD vs NLLB)"):
                with gr.Row():
                    target_c = gr.Dropdown(choices=ALL_TARGETS, value="bashkir", label="Target language")
                    beam_c = gr.Slider(4, 32, value=12, step=1, label="Beam size")
                inp_c = gr.Textbox(lines=4, placeholder="Enter English text…", label="English source")
                with gr.Accordion("Advanced decoding (optional)", open=False):
                    with gr.Row():
                        no_repeat_c = gr.Slider(0, 7, value=3, step=1, label="no_repeat_ngram_size (0=off)", info="Verbietet Wiederholung gleicher n‑Gramme; 3 ist ein guter Start")
                        rep_pen_c = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="repetition_penalty", info=">1.0 bestraft Wiederholungen; 1.1–1.2 üblich")
                        len_pen_c = gr.Slider(0.8, 1.5, value=1.05, step=0.05, label="length_penalty", info=">1.0 erzeugt tendenziell vollständigere/etwas längere Sätze")
                show_uyg_c = gr.Checkbox(value=True, label="Show Uyghur NGA transliteration (for NLLB, if applicable)")
                btn_c = gr.Button("Compare")
                with gr.Row():
                    out_t5 = gr.Textbox(lines=6, label="MADLAD (T5)")
                    out_nllb = gr.Textbox(lines=6, label="NLLB")
                out_ng_c = gr.Textbox(lines=4, label="NLLB Transliteration (NGA)")
                def _defaults_c(t):
                    n, r, l = get_defaults_for_target(t)
                    return n, r, l
                target_c.change(fn=_defaults_c, inputs=[target_c], outputs=[no_repeat_c, rep_pen_c, len_pen_c])
                btn_c.click(
                    fn=translate_compare,
                    inputs=[inp_c, target_c, beam_c, no_repeat_c, rep_pen_c, len_pen_c, show_uyg_c],
                    outputs=[out_t5, out_nllb, out_ng_c],
                )
        return demo


if __name__ == "__main__":
    app = build_ui()
    
    # Optional auth via env var GRADIO_AUTH="user:pass"
    launch_kwargs = {
        "server_name": "0.0.0.0", 
        "server_port": 7860
    }
    
    import os
    auth_env = os.environ.get("GRADIO_AUTH")
    if auth_env and ":" in auth_env:
        try:
            user, password = auth_env.split(":", 1)
            launch_kwargs["auth"] = (user, password)
        except Exception:
            pass  # Ignore malformed auth
    
    app.launch(**launch_kwargs)

