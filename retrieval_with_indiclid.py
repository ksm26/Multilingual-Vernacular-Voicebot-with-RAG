#!/usr/bin/env python3
"""
retrieval_with_indiclid.py

- Uses AI4Bharat IndicLID (fasttext) to detect native vs romanized Indic text.
- Uses ai4bharat-transliteration to transliterate romanized queries into native script
  (word-by-word) before embedding.
- Builds FAISS index from CSV multilingual KB (english/hindi/punjabi columns).
- Applies confidence/OOD fallback like earlier.

Usage:
  python retrieval_with_indiclid.py \
    --csv data/schemes_multilingual.csv \
    --config config.yaml \
    --indiclid_ft_model ./models/IndicLID-FTR-v1.bin \
    --topk 3

If --indiclid_ft_model is not provided or fasttext not installed, falls back to
unicode-script detection (less accurate for romanized inputs).
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import re
import numpy as np
import pandas as pd
import faiss
import yaml
import torch
from transformers import AutoTokenizer, AutoModel
from urllib.parse import urlparse
# Cache for a loaded fasttext IndicLID model (if available)
_FASTTEXT_LID = None

# ---------- Embedding model (simple IndicBERT pooled) ----------
class IndicEmbeddingModel:
    def __init__(self, model_name="ai4bharat/indic-bert", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def encode(self, text: str):
        toks = self.tokenizer(
            text, truncation=True, padding="longest",
            return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**toks, return_dict=True)
            last_hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (summed / counts).squeeze().cpu().numpy()
        vec = mean_pooled.astype("float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

# ---------- FAISS wrapper ----------
class VectorStoreFAISS:
    def __init__(self, dim: int = None):
        if dim is not None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
        self.docs = []

    def build(self, embeddings: np.ndarray, metadatas: list):
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        self.docs = list(metadatas)

    def search_raw(self, query_vec: np.ndarray, topk=3):
        D, I = self.index.search(query_vec.reshape(1, -1), k=topk)
        return D, I

    def search(self, query_vec: np.ndarray, topk=3):
        D, I = self.search_raw(query_vec, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            md = self.docs[idx]
            results.append({"score": float(score), **md})
        return results

# ---------- IndicLID wrapper (fasttext) ----------
class IndicLID:
    """
    Wrapper for AI4Bharat IndicLID fasttext model.
    Expects a fasttext .bin classifier that uses labels like "__label__hin_Latn".
    """
    def __init__(self, ft_model_path: str = None):
        self.ft_model_path = ft_model_path
        self.ft = None
        self.available = False
        if ft_model_path:
            try:
                import fasttext
                self.fasttext = fasttext
                self.ft = fasttext.load_model(ft_model_path)
                self.available = True
            except Exception as e:
                print("[! IndicLID] fasttext not available or failed loading model:", e)
                self.available = False

    def predict(self, text: str, k: int = 1):
        """
        Returns (label, prob) where label is e.g. 'hin_Latn' or 'hin_Deva' or 'eng_Latn'
        If fasttext not available, returns (None, 0.0)
        """
        if not self.available:
            return None, 0.0
        labels, probs = self.ft.predict(text, k=k)
        # fasttext returns labels like "__label__hin_Latn"
        label = labels[0].replace("__label__", "") if labels else None
        prob = float(probs[0]) if probs else 0.0
        return label, prob

# ---------- Transliteration wrapper (AI4Bharat) ----------
class Transliterator:
    """
    Uses ai4bharat-transliteration XlitEngine. The package expects word-level inputs;
    so we tokenise and transliterate word-by-word and rejoin.
    """
    def __init__(self):
        self.engine_cls = None
        self.engines = {}
        try:
            from ai4bharat.transliteration import XlitEngine
            self.engine_cls = XlitEngine
        except Exception as e:
            self.engine_cls = None
            # missing package — transliteration won't be available
            print("[! Transliterator] ai4bharat-transliteration not installed:", e)

    @staticmethod
    def _lang_code_map(indic_base):
        # map IndicLID base codes to XlitEngine language codes (common aliases)
        # IndicLID uses codes like 'hin' (Hindi), 'pan' (Punjabi), etc.
        mapping = {
            "hin": "hi",
            "pan": "pa", "pun": "pa",  # maps 'pan' to 'pa' used by XlitEngine
            "eng": "en",
            # add more if needed
        }
        return mapping.get(indic_base, indic_base)

    def transliterate_sentence(self, text: str, indic_base_lang: str):
        """
        Transliterate romanized sentence `text` into native script for indic_base_lang.
        If transliterator not available, returns original text.
        """
        if not self.engine_cls:
            return text

        lang = self._lang_code_map(indic_base_lang)
        if lang not in self.engines:
            try:
                # default src_script_type is roman -> convert from roman to target
                self.engines[lang] = self.engine_cls(lang2use=lang, beam_width=8, rescore=True)
            except TypeError:
                # fallback to older constructor style
                self.engines[lang] = self.engine_cls(lang, beam_width=8, rescore=True)

        engine = self.engines[lang]

        # simple tokenization (words + punctuation)
        tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
        out_tokens = []
        for tok in tokens:
            if re.match(r"^\W+$", tok):
                out_tokens.append(tok)
            else:
                try:
                    # translit_word returns dict {lang: [cands...]}
                    res = engine.translit_word(tok, topk=1, beam_width=8)
                    # pick the first candidate (if available)
                    if isinstance(res, dict):
                        # res keys may be language code like 'hi' or full names; prefer mapped lang
                        if lang in res and res[lang]:
                            out_tokens.append(res[lang][0])
                        else:
                            # fallback: take the first available value
                            first_vals = next(iter(res.values()))
                            out_tokens.append(first_vals[0] if first_vals else tok)
                    elif isinstance(res, list):
                        out_tokens.append(res[0])
                    else:
                        out_tokens.append(tok)
                except Exception:
                    out_tokens.append(tok)
        return " ".join(out_tokens)

# ---------- Utility: detect script fallback ----------
def simple_script_detect(text: str):
    # quick heuristic: if Devanagari chars present => hi/Deva; if Gurmukhi present => pun/Guru; else Latin
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "Deva"
        if 0x0A00 <= o <= 0x0A7F:
            return "Guru"
    return "Latn"

def _load_fasttext_model_once():
    """Try to load a fasttext-based IndicLID model once. Path taken from env var or default location."""
    global _FASTTEXT_LID
    if _FASTTEXT_LID is not None:
        return _FASTTEXT_LID
    try:
        model_path = os.environ.get("INDICLID_MODEL_PATH", "./models/IndicLID-FTR-v1.bin")
        if os.path.exists(model_path):
            import fasttext
            _FASTTEXT_LID = fasttext.load_model(model_path)
        else:
            _FASTTEXT_LID = None
    except Exception:
        _FASTTEXT_LID = None
    return _FASTTEXT_LID

def detect_lang_from_text(text: str) -> str:
    """
    Return a short language code for `text` (e.g. 'hi', 'pa', 'en', 'bn', 'ta', ...).

    Order of attempts:
      1) If an IndicLID fasttext model is present (path from INDICLID_MODEL_PATH or default),
         use it (cached).
      2) Use Unicode script heuristic (Devanagari -> hi, Gurmukhi -> pa).
      3) Try `langdetect` library if installed (best-effort).
      4) Fallback to 'en'.

    This helper is deliberately conservative and fast.
    """
    txt = (text or "").strip()
    if not txt:
        return "en"

    # 1) fasttext IndicLID if available
    ft = _load_fasttext_model_once()
    if ft is not None:
        try:
            labels, probs = ft.predict(txt, k=1)
            if labels:
                lab = labels[0].replace("__label__", "")
                base = lab.split("_")[0]
                mapping = {
                    "hin": "hi", "pan": "pa", "pun": "pa", "eng": "en",
                    "ben": "bn", "tam": "ta", "tel": "te", "guj": "gu",
                    "mal": "ml", "kan": "kn", "ori": "or", "mar": "mr"
                }
                return mapping.get(base, base)
        except Exception:
            pass

    # 2) Unicode script heuristic
    s = simple_script_detect(txt)
    if s == "Deva":
        return "hi"
    if s == "Guru":
        return "pa"

    # 3) langdetect fallback (if installed)
    try:
        from langdetect import detect as _ldetect
        try:
            det = _ldetect(txt)
            return det
        except Exception:
            pass
    except Exception:
        pass

    # 4) final fallback
    return "en"

# ---------- OOD keyword detector (same as before) ----------
class OutOfDomainDetector:
    def __init__(self, extra_keywords=None):
        base = [
            "kisan", "farmer", "farm", "crop", "farming", "loan", "subsid", "scheme", "yojana", "credit",
            "insurance", "irrigat", "soil", "paddy", "wheat", "dairy", "fish", "micro", "agri", "nrega", "pm",
            # Hindi
            "किसान", "फसल", "कर्ज", "ऋण", "योजना", "सब्सिड", "किसानी", "मिट्टी", "सिंचाई", "खाद",
            # Punjabi
            "ਕਿਸਾਨ", "ਫਸਲ", "ਕਰਜ਼", "ਕਰਜ਼ਾ", "ਯੋਜਨਾ", "ਸਬਸਿਡ", "ਮਿੱਟੀ", "ਸਿੰਚਾਈ", "ਖਾਦ"
        ]
        self.keywords = set(base + (extra_keywords or []))

    def contains_domain_keyword(self, text: str):
        t = text.lower()
        for kw in self.keywords:
            if kw in t:
                return True
        return False

# ---------- Retriever with IndicLID + transliteration + confidence ----------
class RetrieverWithIndicLID:
    def __init__(self, csv_path: str, config_path: str,
                 indiclid_ft_model: str = None,
                 model_name: str = "ai4bharat/indic-bert",
                 threshold: float = 0.70, boost_threshold: float = 0.60, gap_threshold: float = 0.05):
        # load config
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.lang_specs = cfg.get("languages", [{"col": "english"}, {"col": "hindi"}, {"col": "punjab"}])
        self.csv_path = csv_path

        # embedding model
        self.embedder = IndicEmbeddingModel(model_name=model_name)

        # LID + transliteration
        self.indiclid = IndicLID(ft_model_path=indiclid_ft_model) if indiclid_ft_model else IndicLID(None)
        self.translit = Transliterator()

        # OOD
        self.ood_detector = OutOfDomainDetector()

        # thresholds
        self.threshold = threshold
        self.boost_threshold = boost_threshold
        self.gap_threshold = gap_threshold

        # load corpus and build index
        self.corpus = self._load_corpus()
        self._build_index()

    def _url_ok(self, u: str):
        try:
            p = urlparse(str(u))
            return p.scheme in ("http", "https") and bool(p.netloc)
        except:
            return False

    def _load_corpus(self):
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        corpus = []
        for _, row in df.iterrows():
            sid = row.get("id") if "id" in row else row.get("scheme_id", None)
            src = row.get("source_url", "")
            for spec in self.lang_specs:
                col = spec["col"]
                lang_code = spec.get("code", col)   # use code if present
                if col in row and isinstance(row[col], str) and row[col].strip():
                    corpus.append({
                        "id": sid,
                        "lang": lang_code,
                        "text": row[col].strip(),
                        "url": src if self._url_ok(src) else ""
                    })
        return corpus

    def _build_index(self):
        if not self.corpus:
            raise ValueError("Empty corpus")
        embs = []
        for doc in self.corpus:
            v = self.embedder.encode(doc["text"])
            embs.append(v)
        embs = np.vstack(embs).astype("float32")
        dim = embs.shape[1]
        self.vs = VectorStoreFAISS(dim)
        self.vs.build(embs, self.corpus)
        print(f"[i] Built FAISS with {len(self.corpus)} vectors (dim={dim})")

    def _detected_lang_and_script(self, query: str):
        # Try IndicLID first
        label, prob = (None, 0.0)
        if self.indiclid and self.indiclid.available:
            label, prob = self.indiclid.predict(query)
        # If IndicLID gave a label -> parse it
        if label:
            if "_" in label:
                base, script = label.split("_", 1)
                return base, script, prob
            else:
                return label, "Latn", prob
        # fallback: script heuristic
        s = simple_script_detect(query)
        if s == "Deva":
            return "hin", "Deva", 1.0
        if s == "Guru":
            return "pan", "Guru", 1.0
        return "eng", "Latn", 1.0

    def query_with_confidence(self, query: str, topk: int = 3):
        base, script, lid_conf = self._detected_lang_and_script(query)
        # if romanized (Latn) predicted for an Indic language, transliterate
        to_embed_text = query
        if script.lower().startswith("latn") and base not in ("eng", "other"):
            # transliterate romanized query to native script
            to_embed_text = self.translit.transliterate_sentence(query, base)
            # if transliteration returns same as input or translit not available, fall back
            if not to_embed_text:
                to_embed_text = query

        # compute embedding & search
        qv = self.embedder.encode(to_embed_text)
        D, I = self.vs.search_raw(qv, topk)
        scores = list(D[0])
        ids = list(I[0])
        if not scores:
            return {"confident": False, "message": self._fallback_message(base), "results": []}

        top1 = float(scores[0])
        top2 = float(scores[1]) if len(scores) > 1 else 0.0
        gap = top1 - top2

        # domain keyword check (check both original and transliterated text)
        keyword_present = self.ood_detector.contains_domain_keyword(query) or self.ood_detector.contains_domain_keyword(to_embed_text)

        thr = self.threshold if not keyword_present else self.boost_threshold

        # confidence decision
        is_confident = False
        if top1 >= thr:
            if gap >= self.gap_threshold or keyword_present:
                is_confident = True
            else:
                is_confident = False
        else:
            if keyword_present and top1 >= self.boost_threshold:
                is_confident = True
            else:
                is_confident = False

        # gather results
        results = []
        for score, idx in zip(scores, ids):
            if idx == -1: continue
            md = self.vs.docs[idx]
            results.append({"score": float(score), **md})

        if not is_confident:
            return {"confident": False, "message": self._fallback_message(base), "results": results, "top_scores": scores}

        return {"confident": True, "message": None, "results": results, "top_scores": scores}

    def _fallback_message(self, base_lang):
        # pick vernacular fallback by base_lang
        if base_lang in ("hin","hin_deva","hin_latn","hi"):
            return "मुझे इस विषय में जानकारी नहीं है।"
        if base_lang in ("pan","pun","pa","pan_guru","pan_latn"):
            return "ਮੈਨੂੰ ਇਸ ਵਿਸ਼ੇ ਬਾਰੇ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।"
        return "I don't have information on this topic."

# ---------- CLI ----------
def interactive_mode(args):
    retriever = RetrieverWithIndicLID(
        csv_path=args.csv,
        config_path=args.config,
        indiclid_ft_model=args.indiclid_ft_model,
        model_name=args.embed_model,
        threshold=args.threshold,
        boost_threshold=args.boost_threshold,
        gap_threshold=args.gap_threshold
    )
    print("Enter queries (type 'exit' to quit).")
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        out = retriever.query_with_confidence(q, topk=args.topk)
        if not out["confident"]:
            print("\n" + out["message"])
            # show candidates for debugging
            print("\nTop results (candidates) — but returning OOD fallback:")
            for r in out["results"][:args.topk]:
                print(f"- score={r['score']:.4f} | id={r['id']} | lang={r['lang']}")
                print(f"  text: {r['text'][:120]}{'...' if len(r['text'])>120 else ''}")
                if r.get("url"): print(f"  source: {r['url']}")
        else:
            print("\nTop results (confident):")
            for r in out["results"][:args.topk]:
                print(f"- score={r['score']:.4f} | id={r['id']} | lang={r['lang']}")
                print(f"  text: {r['text'][:120]}{'...' if len(r['text'])>120 else ''}")
                if r.get("url"): print(f"  source: {r['url']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default='data/schemes_multilingual.csv')
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--indiclid_ft_model", default=None,
                   help="Path to IndicLID fasttext .bin model (optional). If omitted, falls back to script heuristics.")
    p.add_argument("--embed_model", default="ai4bharat/indic-bert")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--boost_threshold", type=float, default=0.60)
    p.add_argument("--gap_threshold", type=float, default=0.05)
    args = p.parse_args()
    interactive_mode(args)
