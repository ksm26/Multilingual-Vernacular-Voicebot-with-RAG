"""
RAG orchestration (custom): Indic LID -> (optional) translation -> retrieval -> deterministic generation -> re-translation -> optional TTS

Design choices:
- Uses your RetrieverWithIndicLID as retrieval backend (FAISS).
- Uses AI4Bharat IndicTrans2 models for Indic<->English translation (trust_remote_code=True).
- Uses a deterministic seq2seq generator (Flan-T5 by default) with beam search (no sampling).
- Provides debug logging of contexts & prompt; retries generation when output is too short.
"""
import os, re 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval_with_indiclid import RetrieverWithIndicLID, detect_lang_from_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ai4bharat_tts import IndicTTS

class IndicTranslator:
    def __init__(self,
                 indic2en_model: str = "ai4bharat/indictrans2-indic-en-1B",
                 en2indic_model: str = "ai4bharat/indictrans2-en-indic-1B",
                 device: Optional[str] = None,
                 local_files_only: bool = False,
                 trust_remote_code: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.indic2en_model_id = indic2en_model
        self.en2indic_model_id = en2indic_model
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        self.indic2en_tok = None; self.indic2en = None
        self.en2indic_tok = None; self.en2indic = None

        # Try to load IndicProcessor (required for IndicTrans2 preprocessing/postprocessing).
        self.ip = None
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except Exception:
            try:
                # alternative import path used in some installs
                from IndicTransToolkit import IndicProcessor
            except Exception:
                IndicProcessor = None

        if IndicProcessor:
            try:
                self.ip = IndicProcessor(inference=True)
                logger.info("[IndicTranslator] IndicProcessor loaded")
            except Exception as e:
                logger.warning(f"[IndicTranslator] Failed to init IndicProcessor: {e}")
                self.ip = None
        else:
            logger.info("[IndicTranslator] IndicProcessor not available; will fallback to plain tokenizer")

        # Minimal mapping from iso codes to FLORES-style language+script tags (extend as needed)
        self._flores_map = {
            "hi": "hin_Deva", "hin": "hin_Deva",
            "bn": "ben_Beng", "ben": "ben_Beng",
            "pa": "pan_Guru", "pan": "pan_Guru",
            "gu": "guj_Gujr", "guj": "guj_Gujr",
            "ta": "tam_Taml", "tam": "tam_Taml",
            "te": "tel_Telu", "tel": "tel_Telu",
            "kn": "kan_Knda", "kan": "kan_Knda",
            "ml": "mal_Mlym", "mal": "mal_Mlym",
            "or": "ory_Orya", "ori": "ory_Orya",
            "as": "asm_Beng", "asm": "asm_Beng",
            "ur": "urd_Arab", "urd": "urd_Arab",
            "sd": "snd_Arab", "snd": "snd_Arab",
            "mr": "mar_Deva", "mar": "mar_Deva",
            "ne": "npi_Deva", "npi": "npi_Deva",
            "en": "eng_Latn", "eng": "eng_Latn",
        }

    def _load_indic2en(self):
        if self.indic2en is None:
            self.indic2en_tok = AutoTokenizer.from_pretrained(
                self.indic2en_model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            )
            self.indic2en = AutoModelForSeq2SeqLM.from_pretrained(
                self.indic2en_model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            ).to(self.device).eval()

    def _load_en2indic(self):
        if self.en2indic is None:
            self.en2indic_tok = AutoTokenizer.from_pretrained(
                self.en2indic_model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            )
            self.en2indic = AutoModelForSeq2SeqLM.from_pretrained(
                self.en2indic_model_id,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            ).to(self.device).eval()

    def _sanitize_lang_tag(self, lang_tag: Optional[str]) -> Optional[str]:
        # accept 2- or 3-letter codes (e.g., 'hi', 'hin', 'en', 'eng')
        if not lang_tag or not isinstance(lang_tag, str):
            return None
        lt = lang_tag.strip().lower()
        return lt if (len(lt) in (2, 3) and lt.isalpha()) else None

    def _to_flores(self, lang_tag: Optional[str]) -> Optional[str]:
        """
        Convert a 2/3-letter language tag (or already FLORES tag) to the FLORES-style tag.
        Returns None if unknown.
        """
        if not lang_tag:
            return None
        lt = lang_tag.strip()
        # If it already looks like FLORES (has underscore), return as-is
        if "_" in lt:
            return lt
        # normalize
        key = lt.lower()
        return self._flores_map.get(key)

    def indic_to_en(self, text: str, src_lang: Optional[str] = None, max_length: int = 512) -> str:
        """
        Translate indic -> english. If IndicProcessor is available we use it (recommended).
        src_lang may be 'hi'/'hin'/'hin_Deva' etc.
        """
        self._load_indic2en()
        txt = text.strip()
        # try mapping to FLORES code
        src_flores = self._to_flores(self._sanitize_lang_tag(src_lang)) if src_lang else None

        # If IndicProcessor present and we have a FLORES src_lang -> use preprocess/postprocess
        if self.ip and src_flores:
            try:
                batch = self.ip.preprocess_batch([txt], src_lang=src_flores, tgt_lang="eng_Latn")
                inputs = self.indic2en_tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(self.device)
                with torch.no_grad():
                    outs = self.indic2en.generate(**inputs, max_new_tokens=max_length, num_beams=4)
                decoded = self.indic2en_tok.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                post = self.ip.postprocess_batch(decoded, lang="eng_Latn")
                return post[0].strip()
            except Exception as e:
                logger.warning(f"[IndicTranslator] IndicProcessor path failed: {e} — falling back to plain tokenize")

        # Fallback: plain tokenizer (what you had before)
        inputs = self.indic2en_tok([txt], src_lang=src_flores or "hin_Deva",tgt_lang="eng_Latn", return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outs = self.indic2en.generate(**inputs, max_new_tokens=max_length, num_beams=4)
        return self.indic2en_tok.batch_decode(outs, skip_special_tokens=True)[0].strip()

    def en_to_indic(self, text: str, tgt_lang: Optional[str] = None, max_length: int = 512) -> str:
        """
        Translate english -> indic. tgt_lang may be 'hi'/'hin'/'hin_Deva' etc.
        """
        self._load_en2indic()
        txt = text.strip()
        tgt_flores = self._to_flores(self._sanitize_lang_tag(tgt_lang)) if tgt_lang else None

        if self.ip and tgt_flores:
            try:
                batch = self.ip.preprocess_batch([txt], src_lang="eng_Latn", tgt_lang=tgt_flores)
                inputs = self.en2indic_tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(self.device)
                with torch.no_grad():
                    outs = self.en2indic.generate(**inputs, max_new_tokens=max_length, num_beams=4)
                decoded = self.en2indic_tok.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                post = self.ip.postprocess_batch(decoded, lang=tgt_flores)
                return post[0].strip()
            except Exception as e:
                logger.warning(f"[IndicTranslator] IndicProcessor (EN->Indic) failed: {e} — falling back to plain tokenize")

        inputs = self.en2indic_tok([txt], src_lang="eng_Latn",tgt_lang=tgt_flores or "hin_Deva", return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outs = self.en2indic.generate(**inputs, max_new_tokens=max_length, num_beams=4)
        return self.en2indic_tok.batch_decode(outs, skip_special_tokens=True)[0].strip()

# -------------------------
# Deterministic generator wrapper
# -------------------------
class Generator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only

        self._tok = None
        self._model = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        logger.info(f"[Generator] Loading generator model: {self.model_name} (local_only={self.local_files_only})")
        self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code, local_files_only=self.local_files_only)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code, local_files_only=self.local_files_only).to(self.device).eval()
        self._loaded = True

    def generate(self, prompt: str, max_length: int = 256, num_beams: int = 4, temperature: float = 0.5) -> str:
        self._lazy_load()
        if not prompt or not prompt.strip():
            return ""
        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        # Ensure pad/eos ids exist
        pad_id = getattr(self._tok, "pad_token_id", None) or getattr(self._model.config, "pad_token_id", None)
        eos_id = getattr(self._tok, "eos_token_id", None) or getattr(self._model.config, "eos_token_id", None)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=int(num_beams),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
            )

        text = self._tok.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()


# -------------------------
# RAG orchestration class
# -------------------------
class LangchainRAG:
    def __init__(
        self,
        csv_path: str,
        config_path: str = "config.yaml",
        indiclid_ft_model: Optional[str] = None,
        generator_model: str = "google/flan-t5-large",
        translator_models: Tuple[str, str] = ("ai4bharat/indictrans2-indic-en-1B", "ai4bharat/indictrans2-en-indic-1B"),
        device: Optional[str] = None,
        top_k: int = 4,
        translator_local_only: bool = True,
        generator_local_only: bool = False,
        memory_size: int = 20,
    ):
        """
        csv_path/config_path: passed to RetrieverWithIndicLID.
        translator_models: (indic2en, en2indic)
        translator_local_only: if True load translator from local cache only (no network)
        generator_local_only: if True load generator locally only
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.memory_size = memory_size
        # simple in-memory session memory: session_id -> deque(maxlen=memory_size)
        self.session_memory: Dict[str, deque] = {}

        logger.info("[RAG] Initializing RetrieverWithIndicLID (may build index)...")
        self.retriever_core = RetrieverWithIndicLID(csv_path=csv_path, config_path=config_path, indiclid_ft_model=indiclid_ft_model)

        # translator (lazy)
        self.translator = IndicTranslator(
            indic2en_model=translator_models[0],
            en2indic_model=translator_models[1],
            device=self.device,
            local_files_only=translator_local_only,
            trust_remote_code=True,
        )

        # deterministic generator (lazy)
        self.generator = Generator(model_name=generator_model, device=self.device, trust_remote_code=False, local_files_only=generator_local_only)
        # TTS model 
        self.tts_engine = IndicTTS()

    # helpers
    @staticmethod
    def _build_prompt(question_en: str, contexts: List[Dict[str, Any]], answer_language: str = "en") -> str:

        # SYSTEM_PROMPT = (
        #     "You are a helpful assistant. Answer the user’s question using only the information from the provided contexts. "
        #     "Do not copy the contexts verbatim or list them with numbers. "
        #     "Instead, combine the relevant points into a short, fluent answer in natural language. "
        #     "If none of the contexts are relevant, say: \"I don't have information on this topic.\" "
        #     "Give a concise factual answer in 1–2 sentences. "
        #     "Always answer in English."  # 👈 force generator to stay in English
        # )
        
        SYSTEM_PROMPT = (
            "You are a helpful assistant. Answer the user's question using the provided contexts. "
            "You may list relevant schemes mentioned in the contexts. Only attribute an action (e.g. 'Prime Minister launched X') "
            "if a context explicitly states that. If no explicit attribution exists, you may say the schemes are 'government initiatives' "
            "if that is a reasonable general inference, but do not invent named personal attributions. Give a concise factual answer."
        )

        context_text = "\n\n---\n".join(
            [
                f"[{i+1}] (id={c.get('id', 'NA')}, lang={c.get('lang', '')}) {c.get('text','')}"
                for i, c in enumerate(contexts)
                if c.get("text")
            ]
        )

        user_prompt = (
            SYSTEM_PROMPT
            + "\n\nCONTEXTS:\n"
            + context_text
            + "\n\nUser query (in English): "
            + question_en
            + "\n\nAnswer in English:"
        )
        return user_prompt
    
    # ---------- conversational memory helpers ----------
    def _get_session_deque(self, session_id: Optional[str]) -> deque:
        sid = session_id or "default_session"
        if sid not in self.session_memory:
            self.session_memory[sid] = deque(maxlen=self.memory_size)
        return self.session_memory[sid]
    
    def _append_memory(self, session_id: Optional[str], user_text: str, user_en: str, assistant_local: str, assistant_en: str, source_ids: Optional[list] = None):
        """
        Store a finished turn in memory. Each memory entry contains original + English forms.
        """
        dd = {
            "user": user_text,
            "user_en": user_en,
            "assistant": assistant_local,
            "assistant_en": assistant_en,
            "source_ids": source_ids or []
        }
        dq = self._get_session_deque(session_id)
        dq.append(dd)

    def _get_last_turns(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        dq = self._get_session_deque(session_id)
        return list(dq)  # oldest -> newest

    def _is_likely_followup(self, query: str, query_en: str, last_turns: List[Dict[str, Any]]) -> bool:
        """
        Heuristic: short queries or queries with pronouns / interrogative particles likely follow-ups.
        """
        q_tokens = query.strip().split()
        if len(q_tokens) <= 6:
            return True
        # presence of Hindi follow-up tokens or English pronouns that commonly indicate ellipsis
        followup_indicators = ["isme", "usme", "isme", "isme?", "kya", "kya?", "is", "this", "that", "it", "there", "woh", "waha", "kaun"]
        qlow = query.lower()
        for tok in followup_indicators:
            if tok in qlow:
                return True
        return False

    def _rewrite_followup(self, query_en: str, last_turns: List[Dict[str, Any]]) -> str:
        """
        Use the generator to rewrite a follow-up query into a standalone English question.
        If the rewrite fails or is empty, returns original query_en.
        """
        # Build a small conversation context (keep last 3 turns)
        conv_lines = []
        for i, t in enumerate(last_turns[-self.memory_size:]):
            u = t.get("user_en", t.get("user", ""))
            a = t.get("assistant_en", t.get("assistant", ""))
            conv_lines.append(f"User: {u}")
            conv_lines.append(f"Assistant: {a}")
        conv_block = "\n".join(conv_lines)

        rewrite_prompt = (
            "You are a utility that rewrites a follow-up question into a standalone English question. "
            "Do not add facts. Use only the conversation below to resolve pronouns/ellipses. "
            "If context does not clarify, keep the question simple.\n\n"
            f"Conversation:\n{conv_block}\n\n"
            f"Follow-up question: {query_en}\n\n"
            "Rewrite into a single, self-contained English question:"
        )
        try:
            rewritten = self.generator.generate(rewrite_prompt, max_length=128, temperature=0.7)
            rewritten = rewritten.strip()
            # defensive: if generator echoes the input or returns empty, keep original
            if not rewritten or len(rewritten.split()) < 3:
                return query_en
            return rewritten
        except Exception as e:
            logger.warning(f"[RAG] Follow-up rewrite failed: {e}")
            return query_en
#----------------------------------------------------------------------------
    def clean_generation(self, text: str) -> str:
        return re.sub(r"\s*---?\s*\[\d+\]\s*\(id=\d+,\s*lang=\w+\)", "", text).strip()

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        topk: Optional[int] = None,
        translate_if_needed: bool = True,
        min_confidence: float = 0.30,
        return_tts: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute RAG: language detection -> (optional) translate -> retrieve -> translate contexts -> generate -> re-translate -> optionally TTS.
        Returns dict with keys: query, query_lang, query_en, translated (bool), contexts, generated_en, generated_local, confidence, tts_path
        """
        topk = topk or self.top_k
        out: Dict[str, Any] = {"query": query}

        # 1) detect language (use detection helper from retrieval module)
        try:
            qlang = detect_lang_from_text(query)
        except Exception:
            qlang = "en"
        out["query_lang"] = qlang

        # 2) translate to English if needed
        query_en = query
        translated_flag = False
        if qlang != "en" and translate_if_needed:
            try:
                # short = qlang if len(qlang) == 2 else None
                # query_en = self.translator.indic_to_en(query, src_lang=short)
                query_en = self.translator.indic_to_en(query, src_lang=qlang)
                translated_flag = True
            except Exception as e:
                logger.warning(f"[RAG] Translation Indic->EN failed: {e}")
                query_en = query
        out["query_en"] = query_en
        out["translated"] = translated_flag
        # --- conversational context & follow-up handling ---
        out["session_id"] = session_id or "default_session"
        last_turns = self._get_last_turns(session_id)
        out["session_memory_len"] = len(last_turns)

        # if this is likely a follow-up, attempt to rewrite to a standalone English question
        if self._is_likely_followup(query, query_en, last_turns):
            try:
                rew = self._rewrite_followup(query_en, last_turns)
                if rew and rew != query_en:
                    logger.info(f"[RAG] Rewrote follow-up: '{query_en}' -> '{rew}'")
                    query_en = rew
                    out["rewritten_query_en"] = rew
            except Exception as e:
                logger.warning(f"[RAG] Follow-up rewrite exception: {e}")

        # 3) retrieve (prefer vernacular query so retriever matches same-language docs)
        raw_ctxs: List[Dict[str, Any]] = []
        rr = None
        retriever_confident = False
        try:
            # Prefer query_with_confidence if the retriever exposes it
            if hasattr(self.retriever_core, "query_with_confidence"):
                rr = self.retriever_core.query_with_confidence(query, topk=topk)
                # retriever returns a dict with 'results' and 'confident' usually
                if isinstance(rr, dict):
                    raw_ctxs = rr.get("results", []) or []
                    retriever_confident = bool(rr.get("confident", False))
                    # If retriever said 'not confident', try English translation as a fallback
                    if not retriever_confident and query_en and query_en != query:
                        rr2 = self.retriever_core.query_with_confidence(query_en, topk=topk)
                        if isinstance(rr2, dict) and rr2.get("results"):
                            raw_ctxs = rr2.get("results")
                            retriever_confident = bool(rr2.get("confident", False))
                else:
                    # older/simple retrievers might return a list directly
                    raw_ctxs = rr or []
                    retriever_confident = True if raw_ctxs else False
            elif hasattr(self.retriever_core, "query"):
                # backward compatibility: some retrievers provide `.query()`
                raw_ctxs = self.retriever_core.query(query, topk=topk) or []
                retriever_confident = True if raw_ctxs else False
            else:
                logger.warning("[RAG] Retriever has neither query_with_confidence() nor query()")
                raw_ctxs = []
        except Exception as e:
            logger.warning(f"[RAG] Retrieval error: {e}")
            raw_ctxs = []


        logger.info(f"[RAG] Retrieved {len(raw_ctxs)} contexts (topk={topk}), retriever_confident={retriever_confident}")
        if isinstance(rr, dict):
            logger.info(f"[RAG] Retriever top_scores: {rr.get('top_scores')}, message: {rr.get('message')}")
        for i, c in enumerate(raw_ctxs[: min(5, len(raw_ctxs)) ]):
            logger.info(f"[RAG] Candidate {i+1}: id={c.get('id')} lang={c.get('lang')} score={c.get('score')}\n{(c.get('text') or '')[:300]}")
        
        # 4) translate contexts -> English if needed (generator reasons in English)
        contexts_en: List[Dict[str, Any]] = []
        for c in raw_ctxs:
            ctext = c.get("text", "") or ""
            clang = (c.get("lang") or "").lower()
            is_english = clang in ("en", "eng", "english") or clang.startswith("en")
            if is_english:
                c_en = ctext
            else:
                try:
                    short = clang[:2] if clang and len(clang) >= 2 else None
                    c_en = self.translator.indic_to_en(ctext, src_lang=short)
                except Exception:
                    c_en = ctext
            contexts_en.append({"id": c.get("id"), "lang": clang, "text": c_en, "url": c.get("url"), "score": c.get("score")})

        # 5) build prompt & generate
        prompt = self._build_prompt(question_en=query_en, contexts=contexts_en[:topk])
        logger.info(f"[RAG] Prompt length: {len(prompt)} chars")

        gen_en = ""
        # BEFORE generate
        logger.info("[RAG] FINAL PROMPT (first 4k chars):\n%s", prompt[:4000])
        try:
            gen_en = self.generator.generate(prompt, max_length=256, temperature=0.7)
        except Exception as e:
            logger.warning(f"[RAG] Generation failed (first pass): {e}")
            gen_en = ""

        # retry if too short
        if not gen_en or len(gen_en.strip()) < 8:
            logger.info("[RAG] Generation short or empty — retrying with stronger decode")
            try:
                gen_en = self.generator.generate(prompt, max_length=512, temperature=0.7)
            except Exception as e:
                logger.warning(f"[RAG] Generation retry failed: {e}")
                gen_en = ""
        
        gen_en = self.clean_generation(gen_en)

        # AFTER generate
        logger.info("[RAG] Raw model output (repr): %r", gen_en)
        #------------------------------

        # 6) compute confidence (simple mean of retriever scores)
        scores = [float(c.get("score", 0.0) or 0.0) for c in raw_ctxs] if raw_ctxs else []
        top_score = float(max(scores)) if scores else 0.0
        out["raw_scores"] = scores
        # prefer the retriever's own decision when available
        retriever_decision = False
        if isinstance(rr, dict) and "confident" in rr:
            retriever_decision = bool(rr.get("confident"))
        else:
            retriever_decision = top_score >= min_confidence

        out["generated_en"] = gen_en.strip() if gen_en else ""
        # final confidence score exposed to caller (for logging/metrics)
        out["confidence"] = top_score

        # 7) fallback if confidence too low or generation failed
        low_confidence = not retriever_decision
        if low_confidence or not gen_en or len(gen_en.strip()) < 8:
            logger.info(f"[RAG] Low confidence or empty generation (top_score={top_score:.3f}, retriever_decision={retriever_decision}) -> fallback")
            fallback_local = "Mujhe is vishay mein jaankari nahi hai." if str(out.get("query_lang", "")).startswith(("hi", "pa", "bn", "gu", "ta", "te")) else "I don't have information on this topic."
            out.update({"generated_en": gen_en, "generated_local": fallback_local, "contexts": raw_ctxs, "retrieved_texts": [c.get("text", "") for c in raw_ctxs],"tts_path": None})
            try:
                source_ids = [c.get("id") for c in raw_ctxs] if raw_ctxs else []
                self._append_memory(session_id, query, query_en, fallback_local, gen_en or "", source_ids=source_ids)
                session_deque = self._get_session_deque(session_id)
                out["memory_after"] = len(self._get_session_deque(session_id))
            except Exception as e:
                logger.warning(f"[RAG] Failed to append memory (fallback): {e}")

            # 👇 Print/log full chat history
            logger.info(f"[RAG] Full conversation so far (session={session_id}):")
            for turn_idx, turn in enumerate(session_deque, 1):
                logger.info(f"  [{turn_idx}] User: {turn['user']} | Assistant: {turn['assistant']}")

            tts_path = None
            if return_tts:
                txt_for_tts = gen_local if qlang != "en" else gen_en
                try:
                    tts_path = self.tts_engine.text_to_speech(txt_for_tts)  # auto-saves in tts_outputs/<lang>/
                except Exception as e:
                    logger.warning(f"[RAG] TTS generation failed: {e}")
                    tts_path = None

            out["tts_path"] = tts_path

            return out

        # 8) translate back to local language if needed
        gen_local = gen_en
        if qlang != "en":
            try:
                tgt = qlang if len(qlang) == 2 else None
                gen_local = self.translator.en_to_indic(gen_en, tgt_lang=tgt)
            except Exception:
                gen_local = gen_en

        out.update({"generated_en": gen_en, "generated_local": gen_local, "contexts": raw_ctxs, "retrieved_texts": [c.get("text", "") for c in raw_ctxs]})
        try:
            source_ids = [c.get("id") for c in raw_ctxs] if raw_ctxs else []
            self._append_memory(session_id, query, query_en, gen_local, gen_en, source_ids=source_ids)
            session_deque = self._get_session_deque(session_id)
            out["memory_after"] = len(self._get_session_deque(session_id))

        except Exception as e:
            logger.warning(f"[RAG] Failed to append memory (success): {e}")

        # 👇 Print/log full chat history
        logger.info(f"[RAG] Full conversation so far (session={session_id}):")
        for turn_idx, turn in enumerate(session_deque, 1):
            logger.info(f"  [{turn_idx}] User: {turn['user']} | Assistant: {turn['assistant']}")

        # 9) optional TTS
        tts_path = None
        if return_tts:
            txt_for_tts = gen_local if qlang != "en" else gen_en
            try:
                tts_path = self.tts_engine.text_to_speech(txt_for_tts)  # auto-saves in tts_outputs/<lang>/
            except Exception as e:
                logger.warning(f"[RAG] TTS generation failed: {e}")
                tts_path = None

        out["tts_path"] = tts_path

        return out


# -------------------------
# Quick CLI demo
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default='data/schemes_multilingual.csv')
    parser.add_argument("--config", default="config.yaml", help="config yaml with language columns")
    parser.add_argument("--indiclid_ft_model", default=None, help="optional IndicLID fasttext .bin path")
    parser.add_argument("--generator_model", default="google/flan-t5-large", help="generator model id or local path")
    parser.add_argument("--translator_indic2en", default="ai4bharat/indictrans2-indic-en-1B")
    parser.add_argument("--translator_en2indic", default="ai4bharat/indictrans2-en-indic-1B")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--no_translator_local", dest="translator_local", action="store_false", help="do not force translator to local files only")
    parser.add_argument("--no_generator_local", dest="generator_local", action="store_false", help="do not force generator to local files only")
    parser.set_defaults(translator_local=True, generator_local=False)
    args = parser.parse_args()


    # t = IndicTranslator(local_files_only=False)   # or True if cached locally
    # print(t.indic_to_en("मेरा नाम अजय है।", src_lang="hi"))
    # expected: English translation of the sentence

    rag = LangchainRAG(
        csv_path=args.csv,
        config_path=args.config,
        indiclid_ft_model=args.indiclid_ft_model,
        generator_model=args.generator_model,
        translator_models=(args.translator_indic2en, args.translator_en2indic),
        top_k=args.topk,
        translator_local_only=args.translator_local,
        generator_local_only=args.generator_local,
    )

    print("Enter query (any language; type 'exit' to quit):")
    while True:
        q = input("Query> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        # res = rag.run(q, topk=args.topk, return_tts=False)
        res = rag.run(q, session_id="default_user", topk=args.topk, return_tts=True)

        # print("Query lang:", res.get("query_lang"))
        # print("Generated (EN):", res.get("generated_en"))
        # print("Generated (Local):", res.get("generated_local"))
        # print("Confidence:", res.get("confidence"))
        # print("Contexts returned (id, score, lang):", [(c.get("id"), c.get("score"), c.get("lang")) for c in res.get("contexts", [])])
        # print("Retrieved texts:")
        # for i, txt in enumerate(res.get("retrieved_texts", []), 1):
        #     print(f"  {i}. {txt}")

        if res.get("tts_path"):
            print("TTS saved to:", res.get("tts_path"))
