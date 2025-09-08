import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid tokenizers warnings after fork

import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC, pipeline
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# ASRModelWrapper: load processor+model lazily and provide scoring/transcription
# -----------------------------------------------------------------------------
class ASRModelWrapper:
    def __init__(self, model_id: str, device: str = None, use_pipeline_fallback: bool = True):
        """
        model_id: HF model string (e.g., "ai4bharat/indicwav2vec-hindi")
        device: "cuda" / "cpu" / None (auto)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pipeline_fallback = use_pipeline_fallback

        self.processor = None
        self.model = None
        self._pipeline = None  # HF pipeline fallback
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            # processor may fail for some model variants
            print(f"[ASRWrapper] Warning: failed to load processor for {self.model_id}: {e}")
            self.processor = None

        try:
            self.model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
        except Exception as e:
            print(f"[ASRWrapper] Warning: failed to load AutoModelForCTC for {self.model_id}: {e}")
            self.model = None

        self._loaded = True

    def _init_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            device_idx = 0 if ("cuda" in self.device and torch.cuda.is_available()) else -1
            if self.model is not None and self.processor is not None:
                # supply already loaded objects if possible
                # pipeline accepts model and feature_extractor/tokenizer in some versions
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    feature_extractor=self.processor.feature_extractor if hasattr(self.processor, "feature_extractor") else None,
                    tokenizer=self.processor.tokenizer if hasattr(self.processor, "tokenizer") else None,
                    device=device_idx
                )
            else:
                # let HF load model from id (slower)
                self._pipeline = pipeline("automatic-speech-recognition", model=self.model_id, device=device_idx)
        except Exception as e:
            print(f"[ASRWrapper] Pipeline init failed for {self.model_id}: {e}")
            self._pipeline = None

    @staticmethod
    def _resample_to(audio_np: np.ndarray, orig_sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        if orig_sr == target_sr:
            return audio_np.astype("float32"), orig_sr
        # try torchaudio resample
        try:
            wav_t = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, N)
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=orig_sr, new_freq=target_sr)
            return wav_t.squeeze(0).numpy().astype("float32"), target_sr
        except Exception:
            # fallback naive interpolation
            num = int(len(audio_np) * float(target_sr) / orig_sr)
            xp = np.linspace(0, len(audio_np), num=num, endpoint=False)
            res = np.interp(xp, np.arange(len(audio_np)), audio_np).astype("float32")
            return res, target_sr

    def score_audio(self, audio_np: np.ndarray, sr: int) -> float:
        """
        Return a confidence score for how well this model "explains" the audio.
        Method: compute logits (time x vocab), softmax -> take max prob per time frame, average.
        Higher is better (range 0..1). If model missing logits, return -inf.
        """
        self._lazy_load()
        if self.model is None or self.processor is None:
            # if no direct CTC model available, try pipeline fallback and return modest score
            if self.use_pipeline_fallback:
                try:
                    self._init_pipeline()
                    # pipeline doesn't expose a direct confidence easily; return small positive score
                    return 0.2
                except Exception:
                    return -1.0
            return -1.0

        # resample
        audio_res, sr2 = self._resample_to(audio_np, sr, target_sr=16000)
        # processor -> input_values
        try:
            inputs = self.processor(audio_res, sampling_rate=sr2, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_values)
            logits = getattr(outputs, "logits", None)
            if logits is None or logits.dim() != 3:
                return -1.0
            # compute softmax and take max prob per frame
            probs = torch.softmax(logits, dim=-1)  # (1, time, vocab)
            max_probs, _ = torch.max(probs, dim=-1)  # (1, time)
            score = float(max_probs.mean().cpu().numpy())
            # optional: penalize very short utterances
            return score
        except Exception as e:
            print(f"[ASRWrapper] score_audio failed for {self.model_id}: {e}")
            # fallback modest score if pipeline available
            if self.use_pipeline_fallback:
                return 0.2
            return -1.0

    def transcribe(self, audio_np: np.ndarray, sr: int) -> str:
        """
        Transcribe audio to text. Prefers direct model decode (CTC), falls back to pipeline.
        """
        self._lazy_load()
        # resample
        audio_res, sr2 = self._resample_to(audio_np, sr, target_sr=16000)
        # fast path: direct CTC if processor+model present
        try:
            if self.processor is not None and self.model is not None:
                inputs = self.processor(audio_res, sampling_rate=sr2, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_values)
                logits = getattr(outputs, "logits", None)
                if logits is not None and logits.dim() == 3:
                    pred_ids = torch.argmax(logits, dim=-1)
                    # decode with processor (batch)
                    transcription = self.processor.batch_decode(pred_ids)[0]
                    return " ".join(transcription.strip().split())
        except Exception as e:
            print(f"[ASRWrapper] direct transcribe failed for {self.model_id}: {e}")

        # pipeline fallback
        if self.use_pipeline_fallback:
            try:
                self._init_pipeline()
                if self._pipeline is not None:
                    res = self._pipeline(audio_res, chunk_length_s=30)
                    # pipeline may return dict or string
                    if isinstance(res, dict):
                        text = res.get("text", "")
                    elif isinstance(res, list):
                        text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                    else:
                        text = str(res)
                    return " ".join(text.strip().split())
            except Exception as e:
                print(f"[ASRWrapper] pipeline fallback failed for {self.model_id}: {e}")
        return ""

# -----------------------------------------------------------------------------
# ASRRouter: given candidate models, run lightweight scoring and pick best
# -----------------------------------------------------------------------------
class ASRRouter:
    def __init__(self, candidates: Dict[str, str], device: str = None, use_pipeline_fallback=True):
        """
        candidates: dict mapping language code -> HF model id
            e.g., {"hi": "ai4bharat/indicwav2vec-hindi", "pa": "ai4bharat/indicwav2vec-punjabi", "multi": "openai/whisper-small"}
        """
        self.candidates = candidates
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pipeline_fallback = use_pipeline_fallback
        # cache wrappers
        self._wrappers: Dict[str, ASRModelWrapper] = {}

    def _get_wrapper(self, model_id: str) -> ASRModelWrapper:
        if model_id in self._wrappers:
            return self._wrappers[model_id]
        w = ASRModelWrapper(model_id, device=self.device, use_pipeline_fallback=self.use_pipeline_fallback)
        self._wrappers[model_id] = w
        return w

    def route(self, audio_np: np.ndarray, sr: int, top_k: int = 1, min_score_threshold: float = 0.0):
        """
        Score audio against all candidate models and return sorted list of (lang_code, model_id, score).
        """
        scores = []
        for lang_code, model_id in self.candidates.items():
            w = self._get_wrapper(model_id)
            try:
                s = w.score_audio(audio_np, sr)
            except Exception as e:
                print(f"[ASRRouter] score error for {model_id}: {e}")
                s = -1.0
            scores.append((lang_code, model_id, s))
        # sort by score desc
        scores.sort(key=lambda x: x[2], reverse=True)
        # optionally filter by min threshold
        filtered = [t for t in scores if t[2] >= min_score_threshold]
        return filtered[:top_k] if filtered else scores[:top_k]

    def transcribe_with_best(self, audio_np: np.ndarray, sr: int, prefer_lang=None) -> Tuple[str, str, float]:
        """
        Get best model and transcribe with it.
        Returns: (transcript, best_model_id, score)
        """
        ranked = self.route(audio_np, sr, top_k=len(self.candidates))
        if not ranked:
            return "", None, -1.0
        best_lang, best_model, best_score = ranked[0]
        wrapper = self._get_wrapper(best_model)
        text = wrapper.transcribe(audio_np, sr)
        return text, best_model, best_score

# -----------------------------------------------------------------------------
# Example usage / integration snippet
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # simple test harness (requires models to be accessible & audio file)
    import soundfile as sf
    # define candidate models (update ids as per availability)
    CANDIDATES = {
        "hi": "ai4bharat/indicwav2vec-hindi",
        "pa": "manandey/wav2vec2-large-xlsr-punjabi",
        "en": "openai/whisper-small"  # optional multilingual fallback
    }

    router = ASRRouter(CANDIDATES)
    # load a test audio file (mono)
    audio_path = "tts_outputs/hi/tts_18b398d7c199455ebf2a6beddb704d75.wav"  # replace with your file
    audio_np, sr = sf.read(audio_path)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    print("[test] routing...")
    ranked = router.route(audio_np, sr)
    print("Ranked models (lang, id, score):", ranked)
    text, model_id, score = router.transcribe_with_best(audio_np, sr)
    print("Chosen model:", model_id, "score:", score)
    print("Transcript:", text)
