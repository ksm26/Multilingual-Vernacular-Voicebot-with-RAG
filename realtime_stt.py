#!/usr/bin/env python3
"""
realtime_stt_and_retrieve_no_retriever.py

Live recording -> ASR -> detect language -> pass transcript to LangchainRAG
This simplified file removes any dependency on RetrieverWithIndicLID and instead
only records audio, does ASR, detects language heuristically (or via langdetect
if available) and forwards the transcript to LangchainRAG.

Usage example (same CLI shape as before):
python realtime_stt_and_retrieve_no_retriever.py --csv data/schemes_multilingual.csv --config config.yaml --mode single --asr_model ai4bharat/indic-seamless --topk 3

Notes:
- langdetect (pip install langdetect) is optional but recommended for better
  language detection. If it's not available we fall back to a Unicode-script
  heuristic for Indic scripts and default to 'en'.
- This script deliberately does NOT import or construct RetrieverWithIndicLID.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import queue
import re
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from typing import Tuple
from langchain_rag import LangchainRAG
from transformers import AutoProcessor, AutoModelForCTC, pipeline, AutoModel, AutoConfig

# Optional ASRRouter (kept optional — nothing changed here)
try:
    from audio_asr_router import ASRRouter
    HAVE_ASR_ROUTER = True
except Exception:
    HAVE_ASR_ROUTER = False

# ----------------------
# MicRecorder (no file path) - returns numpy audio + samplerate
# ----------------------
class MicRecorder:
    def __init__(self, target_sr=16000, device=None, channels=1, dtype='float32', blocksize=1024):
        self.target_sr = int(target_sr)
        self.device = device
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.samplerate = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[sd status]", status, file=sys.stderr)
        self.q.put(indata.copy())

    def record_until_enter(self, prompt="Press ENTER to start recording, ENTER again to stop...\n") -> Tuple[np.ndarray, int]:
        input(prompt)
        print("Recording... Press ENTER to stop.")
        # try to open stream at target_sr; if fails, fallback to device default sr
        try:
            stream = sd.InputStream(device=self.device, channels=self.channels,
                                    samplerate=self.target_sr, dtype=self.dtype,
                                    blocksize=self.blocksize, callback=self._callback)
            stream.start()
            self.samplerate = self.target_sr
        except Exception as e:
            print("[!] Could not open stream at target sample rate", self.target_sr, ":", e)
            try:
                default = sd.query_devices(kind='input')
                fallback_sr = int(default['default_samplerate'])
            except Exception:
                fallback_sr = 48000
            print(f"[i] Falling back to device default samplerate: {fallback_sr}")
            stream = sd.InputStream(device=self.device, channels=self.channels,
                                    samplerate=fallback_sr, dtype=self.dtype,
                                    blocksize=self.blocksize, callback=self._callback)
            stream.start()
            self.samplerate = fallback_sr

        try:
            input()  # stop
        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt - stopping recording")
        finally:
            stream.stop()
            stream.close()

        # collect frames
        frames = []
        while not self.q.empty():
            frames.append(self.q.get())
        if not frames:
            return np.array([], dtype=self.dtype), self.samplerate or self.target_sr

        audio = np.concatenate(frames, axis=0)
        # stereo -> mono
        if audio.ndim > 1 and audio.shape[1] > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio.reshape(-1)
        audio_mono = audio_mono.astype(self.dtype)
        return audio_mono, int(self.samplerate)


# ----------------------
# Robust single-model ASR wrapper (CTC fast-path + pipeline fallback)
# ----------------------
class SingleASR:
    def __init__(self, model_name, device: str = None, use_pipeline_fallback: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_pipeline_fallback = use_pipeline_fallback

        # lazy loaded attributes
        self.processor = None
        self.model = None
        self._pipeline = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"[ASR] Warning: AutoProcessor.from_pretrained failed for {self.model_name}: {e}")
            self.processor = None

        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, config=config, trust_remote_code=True).to(self.device)
        except Exception as e:
            print(f"[ASR] Info: AutoModel.from_pretrained did not return a usable model object for {self.model_name}: {e}")
            self.model = None

        if self.model is None:
            try:
                self.model = AutoModelForCTC.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            except Exception:
                self.model = None

        self._init_pipeline()
        self._loaded = True

    def _init_pipeline(self):
        if self._pipeline is not None:
            return

        device_idx = 0 if ("cuda" in self.device and torch.cuda.is_available()) else -1

        try:
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device_idx,
                trust_remote_code=True,
            )
            return
        except Exception as e:
            print(f"[ASR] pipeline init failed (name+trust_remote_code): {e}")

        try:
            if self.model is not None and self.processor is not None:
                feature_extractor = getattr(self.processor, "feature_extractor", None)
                tokenizer = getattr(self.processor, "tokenizer", None)
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    device=device_idx,
                )
                return
        except Exception as e:
            print(f"[ASR] pipeline init failed (from objects): {e}")

        try:
            self._pipeline = pipeline("automatic-speech-recognition", model=self.model_name, device=device_idx)
            return
        except Exception as e:
            print(f"[ASR] pipeline final fallback failed: {e}")
            self._pipeline = None

    @staticmethod
    def _resample(audio_np: np.ndarray, orig_sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        if orig_sr == target_sr:
            return audio_np.astype("float32"), orig_sr
        try:
            wav_t = torch.from_numpy(audio_np).float().unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, orig_freq=orig_sr, new_freq=target_sr)
            return wav_t.squeeze(0).numpy().astype("float32"), target_sr
        except Exception:
            import numpy as _np
            num = int(len(audio_np) * float(target_sr) / orig_sr)
            xp = _np.linspace(0, len(audio_np), num=num, endpoint=False)
            res = _np.interp(xp, _np.arange(len(audio_np)), audio_np).astype("float32")
            return res, target_sr

    def transcribe_ndarray(self, audio_np: np.ndarray, sr: int) -> Tuple[str, float]:
        if audio_np is None or audio_np.size == 0:
            return "", 0.0
        self._lazy_load()
        audio_res, sr2 = self._resample(audio_np, sr, 16000)
        # fast path
        if self.processor is not None and self.model is not None:
            try:
                inputs = self.processor(audio_res, sampling_rate=sr2, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_values)
                logits = getattr(outputs, "logits", None)
                if logits is not None and logits.dim() == 3:
                    pred_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(pred_ids)[0]
                    probs = torch.softmax(logits, dim=-1)
                    max_per_frame, _ = torch.max(probs, dim=-1)
                    conf = float(max_per_frame.mean().cpu().numpy())
                    return " ".join(transcription.strip().split()), float(conf)
                else:
                    print("[ASR] Unexpected logits shape or missing logits -> fallback pipeline")
            except Exception as e:
                print("[ASR] Direct decode failed:", e)

        if self.use_pipeline_fallback:
            try:
                self._init_pipeline()
                if self._pipeline is not None:
                    res = self._pipeline(audio_res, chunk_length_s=30)
                    if isinstance(res, dict):
                        text = res.get("text", "")
                    elif isinstance(res, list):
                        text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                    else:
                        text = str(res)
                    return " ".join(text.strip().split()), 0.35
            except Exception as e:
                print("[ASR] Pipeline fallback failed:", e)

        return "", 0.0


# ----------------------
# Language detection helper (try langdetect, else Unicode-script heuristic)
# ----------------------

def detect_language(text: str) -> str:
    """Return a language code like 'en', 'hi', 'pa', 'bn', etc. Falls back to 'en' or 'und'."""
    text = (text or "").strip()
    if not text:
        return "und"

    try:
        # langdetect gives good general detection for many languages
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(text)
        return lang
    except Exception:
        # fallback to simple Unicode-script checks for Indic scripts
        checks = [
            ('hi', r'[\u0900-\u097F]'),   # Devanagari (Hindi, Marathi, Nepali)
            ('pa', r'[\u0A00-\u0A7F]'),   # Gurmukhi (Punjabi)
            ('bn', r'[\u0980-\u09FF]'),   # Bengali
            ('gu', r'[\u0A80-\u0AFF]'),   # Gujarati
            ('ta', r'[\u0B80-\u0BFF]'),   # Tamil
            ('te', r'[\u0C00-\u0C7F]'),   # Telugu
            ('kn', r'[\u0C80-\u0CFF]'),   # Kannada
            ('ml', r'[\u0D00-\u0D7F]'),   # Malayalam
            ('or', r'[\u0B00-\u0B7F]'),   # Oriya
            ('si', r'[\u0D80-\u0DFF]'),   # Sinhala
        ]
        for code, pattern in checks:
            if re.search(pattern, text):
                return code

        # final heuristics
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        if non_ascii > 0:
            return 'und'
        return 'en'


# ----------------------
# Main orchestrator (unified): choose single vs router
# ----------------------
class RealtimeSTT:
    def __init__(self, csv, config, mode="single", asr_model=None, router_candidates=None,
                 indiclid_ft_model=None, embed_model="ai4bharat/indic-bert", topk=4,
                 threshold=0.70):
        self.csv = csv
        self.config = config
        self.mode = mode
        self.asr_model = asr_model
        self.router_candidates = router_candidates or {}
        self.indiclid_ft_model = indiclid_ft_model
        self.embed_model = embed_model
        self.topk = topk
        self.threshold = threshold

        # init recorder
        self.recorder = MicRecorder(target_sr=16000)

        # NOTE: RetrieverWithIndicLID has been removed. This script only records audio,
        # transcribes, detects language and forwards the transcript to LangchainRAG.

        # init ASR mode
        if self.mode == "router":
            if not HAVE_ASR_ROUTER:
                raise RuntimeError("ASRRouter module not available but mode=router requested.")
            print("[i] Initializing ASR router with candidates:", self.router_candidates)
            self.asr_router = ASRRouter(self.router_candidates, device=("cuda" if torch.cuda.is_available() else "cpu"))
            self.asr_single = None
        else:
            print("[i] Initializing single ASR model:", self.asr_model)
            self.asr_single = SingleASR(model_name=self.asr_model, device=("cuda" if torch.cuda.is_available() else "cpu"))
            self.asr_router = None

        self.rag = LangchainRAG(csv_path=self.csv, config_path=self.config, indiclid_ft_model=self.indiclid_ft_model,
                        generator_model="google/flan-t5-large", device=("cuda" if torch.cuda.is_available() else "cpu"),
                        top_k=self.topk)

    def run_interactive_loop(self):
        print("Starting interactive STT+RAG loop. Press ENTER to record, type 'exit' to quit.\n")
        try:
            while True:
                # Record audio
                audio_np, sr = self.recorder.record_until_enter(
                    prompt="Press ENTER to start recording now (Enter again to stop, or type 'exit' to quit)..."
                )

                # Allow user to exit early before recording
                if audio_np.size == 0:
                    cmd = input("No audio captured. Type 'exit' to quit or ENTER to retry: ").strip().lower()
                    if cmd in ("exit", "quit"):
                        break
                    else:
                        continue

                print(f"[i] Recorded {len(audio_np)} samples at {sr} Hz ({len(audio_np)/sr:.2f} sec).")

                # Transcribe using selected mode
                if self.mode == "router":
                    transcript, chosen_model, score = self.asr_router.transcribe_with_best(audio_np, sr)
                    print(f"[ASR router] chosen model: {chosen_model}  score={score:.3f}")
                else:
                    transcript, score = self.asr_single.transcribe_ndarray(audio_np, sr)
                    chosen_model = self.asr_model
                    print(f"[ASR single] model: {chosen_model}  score={score:.3f}")

                print("[ASR] Transcript:", transcript)

                # Confidence check
                if score < 0.25:
                    print("[Bot] I couldn't hear clearly (low ASR confidence). Could you please repeat?")
                    continue

                # Retrieval + generation
                result = self.rag.run(transcript, return_tts=True)
                print("\n[Bot]:", result["generated_en"])

                if result.get("tts_path"):
                    print("[i] Playing reply audio:", result["tts_path"])
                    import soundfile as sf
                    import sounddevice as sd
                    data, sr = sf.read(result["tts_path"], dtype='float32')
                    sd.play(data, sr)
                    sd.wait()

                # Ask if continue or exit
                cmd = input("\nPress ENTER to continue, or type 'exit' to quit: ").strip().lower()
                if cmd in ("exit", "quit"):
                    break

        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt — exiting interactive loop.")

# ----------------------
# CLI
# ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default='data/schemes_multilingual.csv')
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mode", choices=["single", "router"], default="router",
                   help="single = single ASR model; router = multi-model router")
    p.add_argument("--asr_model", default="openai/whisper-small", help="Single ASR model id")
    p.add_argument("--router_candidates", default='{"hi":"ai4bharat/indicwav2vec-hindi","pa":"manandey/wav2vec2-large-xlsr-punjabi","en":"openai/whisper-small"}',
                   help='JSON string for router candidates e.g. \'{"hi":"ai4bharat/indicwav2vec-hindi","pa":"ai4bharat/indicwav2vec-punjabi","multi":"openai/whisper-small"}\'')
    p.add_argument("--indiclid_ft_model", default=None, help="Path to IndicLID fasttext .bin (optional)")
    p.add_argument("--embed_model", default="ai4bharat/indic-bert")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()

    router_candidates = None
    if args.router_candidates:
        import json
        router_candidates = json.loads(args.router_candidates)

    orchestrator = RealtimeSTT(csv=args.csv,
                                          config=args.config,
                                          mode=args.mode,
                                          asr_model=args.asr_model,
                                          router_candidates=router_candidates,
                                          indiclid_ft_model=args.indiclid_ft_model,
                                          embed_model=args.embed_model,
                                          topk=args.topk,
                                          threshold=args.threshold)
    orchestrator.run_interactive_loop()


if __name__ == "__main__":
    main()
