"""
indic_tts.py
Indic Text-to-Speech with language-based folder organization (class-based).
"""

import os
import uuid
import torch
import soundfile as sf
import warnings
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from langdetect import detect

class IndicTTS:
    def __init__(self, 
                 model_id: str = "ai4bharat/indic-parler-tts", 
                 output_root: str = "tts_outputs",
                 device: str = None):
        """
        Initialize the Indic TTS model (loaded once).
        """
        self.model_id = model_id
        self.output_root = output_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Silence logs & warnings
        warnings.filterwarnings(
            "ignore",
            message=r"Config of the .* is overwritten by shared .* config"
        )

        print(f"🔄 Loading model {self.model_id} on {self.device} ...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)

        print("✅ Indic TTS model loaded successfully")

    def text_to_speech(self, text: str, caption: str = None) -> str:
        """
        Convert text (any Indian language) into speech and save as WAV.
        Saves under: tts_outputs/<lang_code>/<uuid>.wav
        Returns: path to saved file
        """
        caption = caption or "A clear, natural Indian voice."

        # detect language
        try:
            lang_code = detect(text)  # e.g., 'hi', 'pa', 'ta'
        except Exception:
            lang_code = "unknown"

        # prepare output folder
        out_dir = os.path.join(self.output_root, lang_code)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"tts_{uuid.uuid4().hex}.wav")

        # tokenize
        desc_inputs = self.desc_tokenizer(caption, return_tensors="pt").to(self.device)
        text_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # ensure both have attention_mask
        if "attention_mask" not in desc_inputs:
            desc_inputs["attention_mask"] = torch.ones_like(desc_inputs["input_ids"]).to(self.device)
        if "attention_mask" not in text_inputs:
            text_inputs["attention_mask"] = torch.ones_like(text_inputs["input_ids"]).to(self.device)

        # generate
        self.model.eval()
        with torch.no_grad():
            audio = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
            )

        # convert to numpy + save wav
        audio = audio.cpu().numpy().squeeze()
        sr = int(getattr(self.model.config, "sampling_rate", 24000))
        sf.write(out_path, audio, sr)
        print(f"✅ Saved speech to {out_path} (sr={sr}, lang={lang_code})")
        return out_path


# ---------------- DEMO ----------------
if __name__ == "__main__":
    tts = IndicTTS()

    # Hindi
    tts.text_to_speech("नमस्ते दुनिया")

    # Punjabi
    tts.text_to_speech("ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੁਨਿਆ")

    # Tamil
    tts.text_to_speech("வணக்கம் உலகம்")
