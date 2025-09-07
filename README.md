# 💡 Multilingual-Vernacular-Voicebot-with-RAG

This project is an **open-source conversational AI assistant** designed for **rural & semi-urban India**, where users can ask questions about **government schemes, banking, and agriculture advisory** in their **local languages** (Hindi, Punjabi, Tamil, Bengali, etc.).

The system supports:

- 🎙️ **Voice-based queries** in multiple Indic languages
- 🔊 **Speech-to-Text (STT)** with Whisper & AI4Bharat ASR models
- 🌐 **Retrieval-Augmented Generation (RAG)** for factual answers
- 🌍 **Translation** between Indic ↔ English (AI4Bharat IndicTrans2)
- 🗂️ **Knowledge base** built from multilingual scheme/advisory documents
- 🤖 **LLM-powered responses** (Flan-T5, IndicGPT, etc.)
- 🔈 **Text-to-Speech (TTS)** in the user’s own language (Indic-TTS)
- 🧠 **Conversational memory** to handle follow-ups naturally

---

## 📂 Project Structure
├── ai4bharat_tts.py\
├── audio_asr_router.py\
├── config.yaml\
├── data\
│   └── schemes_multilingual.csv\
├── langchain_rag.py\
├── models\
│   └── indictrans2-indic-en-1B\
├── README.md\
├── realtime_stt.py\
├── retrieval_with_indiclid.py\
├── tts_outputs\
│   ├── en\
│   │   ├── tts_24314ac84a404a42a98de6d702fd82bc.wav\
│   │   ├── tts_95c2930869af4eb8acf90814f7184fd9.wav\
│   │   └── tts_e420035404a9430682354f9cccfe1890.wav\
│   ├── hi\
│   │   ├── tts_04484fec38e84e44a73541e569bc790e.wav\
│   ├── pa\
│   │   ├── tts_10f3bbd9439f4780898a0f7eae934adf.wav\
│   │   ├── tts_6165c68bd38f4d5781dbcbc5db103b9a.wav\
├── validate_and_clean.py\
└── validation_out\
    └── validation_report.json\


---
## ⚙️ Pipeline Flow

    A[📂 Dataset: schemes_multilingual.csv] --> B[🧹 validate_and_clean.py → Cleaned corpus]\
    B --> C[🎙 realtime_stt.py (MicRecorder → Audio capture)]\
    C --> D[📝 audio_asr_router.py / realtime_stt.py (ASR → Transcript)]\
    D --> E[🌐 realtime_stt.py (Language ID)]\
    E --> F[🌍 langchain_rag.py (IndicTranslator: Vernacular → English)]\
    F --> G[📖 retrieval_with_indiclid.py / langchain_rag.py (Retriever → Context passages)]\
    G --> H[🤖 langchain_rag.py (Generator: Flan-T5 → Draft Answer in English)]\
    H --> I[🔄 langchain_rag.py (IndicTranslator: English → Vernacular)]\
    I --> J[🗣 langchain_rag.py (IndicTTS → Voice synthesis)]\
    J --> K[🔊 realtime_stt.py (Audio playback)]\
    K --> L[🧠 langchain_rag.py (Conversational Memory → Follow-ups)]\

---
### 📑 Technical Notes
⚡ **Model Choices**
- **ASR (STT)**: AI4Bharat IndicWav2Vec (Hindi, Punjabi) + Whisper (multilingual fallback).
- **Retriever**: IndicLID embeddings (multilingual coverage of Indic scripts).
- **Generator**: Flan-T5 (open-source, smaller footprint, reliable factuality).
- **Translator**: IndicTrans2 (robust bidirectional translation between Indic ↔ English).
- **TTS**: Indic-TTS (natural speech in vernacular).

---

### 🔗 Orchestration Flow

1. Audio → Transcript (ASR)
2. Transcript → Language ID → Translation (to English)
3. Retrieval (semantic search with IndicLID + FAISS)
4. Generation (Flan-T5 → English answer)
5. Re-translation (English → vernacular)
6. TTS → Audio reply
7. Memory buffer → follow-up handling

---

### 🌍 Language Consistency
- Always normalize reasoning in English to avoid model drift.
- Translate responses back into user’s language with IndicTrans2.

---

### 🛡️ Anti-Hallucination
- Confidence scoring for ASR + Retriever.
- If score < threshold → Bot replies with “Mujhe is vishay mein jaankari nahi hai.”
- Deterministic generation (beam search, no sampling) to reduce randomness.

---

## 🔄 End-to-End Voicebot Pipeline

**1. Dataset Preparation (Corpus Prep)**

- `validate_and_clean.py`→ Cleans & validates the multilingual dataset (Govt schemes/advisories), fixes encoding, missing translations, and prepares a clean CSV for indexing.

**2. Speech-to-Text (ASR)**

- `audio_asr_router.py` → Routes between multiple ASR models (`ai4bharat/indicwav2vec-hindi`→ Hindi ASR, `manandey/wav2vec2-large-xlsr-punjabi` → Punjabi ASR, `openai/whisper-small` → Multilingual ASR (fallback/general-purpose) ) based on confidence.

- `realtime_stt.py` → Records live microphone input, transcribes using single ASR model or router.

**3. Language Detection**
- `realtime_stt.py` → Uses langdetect (if available) or Unicode heuristics to identify query language.

**4. Translation (Vernacular → English, if needed)**

- `langchain_rag.py (inside IndicTranslator class) → Uses `ai4bharat/indictrans2-indic-en` → Indic → English and 
`ai4bharat/indictrans2-en-indic` → English → Indic.

**5. Retrieval (Semantic Search)**
- `retrieval_with_indiclid.py` → Embeds corpus with **IndicLID** from AI4Bharat)  and retrieves relevant passages.
- `langchain_rag.py` → Integrates retrieval step into RAG pipeline.

**6. Generation (Answer Synthesis in English)**
- `langchain_rag.py` (Generator class) → Uses `google/flan-t5-large` → Factual, deterministic generation via beam search.

**7. Re-Translation (English → User’s Language)**

- `langchain_rag.py` → Translates generated English response back into the user’s vernacular with `ai4bharat/indictrans2-en-indic` → English → Indic.

**8. Text-to-Speech (Voice Response)**
- `langchain_rag.py` → Converts final vernacular text into speech using IndicTTS.
- `realtime_stt.py` → Plays back audio to the user.

**9. Conversational Memory & Follow-ups**
- `langchain_rag.py` → Maintains short-term session memory (last N turns), rewrites follow-up queries, and ensures contextual continuity.

## 🔍 In-Depth Script Descriptions

## Script 1: realtime_stt.py

This script provides a **real-time interactive voicebot loop**.  
It records audio from the microphone, runs speech-to-text, detects language, retrieves context via RAG, generates an answer, and plays back a speech response.

### Features
- Live audio capture (mic → numpy array).
- **Configurable ASR modes:**
- - `--mode single` → Uses a single ASR model specified by `--asr_model`.
- - `--mode router` → Uses multiple ASR candidates (`--router_candidates`) and automatically picks the best one based on confidence.
- STT with Whisper / AI4Bharat IndicWav2Vec models.
- Supports single ASR model or ASR router mode.
- Language detection (via langdetect or Unicode heuristics).
- Integrates with `LangchainRAG` for retrieval + generation.
- Plays back responses via Indic TTS.

### Usage
**Single ASR model (Whisper):**
```bash
python realtime_stt.py \
    --csv data/schemes_multilingual.csv \
    --config config.yaml \
    --mode single \
    --asr_model openai/whisper-small 
```
**Router mode (multiple ASR models for Hindi, Punjabi, English):**
```bash
python realtime_stt.py \
    --csv data/schemes_multilingual.csv \
    --config config.yaml \
    --mode router \
    --router_candidates '{"hi":"ai4bharat/indicwav2vec-hindi","pa":"manandey/wav2vec2-large-xlsr-punjabi","en":"openai/whisper-small"}'
```

## Script 2: audio_asr_router.py

This module implements a multilingual ASR (Automatic Speech Recognition) router for handling user speech queries.  
It evaluates candidate ASR models and selects the most suitable one for transcription.

### Features
- Supports multiple Indic + multilingual ASR models (e.g., Hindi, Punjabi, Whisper).
- Confidence scoring for selecting the best model.
- Handles audio resampling automatically.
- Falls back to Hugging Face `pipeline` when direct decoding fails.

### Usage
```bash
python audio_asr_router.py
```

## Script 3: langchain_rag.py

This script orchestrates the **Retrieval-Augmented Generation (RAG)** pipeline for the voicebot.  
It ties together retrieval, translation, generation, re-translation, and TTS.

### Features
- Handles end-to-end query flow: Language detection → Translation → Retrieval → Generation → Re-translation → TTS → Conversational memory.
- Detects query language (English, Hindi, Punjabi, etc.).
- Translates Indic ↔ English using AI4Bharat IndicTrans2.
- Retrieves top-k relevant passages via FAISS index (RetrieverWithIndicLID).
- Generates factual answers with Flan-T5 (deterministic beam search).
- Re-translates answer into the user’s vernacular.
- Provides spoken response using IndicTTS.
- Maintains conversational memory for follow-ups.
- Confidence handling: falls back to “Mujhe is vishay mein jaankari nahi hai.” when unsure.

### Usage
```bash
python langchain_rag.py \
    --csv data/schemes_multilingual.csv \
    --config config.yaml \
    --indiclid_ft_model ./models/IndicLID-FTR-v1.bin \
    --generator_model google/flan-t5-large
```


## Script 4: retrieval_with_indiclid.py

This script handles **semantic document retrieval** using IndicLID embeddings.  
It is part of the RAG pipeline that powers the voicebot’s knowledge base.

### Features
- Embeds multilingual documents (English, Hindi, Punjabi, etc.) into a shared semantic space.
- Builds and queries a vector store (FAISS/Chroma).
- Supports semantic similarity search for user queries.
- Outputs top-k relevant passages for downstream LLM response generation.

### Usage
```bash
python retrieval_with_indiclid.py
```


## Script 5: validate_and_clean.py

This script validates and cleans the multilingual dataset (CSV) of schemes and advisory content.  
It checks encoding, language correctness, missing/invalid values, and generates a report.

### Features
- Detects CSV encoding automatically.
- Validates presence and correctness of language fields (English, Hindi, Punjabi, etc.).
- Flags missing IDs, empty rows, or invalid `source_url`s.
- Generates a structured JSON report and CSV with problematic rows.
- Optionally outputs a cleaned UTF-8 CSV with consistent formatting.

### Usage
```bash
python validate_and_clean.py \
    --input data/schemes_multilingual.csv \
    --config config.yaml \
    --output_dir ./validation_out \
    --clean
```


