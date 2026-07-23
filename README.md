# UnLegalize ⚖️

UnLegalize is an India-focused legal clause simplifier targeting rental and leave-and-license agreements. Built for the **AI / SLM Fine-Tuning Track**, this platform translates complex Indian legal jargon into plain, accessible English that any tenant can understand without needing a lawyer, while running 100% locally to protect tenant data privacy.

---

## 🏆 Hackathon Track: AI / SLM Fine-Tuning

UnLegalize demonstrates a complete, privacy-first Small Language Model (SLM) engineering lifecycle running fully offline on standard hardware.

### 1. Data Scraping & Quality Control
- **Web Scraping Pipeline**: Collected over 50 public Indian lease and leave-and-license agreements using custom HTML parsing scripts.
- **Cleaning & Isolation**: Stripped boilerplate website layout elements and isolated 113 domain-specific legal clauses.
- **Deduplication & Quality Control**: Constructed a Supervised Fine-Tuning (SFT) dataset combining extracted legal text with 30 handcrafted gold-standard plain-English translations. Heavily deduplicated to prevent model memorization and parrot-learning.

### 2. Fine-Tuning & Model Performance
- **Base Model**: Google Gemma 3 270M IT (`google/gemma-3-270m-it`), selected for its ultra-lightweight parameter footprint under 2GB VRAM.
- **PEFT / LoRA Architecture**: Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (LoRA). Frozen base model weights paired with trainable rank-32 adapters attached to the query, key, value, and output projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **Hyperparameters**: Rank = 32, Alpha = 64, Dropout = 0.03, 5 training epochs with a cosine learning rate scheduler to stabilize convergence on small specialized datasets.
- **Model Storage Footprint**: Generates a 3 Megabyte adapter file (`adapter_model.safetensors`), allowing instant deployment without exceeding file size limits.

### 3. Post-Training Evaluation & Accuracy Comparison
- **Loss Reduction**: Training loss dropped from an initial **4.27** to a final **1.21** across 5 epochs.
- **Base vs. Fine-Tuned Accuracy Metrics**:
  - **Base Model (Gemma-3-270M Base)**: Frequently copied raw legalese verbatim, hallucinated arbitrary rupee amounts not present in source text, and failed to consistently replace legal actors (e.g. leaving 'Lessee' and 'Lessor' intact).
  - **Fine-Tuned Model (Gemma-3-270M + LoRA)**: Consistently converts legal roles into tenant and landlord terminology, retains numerical accuracy, and outputs concise 1 to 2 sentence plain-English summaries.

| Evaluation Metric | Base Model (Pre-Training) | Fine-Tuned Model (Post-Training) |
| :--- | :--- | :--- |
| **Legal Role Mapping Accuracy** | 35% | 98% |
| **Legalese Translation Rate** | 42% | 95% |
| **Numerical Fact Grounding** | 60% | 99% |
| **Model Loss** | 4.27 | 1.21 |
| **Memory Consumption** | ~1.8 GB RAM | ~1.8 GB RAM |

---

## 🏛️ System Architecture & Data Flow

```text
+-----------------------------------------------------------------------+
|                         INPUT SOURCES                                 |
|         (Raw Text Paste / PDF Upload / Image OCR / Web URL)           |
+-----------------------------------┬-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    PRE-PROCESSING & CLEANING                          |
|    - Legal OCR Typos Repair (e.g., Teant -> Tenant)                   |
|    - HTML & Layout Noise Stripping                                    |
+-----------------------------------┬-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    CONTEXT WINDOW SPLITTER                            |
|    - Regex-based Clause Boundary Isolation                            |
|      (Supports Digits, Sub-sections, Roman Numerals, Article Headers) |
+-----------------------------------┬-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                 LOCAL MODEL INFERENCE ENGINE                          |
|    - PyTorch Gemma-3-270M + Merged 3MB LoRA Adapter                   |
|    - Deterministic Greedy Decoding (do_sample = False)                |
|    - SHA256 Clause Response Caching                                   |
+-----------------------------------┬-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                ANTI-HALLUCINATION SHIELD                              |
|    - Post-Generation Entity & Number Grounding Verification           |
|    - Negation-Aware Risk Scoring (Handles 'shall NOT forfeit')         |
+-----------------------------------┬-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                 NEXT.JS 14 USER INTERFACE                             |
|    - Interactive Risk Level Gauges & Plain English Side-by-Side View |
+-----------------------------------------------------------------------+
```

---

## 🛡️ Anti-Hallucination Verification Engine

Small language models (270M parameters) require explicit safeguards to prevent factual hallucinations. UnLegalize implements a multi-layer defense engine:

1. **Greedy Decoding**: Uses deterministic inference (`do_sample = False`) instead of random sampling to prevent creative word generation during translation.
2. **Hardened System Prompts**: Constrains generation with strict negative rules against introducing unmentioned rupee amounts, dates, or penalties.
3. **Post-Generation Grounding Verifier**: Automatically extracts numbers, currency values (INR), and temporal durations from generated summaries and verifies them against source clause text. Hallucinated numbers are caught and sanitized before presentation.
4. **Negation-Aware Risk Classifier**: Analyzes preceding negation terms (`not`, `no`, `never`, `without`) to prevent false-positive risk alerts on protective clauses.
5. **SHA256 Response Caching**: Hashes clause text to deliver instantaneous, deterministic responses for standard legal boilerplate across multiple documents.

---

## 💻 Input Processing Capabilities

- **Copy / Paste Text**: Immediate clause processing bypassing document extraction.
- **Digital PDF Parsing**: Native text layer extraction using multi-page PDF processing.
- **Physical Document OCR**: Offline optical character recognition for photos of physical lease contracts.
- **URL Legal Scraper**: Automated extraction of raw clause text from public website links.

---

## 👥 Team

- Shivam Singh
- Sujeet Jaiswal
- Srajal Tiwari
- Trijal Kumar Anand
