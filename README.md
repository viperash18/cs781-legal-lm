# CS781 Capstone — In-House Legal Intelligence System

**Ashwary Rathore** · roll 240218 · IIT Kanpur · Codabench user `a_240218`

A seven-phase pipeline for Indian legal text, built entirely from our own models: four language models of 185–192M
parameters were pretrained from scratch (own tokenizer, own architectures), adapted with masked-LM training, and then
fine-tuned for named-entity recognition, rhetorical roles, judgment prediction, statute identification, summarization
and prior-case retrieval. No external pretrained weights, corpora or LLM APIs are used anywhere in training or inference.

## Results (official Codabench scores)

| Phase | Task | Metric | Best score | Rank | Best system |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | **0.8284** | 96 | arch2 (LLaMA-style), 2 epochs, T=0.7, top-k 40, repetition penalty 1.2 |
| 2 | Legal NER (L-NER) | strict macro-F1 | **0.6068** | 16 | arch3 + MLM, constrained Viterbi decoding, document-level role resolution |
| 3 | Rhetorical roles | macro-F1 | **0.4404** | 99 / 128 | arch3 + MLM, LoRA, stage-2 document BiLSTM |
| 4 | Court judgment prediction | macro-F1 | **0.5076** | 126 / 128 | arch1, frozen encoder + trained head |
| 5 | Statute identification | macro-F1 | **0.4328** | 14 | arch3 + TAPT full fine-tuning, blended with TF-IDF kNN and TF-IDF logistic regression |
| 6 | Abstractive summarization | BERTScore F1 | **0.8314** | 88 | arch3 prefix-LM, full fine-tuning + continued training, copy-biased beam search |
| 7 | Prior case retrieval | micro-F1@7 | **0.4920** | 9 | learned LambdaRank over lexical, citation-context and neural signals |

The full write-up (data analysis, ablations, error analysis) is in the report (`report/`).

## Repository layout

```
code/
  phase1/   tokenizer/ (own BPE: a1_tokenizer.py + corpus encoding), the four architectures, arch2's second epoch, generation
  phase2/   L-NER pipeline, role resolvers, LoRMA / LoRA / full-FT runs; mlm_adaptation/ (arch3 masked-LM adaptation)
  phase3/   arch3 LoRA + stage 2 (best), TAPT variant, LoRMA, KD, rsLoRA; mlm_adaptation/ (arch2, arch4 masked-LM adaptation)
  phase4/   freezing (best), LoRMA (two batch settings), full-FT teacher, KD student
  phase5/   arch3 TAPT full FT, kNN + linear blend (best), KD, LoRMA, partial FT; tapt/ (arch3 task-adaptive pretraining)
  phase6/   arch3 full FT, continued training, best decoding, LoRMA, LoRA, KD, partial FT, diagnostics
  phase7/   encoders (full FT, LoRMA, LoRA, partial, KD) and the fusion + ranker notebook (best)
report/     the final report (PDF)
```
Files whose name ends in `_BEST` produced the best Codabench submission of their phase. Notebooks are kept exactly as they
were run on Kaggle (outputs included), so each one documents its own settings and logs.

## Setup (all runs were made on Kaggle)

* **Hardware:** one NVIDIA T4 (16 GB) per run, sessions of at most 12 h; long runs save a checkpoint (best + last, with
  optimizer state) at every evaluation and resume from it. Phase-7 fusion runs on CPU.
* **Data:** the official course dataset on Hugging Face (`Exploration-Lab/CS-781-Capstone`). Add your Hugging Face token
  as a Kaggle secret named `HF_TOKEN` and switch Internet on; every notebook downloads its phase with `fetch_phase(PHASE)`.
  The data itself is not redistributed here.
* **Tokenizer:** attach a Kaggle dataset containing `legal_bpe_merges.json` (produced by `code/phase1/tokenizer`).
* **Checkpoints:** model checkpoints (`*.pt`, 0.7–2.2 GB) are not in this repository; each notebook names the file it
  expects in its `CKPTS` dictionary. Attach the dataset that holds it as a Kaggle input.
* **Python packages:** see `requirements.txt` (Kaggle's default image already has all of them).
* **Phases 3–7 use one master notebook.** Only its first cell (the `CONTROL` cell: `ARCH`, `PHASE`, `TECHNIQUE`, `BIDIR`,
  `CKPTS`, `HP`, the phase-specific `P*_MODE` and `P*` dictionaries) differs between runs. Run All / Save Version, then
  download `submission_*.zip` (it contains `prediction.json` at its root) from the notebook's Output.

## Tokenizer and corpus — `code/phase1/tokenizer/`

* Byte-level BPE written from scratch (`a1_tokenizer.py`, Assignment 1/3), pre-tokenizer
  ``" ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+"``, trained on 100M characters of Supreme Court judgments (4,225 documents):
  65,354 tokens + an end-of-document token (vocabulary 65,355). Masked-LM checkpoints add `[MASK]` (65,356).
* Corpus: the **Supreme Court** subset of the Phase-1 data (the other sources are Acts, High Courts and Tribunals),
  encoded to **200M training / 5M validation tokens** (`train_ids_bpe.npy`, `val_ids_bpe.npy`, uint16); 5% of documents go
  to validation.
* Run `tokenizer_and_corpus.ipynb` once (CPU, ~30 min for the BPE, ~2 min for encoding) and save the three files as a dataset.

## Phase 1 — Legal LM pretraining — `code/phase1/`

| Arch | Design | d / layers / heads (KV) | FFN | Positions | Norm | Params |
|---|---|---|---|---|---|---|
| 1 | classic decoder, own attention (pre-norm, tied head) | 1000 / 10 / 8 | GELU 4,000 | sinusoidal | LayerNorm | 185,447,000 |
| 2 | LLaMA-style decoder | 1024 / 10 / 16 (8) | SwiGLU 2,816 | RoPE | RMSNorm | 184,909,824 |
| 3 | prefix-LM on arch-2 blocks: bidirectional prefix (random 2–1,000 tokens), causal continuation, loss on continuation only | 1024 / 10 / 16 (8) | SwiGLU 2,816 | RoPE | RMSNorm | 184,909,824 |
| 4 | deep–thin dense + MoE hybrid: odd layers are mixture-of-experts (4 experts × 1,024, top-2, load-balancing loss 0.01) | 768 / 16 / 12 (6) | SwiGLU 2,048 | RoPE | RMSNorm | 191,800,320 |

* **Training (all four):** context 1,024; 499,712 tokens per optimizer step; AdamW, peak LR 6e-4, 40 warmup steps, cosine to
  6e-5, weight decay 0.1, gradient clipping 1.0, dropout 0.1, fp16 with gradient scaling.
  Best checkpoints: arch1 step 375 (val loss 6.652), arch2 step 400 + a second epoch to step 675 (3.527 → **2.991**,
  `arch2_second_epoch_BEST.ipynb`), arch3 step 399 (3.585, continuation positions only), arch4 step 300 (3.832).
* **Generation:** continue each 512-word prompt to about 128 words. Decoding moved from pure sampling to temperature 0.7 +
  top-k 40, then to full prompts (up to 768 tokens) + repetition penalty 1.2 (`generate_rep_penalty_full_prompts.ipynb`;
  the best submission uses these settings on the second-epoch arch2 checkpoint).
* **Run:** attach the corpus dataset, set `MAX_STEPS` from the measured tokens/s so a commit fits in ~11 h, Save Version.

## Domain adaptation — `code/phase2/mlm_adaptation/`, `code/phase3/mlm_adaptation/`, `code/phase5/tapt/`

Continued **masked-LM** training of the Phase-1 models gives bidirectional encoders for Phases 2–5 and 7: 15% BERT-style
masking (80/10/10), loss on masked positions only, whole window bidirectional. arch3: 975 steps (val 2.037, step 925 used);
arch2: 389 steps (val 2.066); arch4: 339 steps (val 2.838). **TAPT** continues this on each phase's own training text
(arch3 for Phases 3 and 5).

## Phase 2 — Legal NER — `code/phase2/`

* **Method:** BIO token classification (25 classes) over 1,024-token windows with stride 512; each token takes its logits
  from the window where it is most central; class-weighted cross-entropy. Character offsets are recovered exactly by
  encoding with the tokenizer's own pattern and mapping bytes back to characters; spans are then trimmed by boundary rules.
* **Decoding:** constrained Viterbi (valid BIO transitions; every BPE piece of a word gets one label), then a
  document-level resolver for appellant/respondent and their counsel (`my_role_resolver_BEST.py`, `role_resolution_BEST.ipynb`).
* **Settings:** arch3-MLM, 10 epochs, LR 1e-5 (encoder) / 1e-4 (head), batch 4 windows, dropout 0.1, class-weight power 0.2.
* **Run:** `lner_arch3_mlm_pipeline_BEST.ipynb` (train on folds 1+2, predict fold 3), then `role_resolution_BEST.ipynb` on its
  predictions. Other runs: arch2 + LoRMA, arch4 + LoRA, arch2 full fine-tuning.

## Phase 3 — Rhetorical roles — `code/phase3/`

* Sentences are packed into 1,024-token windows (≤166 tokens per sentence, EOS between sentences); each sentence is the mean of
  its token states → BiLSTM (hidden 256) → linear head; focal loss (γ = 2); CL and IT trained jointly.
* Stage 2 freezes the encoder and trains a document-level BiLSTM on whole documents (60 epochs, LR 2e-5).
* **Best:** `arch3_lora_stage2_BEST.ipynb` (LoRA r = 8, α = 8 on q/k/v/out, 30 epochs, LR 4e-4 / 1e-4): dev macro-F1
  0.4447 → 0.4627 after stage 2. Other runs: TAPT encoder, arch2 + LoRMA, arch4 KD student, arch1 + rsLoRA.

## Phase 4 — Court judgment prediction — `code/phase4/`

* Each case is split into 1,024-token chunks (20% overlap, at most 8, keeping the first and last), encoded, pooled and
  classified; one label per case; dev is checked every 4,000 cases and the best-dev weights are kept.
* **Best (Codabench):** `arch1_freezing_BEST.ipynb`: frozen encoder, trained head (LR 1e-4, 32 cases per step).
* **Best on dev (0.6113, not submitted):** `arch2_lorma_32_cases_per_step.ipynb`. The same LoRMA run with 1 case per step
  (`arch2_lorma_1_case_per_step.ipynb`) collapsed to a single class.

## Phase 5 — Statute identification — `code/phase5/`

* Encoder states → max-pool over all tokens of all chunks → 100 sigmoid outputs; BCE with per-label positive weights
  (power 0.7); per-label thresholds tuned on dev.
* **Best:** `knn_linear_blend_BEST.ipynb` blends the arch3-TAPT model (`arch3_tapt_full_ft.ipynb`, 2 epochs, LR 2e-5 / 1e-4)
  with a TF-IDF kNN over train + dev and a TF-IDF logistic regression (weights 0.1 / 0.3 / 0.6).
* Other runs: arch4 KD student (2 of 16 blocks), arch2 + LoRMA, arch1 partial fine-tuning.

## Phase 6 — Abstractive summarization — `code/phase6/`

* Decoder-only formulation: the first 550 tokens of the judgment are the (bidirectional, for arch3) prefix, the summary is the
  continuation (up to 470 tokens); loss on summary tokens only. 100 training documents are held out for local BERTScore.
* **Best:** `arch3_full_ft.ipynb` (2 epochs, LR 1e-4, 32 documents per step) → `arch3_continued_training.ipynb` (2 more
  epochs from the saved weights) → `arch3_decoding_BEST.ipynb`: beam 8, length penalty 1.8, 200–470 new tokens,
  repetition penalty 1.2, no-repeat n-gram 3, and a copy bias (+4 to the logit of every input token, +4 to the token that
  continues a 4-token phrase of the input).
* Other runs: arch2 + LoRMA, arch4 + LoRA, arch4 KD student, arch1 partial fine-tuning; `diagnostics.ipynb` analyses the outputs.

## Phase 7 — Prior case retrieval — `code/phase7/`

* **Encoders:** a shared bi-encoder (1,024-token chunks, mean-pooled, L2-normalised) trained with InfoNCE (τ = 0.05) against one
  relevant and three TF-IDF hard negatives from each query's top 50; arch3 full FT (`arch3_full_encoder.ipynb`), arch2 LoRMA,
  arch4 LoRA, arch1 partial FT, and an arch3 KD student. Each run also saves document embeddings for the fusion step.
* **Best:** `fusion_and_ranker_BEST.ipynb` (CPU). Cells 7.5–7.15, run in order in one session, build per (query, candidate)
  signals: word 1–2/3/4-gram TF-IDF, citation kNN over training queries, candidate prior, judgment-year order, matching of
  each `<CITATION>` context against candidates (best match, best-two mean, per-citation votes, several context widths),
  rare 8-gram overlap, and the encoders' similarities. A LightGBM LambdaRank model (raw, within-query z-score and rank
  features; 3 seeds) is trained on the 827 training queries, fused with a dev-tuned linear system, checked with 2-fold
  cross-validation on dev, and finally refit on train + dev (cell 7.14, `submission_p7_fusion_v9_trdev.zip`).

## Reproducing a submission

1. Import the notebook into Kaggle, attach the tokenizer dataset and the checkpoint dataset it names, add `HF_TOKEN`,
   enable Internet and a GPU (CPU for Phase-7 fusion).
2. Keep the `CONTROL` cell as saved (it holds the exact settings of that run). Save Version (Run All).
3. Download `submission_*.zip` from the Output tab without letting the browser unzip it, and upload it to Codabench.

## Use of AI assistance

Declared in full in the report's LLM Usage Declaration. In short: the Phase-1 architectures and training code and the task
models (heads, losses, PEFT modules, KD students) are the author's own. Claude (Anthropic) wrote supporting code (data
loading, training/evaluation harnesses, decoding, metrics, submission writers, diagnostics, plotting), explained run
diagnostics during development, suggested top-k 40 for Phase-1 decoding (GPT-2's published value), wrote the search code
that evaluated sweep settings, blend weights and thresholds on dev in Phases 3 and 5, and, with the instructor's permission
for Phase 7, chose the Phase-7 retrieval settings and built the Phase-7 fusion pipeline. All other hyperparameters were
chosen by the author. The report text and this README were drafted with Claude, as permitted by the instructor's
announcement on LLM use for reports, and checked by the author.

## Key references

Vaswani et al. 2017 (Transformer) · Touvron et al. 2023 (LLaMA) · Raffel et al. 2020 (T5, prefix-LM) · Shazeer et al. 2017 and
Fedus et al. 2022 (mixture of experts) · Devlin et al. 2019 (BERT) · Gururangan et al. 2020 (TAPT) · Hu et al. 2022 (LoRA) ·
Bihany et al. 2025 (LoRMA) · Hinton et al. 2015 (distillation) · Joshi et al. 2024 (IL-TUR) · Malik et al. 2021 (ILDC) ·
Shukla et al. 2022 (IN-Abs) · Joshi et al. 2023 (U-CREAT / IL-PCR) · Burges 2010 (LambdaMART) · Ke et al. 2017 (LightGBM).
