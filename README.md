# Persian ASR with LLM-Based Text Refinement

A two-stage Persian (Farsi) speech recognition system: a progressively fine-tuned Whisper model for speech-to-text, followed by an LLM-based refinement layer targeting output readability rather than raw WER minimization.

## Goal

Build a Persian ASR system with acceptable quality through two stages:

1. **Progressive fine-tuning** of `openai/whisper-small` on Common Voice Persian, done incrementally over multiple rounds due to Kaggle's free-tier resource limits.
2. **A text refinement layer** built on a fine-tuned LLM, aimed at improving the readability of Whisper's raw output — particularly half-spaces (*nim-fāsele*), punctuation, and lexical errors caused by phonetic similarity.

The project's explicit goal was to **improve output quality and readability**, not necessarily to minimize Word Error Rate (WER) — a distinction that turns out to matter a great deal in the final results (see below).

## 1. Data Collection

**Dataset:** Common Voice Scripted Speech Persian 26.0
- 394,397 audio clips (430.86 hours) with matching Persian transcripts
- 373.24 validated hours, from 4,660 speakers, 354,627 sentences
- Only clips that passed Common Voice's validation step (`validated_sentences.tsv`) were used, to ensure data quality
- Dataset host migrated to the new Mozilla Data Collective SDK during the project ([dataset link](https://mozilladatacollective.com/datasets/cmqinhw5100v8nr07gyg5gi4vf))

**Selection process (per round):**
- The full metadata table was shuffled once with a fixed seed, for reproducibility
- Clips were then added in shuffle order, provided the sentence was under 400 characters and the audio file was readable
- Selection continued until the round's target duration was reached (e.g., 20 hours for round 1)
- Selection was fully random (not stratified by speaker, gender, or accent), but fully reproducible thanks to the fixed seed
- A running `used_clips.txt` manifest tracked every clip used in every round, to prevent unplanned repeat exposure across rounds — except for a small, deliberately reselected portion used for the Replay strategy below

## 2. Whisper Fine-Tuning: Progressive Learning with Replay

Kaggle's session-length and weekly GPU quota made it impossible to fine-tune on the full 80-hour target dataset in one pass. The training was split into **5 rounds**, each loading from the *previous round's checkpoint* (not the base model) and training on new data plus a replayed slice of prior rounds' data — a technique used specifically to prevent catastrophic forgetting.

| Round | New Data | Replay Strategy | Final WER |
|---|---|---|---|
| 1 | 20 h | — | 33.40 |
| 2 | 15 h | 15% of prior rounds' data | 28.50 |
| 3 | 15 h | 10% of prior rounds' data | 27.66 |
| 4 | 15 h | 10% of prior rounds' data | 25.87 |
| 5 | 15 h | max 5 h of prior rounds' data (fixed cap) | **24.15** |

After 5 rounds (~80 hours of effective data, including replay), WER on the eval set dropped from 33.40 to **24.15** — roughly a 9-point improvement. Gains slowed with each round (diminishing returns), suggesting the model was approaching the learning ceiling for the `whisper-small` architecture at this data scale — which is why round 5 was chosen as the final Whisper fine-tuning round.

**Final model:** [`amirsz8203/whisper-small-fa-finetuned`](https://huggingface.co/amirsz8203/whisper-small-fa-finetuned)

## 3. LLM Refinement Layer

### Initial approach: prompting only (no fine-tuning)
Whisper's raw output was first passed through a general-purpose LLM (Qwen2.5-7B-Instruct) with a tightly scoped prompt restricted to fixing half-spaces and punctuation, with strict rules against adding or removing words.

This worked reliably for half-space/punctuation fixes, but was nearly useless against genuine lexical errors — e.g. a phonetically similar word substitution like *garmz* for *ghermez* ("red") — since without hearing the audio, the LLM has no reliable way to catch these.

### Moving to fine-tuning
To address lexical errors, the project moved to fine-tuning a dedicated LLM, with several iterations along the way:

- **Model choice:** Qwen was tried first; **Gemma 3 (4B)** was ultimately selected, fine-tuned via **Unsloth + QLoRA** for efficient training on limited GPU.
- **Dataset construction:** training pairs (raw Whisper output → correct text) were built from two sources — each Whisper round's eval data, and samples from the rounds' training sets.
  - *Quality filtering:* pairs with mismatched word counts were dropped (to avoid teaching the model to add/remove words). A text-similarity threshold (≥40%) filtered out pairs that were too unrelated — likely severe Whisper errors on unfamiliar audio rather than simple lexical mistakes.
  - *Balance correction:* most pairs initially required "no change," which risked teaching the model to always leave text untouched. Adding more substitution examples rebalanced the dataset to roughly 60% needing correction / 40% unchanged, largely fixing this.
- **Safety net at inference:** if the LLM's output changed the word count or dropped a punctuation mark, its output was discarded and the raw Whisper text was kept instead — an effective guard against gross errors like accidental word deletion.

**Final training dataset:** [`raw-pairs-v10k-cleaned`](https://www.kaggle.com/datasets/amirsafarzadeh8203/raw-pairs-v10k-cleaned)
**Final model:** [`amirsz8203/gemma3-fa-whisper-refinement-lora-v13`](https://huggingface.co/amirsz8203/gemma3-fa-whisper-refinement-lora-v13)

## 4. Final Evaluation

To isolate each stage's real contribution, a controlled test was run on 100 fresh samples unseen by either the Whisper or LLM fine-tuning:

| Scenario | Whisper Output WER | WER After LLM Refinement | Δ |
|---|---|---|---|
| Neither fine-tuned | 108.33 | 107.61 | -0.72 |
| Whisper fine-tuned only | 26.87 | 28.74 | +1.87 |
| LLM fine-tuned only | 108.33 | 107.18 | -1.15 |
| Both fine-tuned | 26.87 | 28.30 | +1.44 |

**Interpretation:**
- Nearly all of the improvement comes from **Whisper fine-tuning**: WER drops from ~108 (a base model that's effectively unusable for Persian) to 26.87 — by far the project's biggest win.
- The **LLM refinement layer slightly increases numeric WER rather than reducing it**. This is consistent with the layer's actual goal (readability, not WER): WER scores strictly word-for-word against the reference, so a grammatically or typographically better sentence — one with, say, a punctuation mark the reference doesn't have — is scored as an "error" even when it reads better to a human.
- On the very poor baseline outputs (scenarios 1 and 3), LLM refinement did nudge WER down slightly — likely because on severely broken text, even shallow corrections have a small positive numeric effect.

## Summary

The project combined two complementary stages: progressive Whisper fine-tuning with a replay strategy for stable WER reduction (33% → 24.15% over five rounds), and a Gemma-3-based (LoRA fine-tuned) text refinement layer aimed at readability. The ASR fine-tuning was the dominant driver of quality gains. The LLM refinement layer, while qualitatively effective at fixing half-spaces, punctuation, and some lexical errors, did not improve the numeric WER metric — highlighting a real gap between standard numeric ASR metrics and qualitative readability, worth exploring further with complementary metrics such as human evaluation or semantic similarity scores.

## Tech Stack

- **Compute:** Kaggle notebooks (dual T4 GPUs)
- **Whisper fine-tuning:** Hugging Face `Seq2SeqTrainer`, end-to-end pipeline
- **LLM fine-tuning:** Gemma 3 4B via Unsloth + QLoRA
- **Dataset:** Common Voice Persian 26.0 (via Mozilla Data Collective)

## Notebooks

| Notebook | Purpose |
|---|---|
| `round-1-5-whisper-finetune.ipynb` | Fine-tuning Whisper in each round |
| `whisper-10000.ipynb` | Generating (raw Whisper output, reference text) pairs for LLM fine-tuning; also merges with earlier data into the final `raw-pairs-v10k` dataset |
| `clean-whisper-data-2.ipynb` | Cleaning, filtering, and balancing the pairs dataset into `raw-pairs-v10k-cleaned` |
| `gemma-3-4b-unsloth.ipynb` | LLM fine-tuning and dataset exploration/visualization |
| `inference-whisper-1234.ipynb` | Runs the 4-scenario comparison and exports results to Excel (`comparison_1_no_finetuning.xlsx` through `comparison_4_both_finetuned.xlsx`) |

## Links

- Dataset: [Common Voice Persian 26.0](https://mozilladatacollective.com/datasets/cmqinhw5100v8nr07gyg5gi4vf)
- LLM training pairs (raw): [`raw-pairs-v280`](https://www.kaggle.com/datasets/amirsafarzadeh8203/raw-pairs-v280) · [`raw-pairs-v10k`](https://www.kaggle.com/datasets/amirsafarzadeh8203/raw-pairs-v10k) · [`raw-pairs-v10k-cleaned`](https://www.kaggle.com/datasets/amirsafarzadeh8203/raw-pairs-v10k-cleaned)
- Whisper model: [`amirsz8203/whisper-small-fa-finetuned`](https://huggingface.co/amirsz8203/whisper-small-fa-finetuned)
- Gemma refinement model: [`amirsz8203/gemma3-fa-whisper-refinement-lora-v13`](https://huggingface.co/amirsz8203/gemma3-fa-whisper-refinement-lora-v13)
