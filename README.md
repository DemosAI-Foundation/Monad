# MONAD, Manifold Oriented Neural Agent Daemon

A conversational AI built on the **Free Energy Principle** (Friston 2010). The system maintains a 3D latent state updated via Kalman filtering, uses prediction error to drive all behavior, and treats an LLM (llama.cpp / Qwen) strictly as a **language rendering surface** — not the reasoning engine. All cognition happens in vector space.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│                    FastAPI + WebSocket Hub                       │
│            Serves index.html, routes user input                 │
└───────────┬────────────────────────────┬────────────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐   ┌─────────────────────────────────────┐
│      brain.py         │   │           index.html                │
│   BayesianBrain       │   │    Frontend: chat + trace panels    │
│  14-step pipeline     │   │    WebSocket consumer               │
│  Kalman filter        │   └─────────────────────────────────────┘
│  Idle dynamics        │
│  Memory orchestration │
└──┬──────┬──────┬──────┘
   │      │      │
   ▼      ▼      ▼
┌──────┐┌──────┐┌──────────────┐
│llm.py││emb.py││  memory.py   │
│      ││      ││  SQLite      │
│LLM   ││Vector││  persistence │
│client ││store ││              │
└──────┘└──────┘└──────────────┘
   │      │
   ▼      ▼
llama.cpp   ChromaDB + sentence-transformers
(Qwen)      (all-MiniLM-L6-v2)
```

## Core Design Principle

> The LLM is a tool, not the agent. It generates text when asked. The Bayesian brain decides **when** to ask, **what** to ask, **how constrained** the ask is, and **what to do** with the result.

Every LLM call receives 8 physics-derived sampling parameters computed from the system's latent state. There are no hardcoded sampling values anywhere.

## Files

| File | Lines | Role |
|---|---|---|
| `brain.py` | ~1,950 | **Core**: 14-step processing pipeline, Kalman filter, Langevin idle dynamics, episodic memory orchestration, self-model, allostatic regulation |
| `llm.py` | ~1,080 | **LLM interface**: streaming + EOS bias, 10 task profiles producing 8 sampling params, universal word budget tracking, perplexity measurement (3 strategies), HyDE validation, working memory compression |
| `embeddings.py` | ~1,030 | **Vector space**: dual-track retrieval (semantic + temporal), entropy-based memory decay, epistemic foraging, boredom-driven class weighting, VAD affective model, intent classification |
| `memory.py` | ~390 | **Persistence**: SQLite tables for conversation, episodic memories, latent state, predictions, beliefs, concept uncertainty |
| `diagnostic.py` | ~210 | **Diagnostics**: collects all trace events per cycle, assembles full-text report, LLM interprets cycle health |
| `main.py` | ~150 | **Server**: FastAPI app, WebSocket hub, startup lifecycle |
| `index.html` | ~1,010 | **Frontend**: chat interface, 9-tab diagnostic panel (beliefs, predictions, errors, silence, priors, vectors, trace, JSON, architecture), COPY ALL for traces |
| `train_action_gmm.py` | ~150 | **Training**: fits Gaussian Mixture Model over action embeddings, auto-selects k via BIC |
| `train_jepa_predictor.py` | ~230 | **Training**: 2-layer MLP for latent state rollouts (JEPA predictor: given V_t + action → predict V_{t+1}) |

## The 14-Step Processing Pipeline

Every user message passes through this pipeline in `brain.py → process_input()`:

```
USER INPUT
    │
    ▼
1.  ENCODE — sentence-transformers → 384d vector, VAD scoring, intent classification
1b. PERCEPTUAL ENRICHMENT — irony distance, subtext
1c. USER MODEL UPDATE — track user's vocabulary, valence baseline, intent patterns
1d. ENCODE EXPECTATION — LLM predicts TYPE of next user message (short label)
    │
    ▼
2.  PREDICTIVE ERROR — compare prediction (from previous cycle) with actual input
    • Simulation accuracy: cosine(predicted_vec, actual_vec)
    • Deliberation gain: did the extra LLM thinking improve prediction?
    │
    ▼
3.  AFFECTIVE BLEED — recalled memories modify latent state as Gaussian observations
3b. EPISODIC RECALL — 6-rule dynamics: reconsolidation, compression, decay, primacy
    │
    ▼
4.  AFFECTIVE PREDICTION ERROR — predicted valence/arousal vs observed
    │
    ▼
5.  LEXICAL ERROR — LLM perplexity on the input (3 strategies: logprobs, /completion, echo)
    │
    ▼
6.  ERROR COMPOSITION — hierarchical blend:
    • Primary: semantic error (system's own prediction) or lexical (perplexity)
    • Modulated by: pragmatic error (intent shift) + affective error (emotional surprise)
    │
    ▼
7.  PRECISION WEIGHTING — coherence × intent consistency × lexical precision
    • High precision → amplify prediction errors (trust the signal)
    • Low precision → attenuate (noisy signal, don't overreact)
    │
    ▼
8.  KALMAN UPDATE — 3D state update with full covariance:
    • P_pred = P + Q  (predict: uncertainty grows)
    • K = P_pred / (P_pred + R)  (gain: how much to trust observation)
    • x_new = x_old + K × (observation - x_old)  (update)
    • P_new = (1 - K) × P_pred  (posterior uncertainty shrinks)
    • Free energy = prediction_error + KL(posterior || prior)
    │
    ▼
9.  ALLOSTATIC REGULATION — phenotype prior drifts under chronic stress
9b. INTEROCEPTIVE STATE — energy from LLM call count + latency + overshoot
9c. LONG-TERM MEMORY ENCODING — LLM records what was said (recorder role)
    • Commitment binding: if input is low-perplexity + high causal impact,
      store user's text with a note about what they confirmed
    • HyDE validation: generate independent LLM knowledge, compare embedding
    │
    ▼
10. BUILD CONTEXT — working memory compression + long-term retrieval
    • WM: LLM summarizes conversation transcript (note-taker role)
    • LT: dual-track retrieval (semantic cosine + temporal recency injection)
    • Merge with deduplication
    │
    ▼
11. PRE-ACTION TRAJECTORY — predict user's next message before responding
    • Step 1: generate hypothetical assistant reply
    • Step 2: given that reply, predict what user would say next
    │
    ▼
12. OUTPUT — LLM generates response
    • System prompt: "You are an AI assistant." + topic
    • Context: [Conversation so far]\n...\n\n[Current message]\n...
    • 8 sampling params from _compute_manifold(latent, "output")
    • Streaming with sentence-boundary abort at budget
12a. Encode output as memory
12b. Post-action trajectory prediction (given what was actually said)
12c. Counterfactual depth: how far from conversation centroid?
12d. Self-model update: predicted action type vs actual
12e. Output quality: word budget accuracy, valence alignment
    │
    ▼
13. CAUSAL RELEVANCE SCORING — did retrieved memories help or hurt?
    • Compare current error with baseline → boost or degrade memory strength
    │
    ▼
14. CYCLE DIAGNOSTIC — full trace dump + LLM analysis of cycle health
```

## Latent State

Three dimensions, updated via Kalman filter every cycle:

| Dimension | Range | Meaning |
|---|---|---|
| **surprise** | [0, 1] | How unexpected the input was. Drives exploration, memory encoding strength, word budget. |
| **valence** | [-1, +1] | Affective tone. Modulates memory retrieval (negative → recall similar negative memories). |
| **velocity** | [0, 1] | Rate of conversational change. High → topic shifts, new information. Low → repetition, deepening. |

**Process noise** Q = [0.01, 0.005, 0.01] — uncertainty grows between observations.

**Measurement noise** R = f(precision) — high precision → trust observations more, low → attenuate.

## Physics-Driven Sampling (10 Task Profiles)

Every LLM call routes through `_compute_manifold(latent, task)` which produces 8 sampling parameters from the latent state:

| Parameter | Derivation |
|---|---|
| `temperature` | `t_base + t_scale × surprise` — high surprise → more creative output |
| `top_p` | `p_base + p_scale × (1 - surprise)` — low surprise → narrower sampling |
| `min_p` | `0.05 + velocity × 0.03` |
| `presence_penalty` | `0.2 + surprise × 0.2 - valence × 0.1` |
| `frequency_penalty` | `feedback.velocity_misalignment × 0.3` |
| `repetition_penalty` | `1.0 + surprise × 0.2 + velocity × 0.1` |
| `dynatemp_range` | `surprise × 0.3 + velocity × 0.1` |
| `max_tokens` | `budget_base + budget_scale × surprise` (word budget) |

### Task Profiles

| Profile | Budget Range | Use |
|---|---|---|
| `output` | 50–200 words | Main response generation |
| `hypothetical` | 50–200 words | Hypothetical assistant reply for prediction |
| `predict` | 15–30 words | User's next message prediction |
| `internal` | 20–60 words | Internal reasoning (subtext, reconciliation) |
| `creative` | 20–60 words | Dream synthesis |
| `compression` | 30–70 words | Memory / working memory compression |
| `summary` | 30–60 words | Topic summary |
| `diagnostic` | 80–140 words | Cycle diagnostic |
| `label` | 5–10 words | Short labels (expectation type) |
| `recorder` | 10–30 words | Memory encoding (factual recording) |

## Word Budget System

Every LLM call gets a word budget injected into its system prompt:

```
"[original system prompt] Limit: {N} words. Do not mention this limit."
```

After each call, the system measures actual vs budgeted words with continuous energy consequences:

- **Overshoot**: `energy_drain = overshoot² × 0.04` — quadratic, cumulative, capped at -0.3
- **Undershoot**: `penalty_delta = undershoot × 0.1 - 0.02` — linear, recovers when close to target
- **EOS bias**: `alpha = 0.3 + (1.0 - accuracy_ema) × 1.0` — chronic overshoot → stronger EOS pressure

Streaming output stops at the first sentence boundary (`.!?\n`) after exceeding the word budget. Hard abort at 2× budget.

## Memory Architecture

```
USER INPUT
    │
    ├──→ self.conversation (raw buffer, last 12 turns)
    │
    ├──→ Long-term encoding:
    │      LLM encodes as memory (recorder role) → ChromaDB vector store
    │      HyDE validation: LLM generates independent knowledge about topic
    │      Embedding similarity between hypothesis and memory → boost if aligned
    │
    ├──→ Long-term retrieval (dual-track):
    │      Track 1 — Semantic: cosine similarity to current input vector
    │      Track 2 — Temporal: 4 most recent memories injected regardless of similarity
    │      Score = semantic_sim × recency × encoding_strength
    │      Recency = exp(-turns_since × 0.3)  (exponential decay)
    │
    ├──→ Working memory compression:
    │      One LLM call (note-taker role) summarizes recent conversation
    │      Budget: 40–150 words (scales with surprise)
    │
    └──→ Context assembly (what the output LLM sees):
           WM summary + LT memories (deduplicated)
           Wrapped as: [Conversation so far]\n...\n\n[Current message]\n...
```

### Memory Dynamics

- **Encoding strength**: f(prediction_error, precision, arousal) — surprising, precise, arousing inputs create stronger memories
- **Entropy-based decay**: when a new memory is similar to an old one, the old one's strength decays: `Π_old -= similarity × encoding_strength × 0.3`
- **Reconsolidation**: recalled memories are re-encoded with modified latent state, creating valence drift over time
- **Commitment binding**: low-perplexity inputs (e.g., "continue", "yes") with high causal impact store the user's raw text plus a note about what was confirmed
- **Repetition suppression**: memories similar to recent encodings get suppressed (Grill-Spector et al. 2006)

## Idle Dynamics (Default Mode Network)

When no user input arrives, the system doesn't sleep — it runs Langevin dynamics on a potential landscape:

```
dx = -∇U(x)·dt + √(2T)·dW
```

Where:
- `U(x)` = potential energy from distance to phenotype prior + unresolved memory attractors
- `T` = free energy (uncertainty)
- `dW` = Brownian noise

This drives:
- **Covariance inflation**: `P = P + Q` each idle tick → free energy rises → hallucination threshold
- **Dream synthesis**: connect unresolved memories into novel associations
- **Memory compression**: old memories get terser over time (entropy-based)
- **Attractor dynamics**: high-tension memories pull the latent state, creating "thoughts"

## Neuroscience Grounding

| Component | Reference | Implementation |
|---|---|---|
| Predictive coding | Rao & Ballard 1999, Friston 2005 | Kalman filter on 3D latent state; hierarchy: lexical → semantic → pragmatic |
| Precision weighting | Feldman & Friston 2010 | Coherence × intent consistency × lexical precision modulates Kalman gain |
| Free Energy Principle | Friston 2010 | `FE = prediction_error + KL(posterior ∥ prior)`; idle covariance inflation |
| Memory consolidation | Diekelmann & Born 2010 | Dream synthesis + entropy-based compression during idle |
| Repetition suppression | Grill-Spector et al. 2006 | Encoding strength reduced for repeated content |
| Primacy effect | Murdock 1962 | First-cycle encodings get boosted strength |
| Allostatic regulation | Sterling 2012 | Phenotype prior drifts under chronic stress |

## LLM Prompts (All Calls)

Every prompt follows the pattern: **ROLE** (who you are) + **TASK** (what to do) + **BOUNDARY** (what NOT to do) + **BUDGET** (word limit, injected automatically).

| Call | System Prompt |
|---|---|
| `encode_memory` | "You are a memory recorder. Your ONLY job is to write a short factual note... Do NOT answer their question. Do NOT perform their request." |
| `compress_working_memory` | "You are a note-taker. Read the transcript... Do NOT continue the conversation. Do NOT add new content." |
| `output` | "You are an AI assistant." + optional topic |
| `predict (pre+post)` | "You are predicting what the USER will say next... Output ONLY what the user would say." |
| `hypothetical reply` | "You are an AI assistant. Respond to the user's latest message." |
| `encode_expectation` | "Read the conversation and predict the TYPE... Output ONLY the label." |
| `hyde_validate` | "Answer briefly from your own knowledge." |
| `compress_memory` | "Compress this memory. Preserve the core meaning." |
| `dream_synthesis` | "These are separate memories. Find what connects them. Write ONE sentence." |
| `diagnostic` | "You are a diagnostic module analyzing one processing cycle..." |

## Training Infrastructure

### JEPA Data Logging

Every cycle logs `(V_t, action_vec[:32], V_{t+1})` to `jepa_training_data.jsonl`. This enables training a predictor that can roll out future latent states without querying the LLM.

### `train_action_gmm.py`

Fits a Gaussian Mixture Model over action embeddings collected from system output. Auto-selects cluster count via BIC. Replaces hardcoded `ACTION_ANCHORS` with emergent topological structure.

### `train_jepa_predictor.py`

Trains a 2-layer MLP (64→32→3) to predict `V_{t+1}` from `(V_t, action_vec)`. Includes `predict()` for single-step and `rollout()` for multi-step trajectory simulation.

## Running

### Prerequisites

- Python 3.11+
- llama.cpp server running on port 8083 (tested with Qwen models)
- ChromaDB, sentence-transformers, aiohttp, fastapi, uvicorn

### Start

```bash
# Delete stale vector store on architecture changes
rm -rf chroma_db/

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser. The WebSocket connects automatically.

### Configuration

- LLM endpoint: `main.py` line 24 (`http://127.0.0.1:8083`)
- Embedding model: `embeddings.py` (`all-MiniLM-L6-v2`)
- ChromaDB path: `embeddings.py` (`./chroma_db`)
- Kalman process noise Q: `brain.py` (`[0.01, 0.005, 0.01]`)
- Base prior (genotype): `brain.py` (`surprise=0.25, valence=0.30, velocity=0.40`)

## Key Constants

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `BASE_PRIOR` | `{s:0.25, v:0.30, vel:0.40}` | brain.py | Phenotype genotype — where the system wants to be |
| Process noise Q | `[0.01, 0.005, 0.01]` | brain.py | Uncertainty growth rate per cycle |
| Exponential recency λ | 0.3 | embeddings.py | Memory recency decay: `exp(-turns × 0.3)` |
| Entropy decay factor | 0.3 | embeddings.py | Interference: `Π_old -= sim × enc × 0.3` |
| Temporal track injection | 4 | embeddings.py | Most recent memories always retrieved |
| Epistemic foraging threshold | trace(P) > 0.35 | brain.py | Covariance-driven exploration |
| WM retention floor | 0.25 | brain.py | Minimum working memory detail |
| EOS bias base alpha | 0.3 | llm.py | Scales to 1.3 with poor accuracy history |
| Energy recovery per cycle | +0.02 | llm.py | Metabolic recovery rate |

## Pending Work

- [ ] Wire trained GMM into `embeddings.py` to replace `ACTION_ANCHORS` (needs 50+ cycles of data)
- [ ] Wire trained JEPA predictor into `brain.py` to replace LLM counterfactuals (needs 50+ cycles)
- [ ] Zipfian semantic aphasia (frequency-ranked vocabulary penalties instead of fixed list)
- [ ] Remove `_state_conditioning()` from topic_summary and epistemic_reconciliation
