# MONAD, Manifold Oriented Neural Agent Daemon

A conversational system built on **Bayesian** and **Free Energy Principles**. The system maintains a 3D latent state updated via Kalman filtering, uses prediction error to drive all behavior, and treats an local LLM (llama.cpp / Qwen) strictly as a **language rendering module** — not the reasoning engine.

*Update: Now includes modular parts so you can switch around and input different data, like doing brain surgery live!*


## Interesting test results

Shows good promise in improving local AI agent systems through the prediction mechanism, even though data is intenionally degraded before the LLM input. System shows coherence and output shows the system quasi inferring it's design upon multiturn usage, without pretrained data.


## ToDO

* Improve further vector space prediction modules
* Add more modularity of data for testing purposes
* Improve on multiturn diagnostics


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


## Running

### Prerequisites

- Python 3.11+
- llama.cpp server running on port 8083 (tested with Qwen Coder Next model)
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

