"""
brain.py — Bayesian Brain: predictive coding conversational agent.

Core processing engine. Implements a 14-step pipeline per user message:
  1. Encode (vector + affect + intent)
  2. Predictive error (compare prediction with actual input)
  3. Affective bleed + episodic recall (memory → latent state)
  4. Affective prediction error (valence/arousal mismatch)
  5. Lexical error (LLM perplexity on input)
  6. Error composition (hierarchical blend: semantic > lexical > cosine)
  7. Precision weighting (coherence × intent consistency × lexical precision)
  8. Kalman update (3D latent state with diagonal covariance)
  9. Allostatic regulation + memory encoding
  10. Context assembly (working memory compression + long-term retrieval)
  11. Pre-action trajectory prediction
  12. Output generation + post-action analysis
  13. Causal relevance scoring (did memories help or hurt?)
  14. Cycle diagnostic (full trace dump + LLM interpretation)

The LLM is a tool — all cognition happens in vector space via Kalman filtering.
Latent state: (surprise ∈ [0,1], valence ∈ [-1,+1], velocity ∈ [0,1]).
"""

import asyncio
import logging
import json
import math
import random
import urllib.parse
import aiohttp
from collections import deque, defaultdict, Counter
from datetime import datetime
from typing import Optional, Callable, Any
from embeddings import compute_encoding_strength, compute_memory_impact
from diagnostic import CycleDiagnostic

logger = logging.getLogger(__name__)

IDLE_PHASES = [
    {"threshold": 0,   "label": "IDLE",      "color": "#3a3e55", "fe_drift": 0.000},
    {"threshold": 10,  "label": "DRIFT",     "color": "#8b8fa8", "fe_drift": 0.002},
    {"threshold": 25,  "label": "RESTLESS",  "color": "#f5a623", "fe_drift": 0.005},
    {"threshold": 45,  "label": "THRESHOLD", "color": "#ff3b5c", "fe_drift": 0.010},
]

ACTION_THRESHOLD_S = 60
ACTION_REPEAT_S    = 30
VELOCITY_DRIFT     = 0.015
MAX_IDLE_FE        = 2.5

# THE GENOTYPE
BASE_PRIOR = {"surprise": 0.25, "valence": 0.30, "velocity": 0.40}

# ── Pipeline Configuration ──
# Defines which data sources feed into each pipeline function.
# Modifiable at runtime via the PIPELINE tab in the UI.
# Available sources:
#   raw_conversation — actual chat turns from self.conversation
#   encoded_inputs  — LLM-recorded user memory notes from ChromaDB
#   encoded_outputs — LLM-recorded assistant memory notes from ChromaDB
#   lt_memories     — semantically + temporally retrieved long-term memories
#   working_memory  — compressed WM text (output of WM compression stage)
#   current_message — the current user input text
DEFAULT_PIPELINE_CONFIG = {
    "working_memory":      {"sources": ["encoded_inputs", "encoded_outputs"], "state_conditioning": False},
    "output":              {"sources": ["working_memory", "lt_memories", "current_message"], "state_conditioning": True},
    "hypothetical":        {"sources": ["working_memory", "current_message"], "state_conditioning": True},
    "predict_pre":         {"sources": ["working_memory"], "state_conditioning": True},
    "predict_post":        {"sources": ["working_memory"], "state_conditioning": True},
    "encode_expectation":  {"sources": ["raw_conversation"], "state_conditioning": True},
    "encode_memory":       {"sources": ["current_message"], "state_conditioning": False},
}

PIPELINE_SOURCES = [
    {"id": "raw_conversation", "label": "Raw Conversation", "desc": "Actual chat turns"},
    {"id": "encoded_inputs",   "label": "Encoded Inputs",   "desc": "LLM-recorded user memories"},
    {"id": "encoded_outputs",  "label": "Encoded Outputs",  "desc": "LLM-recorded assistant memories"},
    {"id": "lt_memories",      "label": "LT Memories",      "desc": "Retrieved long-term memories"},
    {"id": "working_memory",   "label": "Working Memory",   "desc": "Compressed WM summary"},
    {"id": "current_message",  "label": "Current Message",  "desc": "Current user input"},
]

# ── Pure Math Helpers ──

def ts() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def initial_latent() -> dict:
    """Starting latent state: moderate surprise, neutral valence, moderate velocity."""
    return {"surprise": 0.30, "valence": 0.00, "velocity": 0.30}

def _state_conditioning_simple(latent: dict) -> str:
    """Minimal state conditioning for internal brain.py modules."""
    s, v, vel = latent.get("surprise", 0.3), latent.get("valence", 0.0), latent.get("velocity", 0.3)
    return f"--- STATE CONDITIONING ---\nSurprise {s:.2f} / range [0,1]\nValence {v:+.2f} / range [-1,+1]\nVelocity {vel:.2f} / range [0,1]"

def update_latent(old: dict, obs_surprise: float, obs_valence: float, obs_velocity: float, 
                  precision: float = 1.0, P: list = None, Q: list = None) -> tuple[dict, list, float, dict]:
    """Proper Kalman filter with diagonal covariance matrix.
    
    P = [P_s, P_v, P_vel] — state uncertainty (diagonal covariance)
    Q = [Q_s, Q_v, Q_vel] — process noise
    
    Returns: (new_latent, new_P, effective_gain_mean, gain_breakdown)
    """
    if P is None: P = [0.5, 0.5, 0.5]
    if Q is None: Q = [0.01, 0.005, 0.01]
    
    # Predict step: uncertainty grows by process noise
    P_pred = [P[i] + Q[i] for i in range(3)]
    
    # Measurement noise: inversely proportional to precision
    R = max(0.05, 1.0 / max(precision, 0.1))
    
    # Per-dimension Kalman gains
    K = [round(P_pred[i] / (P_pred[i] + R), 4) for i in range(3)]
    
    obs = [obs_surprise, obs_valence, obs_velocity]
    old_vals = [old["surprise"], old["valence"], old["velocity"]]
    keys = ["surprise", "valence", "velocity"]
    
    new_latent = {}
    new_P = []
    for i, k in enumerate(keys):
        new_latent[k] = round(old_vals[i] + K[i] * (obs[i] - old_vals[i]), 4)
        new_P.append(round((1 - K[i]) * P_pred[i], 4))
    
    effective_gain = round(sum(K) / 3, 4)
    
    return new_latent, new_P, effective_gain, {
        "gains": {"surprise": K[0], "valence": K[1], "velocity": K[2]},
        "P_prior": [round(p, 4) for p in P_pred],
        "P_posterior": new_P,
        "measurement_noise": round(R, 4),
    }

def compute_free_energy(current_error: float, latent: dict, phenotype_prior: dict, P: list, beta: float = 0.3) -> float:
    """Proper variational free energy: accuracy + complexity.
    Accuracy: prediction error (how well the model predicts observations).
    Complexity: Gaussian KL from phenotype prior (how far the system has drifted).
    P modulates complexity — uncertain states can drift cheaply."""
    accuracy_cost = current_error
    keys = ["surprise", "valence", "velocity"]
    complexity_cost = sum(
        (latent.get(k, 0) - phenotype_prior.get(k, 0)) ** 2 / (2 * max(P[i], 0.01))
        for i, k in enumerate(keys)
    )
    return round(accuracy_cost + beta * complexity_cost, 4)

def compute_precision(msg_vec: list, recent_vecs: list, observed_intent: str,
                      intent_history: list, logprob_variance: float = None) -> tuple[float, dict]:
    """Estimate precision (inverse variance) of the current error signal.
    
    Components:
    - Context coherence: is this message in the same region as recent messages? 
      High coherence + high surprise = genuine signal (boost precision).
      Low coherence + high surprise = possible noise (lower precision).
    - Intent consistency: does the classified intent match the recent pattern?
      Consistent intent = the user is refining within a mode (high precision).
      Inconsistent intent = mode switch, lower confidence in error meaning.
    - Lexical confidence: variance of token-level logprobs.
      Low variance = model is uniformly (un)sure (high precision).
      High variance = mixed confidence, some tokens very surprising (lower precision).
    """
    # Context coherence: cosine to recent centroid
    coherence = 0.5
    if recent_vecs and len(recent_vecs) >= 2:
        centroid = [sum(v[i] for v in recent_vecs) / len(recent_vecs) for i in range(len(recent_vecs[0]))]
        coherence = max(0.0, _cosim(msg_vec, centroid))

    # Intent consistency: fraction of recent intents that match
    intent_match = 0.5
    if intent_history:
        intent_match = sum(1 for i in intent_history if i == observed_intent) / len(intent_history)

    # Lexical precision: low logprob variance = high confidence
    lp_precision = 0.5
    if logprob_variance is not None:
        # Normalize: variance of ~0 → precision 1.0, variance > 3 → precision ~0.1
        lp_precision = max(0.1, 1.0 - min(1.0, logprob_variance / 3.0))

    # Weighted combination → raw precision [0, 1]
    raw = coherence * 0.40 + intent_match * 0.30 + lp_precision * 0.30
    # Map to [0.3, 1.5]: center ~0.9 for "normal" signals
    precision = round(0.3 + raw * 1.2, 4)

    breakdown = {
        "coherence": round(coherence, 3),
        "intent_consistency": round(intent_match, 3),
        "lexical_precision": round(lp_precision, 3),
        "raw": round(raw, 3),
        "final_precision": precision,
    }
    return precision, breakdown

def idle_drift(latent: dict, phenotype_prior: dict = None, 
               free_energy: float = 0.5, unresolved_attractors: list = None) -> dict:
    """Langevin dynamics on a potential energy landscape.
    
    Instead of if/else attractor classification, the latent state evolves via:
        dx = -∇U(x)·dt + √(2T)·dW
    
    where:
        U(x) = potential energy (sum of attractor wells)
        T = temperature (from free energy — high FE = high noise = restless)
        dW = Wiener process (Gaussian noise)
    
    Attractor wells:
        1. Phenotype prior (home basin) — always present, strength ∝ 1/distance²
        2. Unresolved memories — each pulls with strength ∝ tension × 1/distance
    
    The state naturally "rolls" toward the strongest nearby well,
    with random perturbation proportional to free energy.
    """
    import random
    
    if phenotype_prior is None:
        phenotype_prior = {"surprise": 0.25, "valence": 0.30, "velocity": 0.40}
    unresolved = unresolved_attractors or []
    
    s, v, vel = latent["surprise"], latent["valence"], latent["velocity"]
    state = [s, v, vel]
    keys = ["surprise", "valence", "velocity"]
    prior = [phenotype_prior.get(k, 0.3) for k in keys]
    
    dt = 0.02  # time step per idle tick
    
    # ── Compute gradient of potential energy ──
    gradient = [0.0, 0.0, 0.0]
    
    # Well 1: Phenotype prior (home basin)
    # Pull strength: κ × (x - x_prior), κ = 0.5 (moderate pull home)
    kappa_home = 0.5
    for i in range(3):
        gradient[i] += kappa_home * (state[i] - prior[i])
    
    # Well 2+: Unresolved memory attractors
    # Each attractor has a position in latent space and a tension value
    for attr in unresolved:
        attr_pos = [attr.get("surprise", 0.3), attr.get("valence", 0.0), attr.get("velocity", 0.3)]
        tension = attr.get("tension", 0.5)
        # Pull toward attractor, strength ∝ tension
        kappa_mem = tension * 0.3
        for i in range(3):
            gradient[i] += kappa_mem * (state[i] - attr_pos[i])
    
    # ── Temperature from free energy ──
    # High FE = restless = more Brownian noise
    temperature = max(0.001, free_energy * 0.1)
    noise_scale = (2.0 * temperature) ** 0.5
    
    # ── Langevin update: dx = -∇U·dt + √(2T)·dW ──
    new_state = {}
    for i, k in enumerate(keys):
        drift = -gradient[i] * dt
        noise = noise_scale * random.gauss(0, 1) * (dt ** 0.5)
        new_val = state[i] + drift + noise
        # Clamp to valid ranges
        if k == "surprise":
            new_val = max(0.05, min(0.95, new_val))
        elif k == "valence":
            new_val = max(-0.8, min(0.8, new_val))
        elif k == "velocity":
            new_val = max(0.0, min(0.9, new_val))
        new_state[k] = round(new_val, 4)
    
    return new_state

def classify_terminal_state(latent: dict) -> str:
    """DEPRECATED: returns label for backward compat but no longer drives dynamics.
    The Langevin dynamics handle all idle state evolution."""
    s, v, vel = latent["surprise"], latent["valence"], latent["velocity"]
    if abs(v) > 0.35:      return "rumination"
    if vel > 0.50:          return "refractory"
    if s > 0.55:            return "consolidation"
    return "default_mode"

def efe_distance(state: dict, preferred: dict) -> float:
    """Euclidean distance in latent space between current state and preferred state."""
    return round(sum((state.get(k, 0) - preferred.get(k, 0)) ** 2 for k in preferred) ** 0.5, 4)

def _cosim(a: list, b: list) -> float:
    """Cosine similarity for causal impact measurement."""
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))

def idle_phase(seconds: int) -> dict:
    """Return the current idle phase based on elapsed seconds since last user input."""
    phase = IDLE_PHASES[0]
    for p in IDLE_PHASES:
        if seconds >= p["threshold"]: phase = p
    return phase

# ── Main Class ──

class BayesianBrain:
    """Predictive coding conversational agent.
    
    Maintains a 3D latent state (surprise, valence, velocity) updated via
    Kalman filtering. The LLM is a language rendering surface — all cognition
    happens in vector space.
    
    Key state:
      latent: current 3D state (surprise ∈ [0,1], valence ∈ [-1,+1], velocity ∈ [0,1])
      _P: diagonal Kalman covariance [P_surprise, P_valence, P_velocity]
      phenotype_prior: adaptive baseline — drifts under chronic stress (allostatic load)
      free_energy_val: KL(posterior ∥ prior) + prediction error
      conversation: raw message buffer for perplexity + context assembly
    
    Idle behavior: Langevin dynamics on potential landscape with memory attractors,
    covariance inflation (P = P + Q), dream synthesis, memory compression.
    """
    PROCESS_NOISE = [0.01, 0.005, 0.01]  # Q for Kalman: uncertainty growth per cycle
    N_PROTOTYPES = 8      # self-model prototype count
    PROTOTYPE_SIGMA = 0.3 # prototype activation width

    def __init__(self):
        self.latent: dict = initial_latent()
        self.latent_prev: dict = initial_latent()
        self.phenotype_prior: dict = dict(BASE_PRIOR)
        self.dynamic_axes: dict = {}
        self.conversation: list = []      # raw messages (for perplexity only)
        self.messages: list = []
        self.current_prediction: str  = "(awaiting first input…)"
        self.prediction_texts: list = []
        self.last_enriched: Optional[dict] = None
        self.free_energy_val: float = 0.0
        self._error_history: deque = deque(maxlen=8)
        self.topic_summary: str = ""
        self._summary_counter: int = 0
        self.idle_seconds: int = 0
        self.idle_fe: float = 0.0
        self.action_count: int = 0
        self.is_processing: bool = False
        self._idle_task: Optional[asyncio.Task] = None
        self._msg_counter: int = 0
        # ── Kalman Covariance (Point 4) ──
        self._P: list = [0.5, 0.5, 0.5]  # diagonal covariance [surprise, valence, velocity]
        # ── True Predictive Error ──
        self._predicted_next_vec: Optional[list] = None
        self._predicted_next_text: str = ""
        self._predicted_hyp_vec: Optional[list] = None
        self._predicted_hyp_text: str = ""
        # ── Simulation Accuracy Tracking ──
        self._sim_accuracy_deliberated: deque = deque(maxlen=20)
        self._sim_accuracy_hypothetical: deque = deque(maxlen=20)
        # ── Affective Prediction ──
        self._valence_ema: float = 0.0
        self._arousal_ema: float = 0.5
        self._affective_alpha: float = 0.3
        # ── Precision-Weighting ──
        self._intent_history: deque = deque(maxlen=6)
        self._recent_msg_vecs: deque = deque(maxlen=6)
        # ── Information Density (drives boredom + adaptive thresholds) ──
        self._suppression_history: deque = deque(maxlen=6)  # recent suppression factors [0-1]
        self._input_lengths: deque = deque(maxlen=12)  # word counts of user messages
        self._user_msg_lengths: deque = deque(maxlen=12)
        self._user_word_ema: float = 10.0  # default: assume moderate-length messages
        self._last_precision: float = 1.0
        # ── Interoceptive State ──
        self._last_interoceptive: dict = {"energy": 1.0, "cycle_calls": 0}
        # ── Hierarchical Error Tracking ──
        self._error_lexical: deque = deque(maxlen=10)
        self._error_semantic: deque = deque(maxlen=10)
        self._error_pragmatic: deque = deque(maxlen=10)
        # ── Content-Dependent Dynamics ──
        self._terminal_state: str = "default_mode"  # backward compat label
        self._terminal_valence: float = 0.0
        self._terminal_arousal: float = 0.5
        self._terminal_surprise: float = 0.3
        self._idle_attractors: list = []  # Langevin attractor wells from unresolved memories
        # ── Counterfactual Depth ──
        self._explore_exploit_ratio: float = 0.5
        self._last_causal_impact: float = 0.0  # from previous cycle
        self._conversation_centroid: Optional[list] = None
        # ── Attentional Inertia ──
        self._attention_anchor_vec: Optional[list] = None  # embedding of last substantive memory
        # ── Self-Model: Prototype-Based (Point 9) ──
        self._prototypes: list = self._init_prototypes()  # N_PROTOTYPES x 3D latent vectors
        self._proto_action_counts: list = [defaultdict(int) for _ in range(self.N_PROTOTYPES)]
        self._self_model_error: float = 0.0
        self._last_predicted_action: str = ""
        self._last_actual_action: str = ""
        # ── Output Quality Measurement (Point 7) ──
        self._output_valence_alignment: float = 0.0  # EMA: |reply_valence - target_valence|
        self._output_velocity_alignment: float = 0.0  # EMA: |actual_tokens - budgeted_tokens| / budget
        self._output_context_usage: float = 0.0  # EMA: cosine(reply, injected_context)
        # ── Closed-Loop Feedback EMAs (Point 10) ──
        self._deliberation_gain_ema: float = 0.0
        self._sim_accuracy_ema: float = 0.5
        # ── Diagnostic ──
        self._diag: CycleDiagnostic = CycleDiagnostic()
        self._diag_cycle: int = 0
        # ── Pipeline Configuration (modifiable via UI) ──
        self._pipeline_config: dict = json.loads(json.dumps(DEFAULT_PIPELINE_CONFIG))
        self._cycle_data: dict = {}

    def _init_prototypes(self) -> list:
        """Initialize prototypes evenly distributed across latent space."""
        protos = []
        for i in range(self.N_PROTOTYPES):
            s = (i % 4) / 3.0
            v = -0.5 + (i % 3) * 0.5
            vel = 0.2 + (i % 2) * 0.5
            protos.append([round(s, 2), round(v, 2), round(vel, 2)])
        return protos

    # ── Pipeline Configuration ──

    @property
    def pipeline_config(self) -> dict:
        """Current pipeline wiring config. Serializable to JSON for UI."""
        return self._pipeline_config

    def update_pipeline_config(self, new_config: dict):
        """Update pipeline config from UI. Empty config resets to defaults."""
        if not new_config:
            self._pipeline_config = json.loads(json.dumps(DEFAULT_PIPELINE_CONFIG))
            return
        for key in DEFAULT_PIPELINE_CONFIG:
            if key in new_config:
                self._pipeline_config[key] = new_config[key]

    def _get_stage_config(self, stage: str) -> dict:
        return self._pipeline_config.get(stage, DEFAULT_PIPELINE_CONFIG.get(stage, {"sources": [], "state_conditioning": False}))

    def resolve_context(self, stage: str) -> list:
        """Resolve data sources for a pipeline stage into [{role, content}] messages.
        
        Items with _order metadata (from encoded_inputs/outputs) are merged
        and sorted chronologically, producing interleaved User/Assistant turns.
        Items without _order keep their relative position after ordered items.
        """
        sources = self._get_stage_config(stage).get("sources", [])
        ordered = []   # items with _order → will be sorted
        unordered = [] # items without → appended after
        
        for src in sources:
            data = self._cycle_data.get(src)
            if data is None:
                continue
            if isinstance(data, str):
                if data.strip():
                    unordered.append({"role": "user", "content": data})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("content", "").strip():
                        if "_order" in item:
                            ordered.append(item)
                        else:
                            unordered.append(item)
                    elif isinstance(item, str) and item.strip():
                        unordered.append({"role": "user", "content": item})
        
        # Sort ordered items chronologically then merge
        ordered.sort(key=lambda x: x.get("_order", 0))
        
        # Strip _order metadata before returning
        result = []
        for item in ordered + unordered:
            result.append({"role": item.get("role", "user"), "content": item["content"]})
        return result

    def stage_uses_state(self, stage: str) -> bool:
        """Check if a pipeline stage should include latent state conditioning."""
        return self._get_stage_config(stage).get("state_conditioning", False)

    async def _get_recent_encoded_memories(self, vectors, n: int = 10, mode: str = "both") -> list:
        """Get recent encoded memories from ChromaDB by type. Returns [{role, content}].
        mode: 'input', 'output', or 'both'. Turn count scales with surprise."""
        if not vectors or not hasattr(vectors, '_episodic_col') or not vectors._episodic_col:
            return []
        try:
            count = vectors._episodic_col.count()
            if count == 0:
                return []
            results = vectors._episodic_col.get(
                include=["documents", "metadatas"], limit=min(n * 2, count))
            if not results or not results.get("documents"):
                return []
            entries = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                mem_type = meta.get("memory_type", "input")
                if mode == "input" and mem_type != "input":
                    continue
                if mode == "output" and mem_type != "output":
                    continue
                entries.append({
                    "role": "user" if mem_type == "input" else "assistant",
                    "content": doc,
                    "msg_counter": meta.get("msg_counter", 0),
                })
            entries.sort(key=lambda e: e.get("msg_counter", 0))
            s = self.latent.get("surprise", 0.3)
            max_turns = max(3, min(n, int(4 + s * 8)))
            entries = entries[-max_turns:]
            return [{"role": e["role"], "content": e["content"], "_order": e.get("msg_counter", 0)} for e in entries]
        except Exception as e:
            logger.debug(f"Encoded memory retrieval failed: {e}")
            return []

    # ── Active Inference: Policy Selection via Expected Free Energy ──

    def evaluate_policies(self, energy: float, current_error: float = 0.5) -> dict:
        """Score candidate policies via Expected Free Energy (Friston 2017).
        
        G(π) = pragmatic_value + epistemic_value + energy_cost
        
        Pragmatic: how far does this policy move us from preferred state?
        Epistemic: how much will this policy reduce uncertainty (trace(P))?
        Energy: metabolic cost of executing the policy.
        
        Returns: {G: {policy: score}, probs: {policy: prob}, selected: str}
        No LLM calls. Pure math from existing state variables.
        """
        trace_P = sum(self._P)
        efe_dist = efe_distance(self.latent, self.phenotype_prior)
        n_attractors = len(getattr(self, '_idle_attractors', []))
        fe = self.free_energy_val
        
        # ── G(respond): full pipeline → output to user ──
        # High pragmatic value (moves toward preferred state via engagement)
        # High epistemic value (full cycle with observation → P shrinks)  
        # High energy cost (8-10 LLM calls)
        G_respond = (
            -efe_dist * 0.8             # pragmatic: responding helps reach preferred state
            - trace_P * 0.6             # epistemic: full cycle resolves ~60% uncertainty
            + (1.0 - energy) * 0.3      # cost: expensive when tired
            + current_error * 0.1       # slight penalty: high error = maybe not ready to respond
        )
        
        # ── G(think): run hypothetical, feed back as observation, no output ──
        # Low pragmatic value (no direct user satisfaction)
        # Medium-high epistemic value (imagined observation → partial P reduction)
        # Low energy cost (0 extra LLM calls — reuses existing hypothetical)
        G_think = (
            -efe_dist * 0.2             # pragmatic: thinking indirectly improves future responses
            - trace_P * 0.4             # epistemic: imagination resolves ~40% uncertainty
            - n_attractors * 0.08       # bonus: more unresolved tensions → thinking helps more
            + (1.0 - energy) * 0.03     # cost: very cheap (no new LLM calls)
        )
        
        # ── G(recall): forage episodic memory, reconsolidate ──
        # No pragmatic value (no output)
        # High epistemic value when many unresolved attractors
        # Very low energy cost
        G_recall = (
            0.0                                                      # pragmatic: zero
            - trace_P * 0.25 * n_attractors / max(n_attractors, 1)   # epistemic: scales with unresolved count
            + 0.01                                                   # cost: near-zero
        )
        
        # ── G(silence): do nothing, let P inflate naturally ──
        # Pragmatic value depends on distance: near preferred = silence is fine
        # Zero epistemic value
        # Negative cost (energy recovery)
        G_silence = (
            efe_dist * 0.6              # pragmatic: BAD when far from preferred (need to act)
            - 0.0                       # epistemic: zero information gain
            - 0.02                      # benefit: energy recovery
        )
        
        G = {"respond": round(G_respond, 4), "think": round(G_think, 4), 
             "recall": round(G_recall, 4), "silence": round(G_silence, 4)}
        
        # Softmax selection: P(π) ∝ exp(-β × G(π))
        # Precision β scales with confidence — high precision = more decisive
        beta = max(0.5, getattr(self, '_last_precision', 1.0) * 2.0)
        
        min_G = min(G.values())
        weights = {k: math.exp(-beta * (v - min_G)) for k, v in G.items()}
        total = sum(weights.values())
        probs = {k: round(w / total, 4) for k, w in weights.items()}
        
        selected = min(G, key=G.get)
        
        return {
            "G": G, "probs": probs, "selected": selected,
            "trace_P": round(trace_P, 4), "efe_dist": round(efe_dist, 4),
            "n_attractors": n_attractors, "energy": round(energy, 4),
            "beta": round(beta, 4),
        }

    async def _execute_think(self, hyp_text: str, hyp_vec: list, 
                              latent: dict, vectors, broadcast) -> dict:
        """Execute the 'think' policy: feed hypothetical reply as imagined observation.
        
        No new LLM calls. Takes the already-generated hypothetical from step 11,
        measures its affective properties, and creates a low-precision Kalman 
        observation. Uncertainty shrinks because the system processed information 
        internally, even though nothing was communicated to the user.
        
        Returns: updated (latent, P, free_energy) after internal Kalman update.
        """
        if not hyp_text or not hyp_vec or not vectors:
            return {"latent": latent, "P": self._P, "fe": self.free_energy_val}
        
        # Measure affective properties of the imagined response
        thought_valence, thought_arousal, _ = await vectors._score_vad_text(hyp_text, hyp_vec)
        
        # Compute velocity WITHOUT mutating the embed_window
        # (imagined content should not pollute real velocity tracking)
        window = list(vectors._embed_window)
        if window:
            last_real = window[-1]
            thought_velocity = min(1.0, (1.0 - sum(a*b for a,b in zip(hyp_vec, last_real))) / 0.45)
        else:
            thought_velocity = 0.3
        
        # The imagined observation:
        # - Low surprise (we generated it — it's consistent with our model)
        # - Valence/velocity from the hypothetical content
        obs_surprise = 0.1  # imagined content is not surprising
        obs_valence = thought_valence
        obs_velocity = thought_velocity
        
        # Lower precision for imagined observations (less trustworthy than real input)
        # Real observations use precision ~1.0-1.25, imagined use ~0.5
        think_precision = 0.5
        
        # Kalman update with imagined observation
        new_latent, new_P, eff_gain, breakdown = update_latent(
            latent, obs_surprise, obs_valence, obs_velocity,
            precision=think_precision, P=list(self._P), Q=[0.0, 0.0, 0.0]  # no process noise for internal step
        )
        
        # Update system state
        self._P = new_P
        new_fe = compute_free_energy(
            self.rolling_error, new_latent, self.phenotype_prior, self._P)
        
        await broadcast({
            "type": "system_trace", "label": "think_action", "duration_ms": 0,
            "summary": f"Internal thought: P [{self._P[0]:.3f},{self._P[1]:.3f},{self._P[2]:.3f}] → FE {new_fe:.3f}",
            "details": {
                "thought_valence": round(thought_valence, 3),
                "thought_velocity": round(thought_velocity, 3),
                "observation": {"surprise": obs_surprise, "valence": obs_valence, "velocity": obs_velocity},
                "precision": think_precision,
                "P_before": [round(p, 4) for p in self._P],
                "P_after": new_P,
                "gain": breakdown,
                "thought_preview": hyp_text[:100],
            }
        })
        
        # Log JEPA data: (V_t, action="think", V_{t+1})
        try:
            import json as _json
            with open("jepa_training_data.jsonl", "a") as f:
                f.write(_json.dumps({
                    "V_t": [latent["surprise"], latent["valence"], latent["velocity"]],
                    "action": "think",
                    "action_vec": hyp_vec[:32] if hyp_vec else [],
                    "V_t1": [new_latent["surprise"], new_latent["valence"], new_latent["velocity"]],
                }) + "\n")
        except Exception:
            pass
        
        return {"latent": new_latent, "P": new_P, "fe": new_fe}

    async def _execute_recall(self, vectors, memory, broadcast) -> bool:
        """Execute the 'recall' policy: forage episodic memory mid-pipeline.
        Reuses existing _unresolved_attractor logic. Returns True if memories were found."""
        if not vectors:
            return False
        try:
            await self._unresolved_attractor(vectors, memory, broadcast)
            return len(getattr(self, '_idle_attractors', [])) > 0
        except Exception:
            return False

    @property
    def rolling_error(self) -> float:
        """Exponentially-weighted recent prediction error. Recency-biased: recent errors matter more."""
        if not self._error_history: return 0.5
        weights = [0.5 ** i for i in range(len(self._error_history))]
        return round(sum(e * w for e, w in zip(list(reversed(self._error_history)), weights)) / sum(weights), 4)

    def _latent_label(self) -> str:
        s, v, vel = self.latent["surprise"], self.latent["valence"], self.latent["velocity"]
        if s > 0.65 and v < -0.30: return "destabilised"
        if s > 0.65: return "surprised"
        if s < 0.20 and vel < 0.20: return "stagnant"
        if v > 0.45 and s < 0.35: return "flow"
        if v < -0.35: return "troubled"
        if vel > 0.65: return "fast-moving"
        return "engaged"

    def state_summary(self) -> str:
        """One-line human-readable state label + numeric values."""
        s, v, vel = self.latent["surprise"], self.latent["valence"], self.latent["velocity"]
        return f"{self._latent_label()} — surprise={s:.2f} valence={v:+.2f} velocity={vel:.2f}"

    def state_snapshot(self) -> dict:
        """Full state dump for frontend WebSocket broadcast. Includes everything the UI needs."""
        phase = idle_phase(self.idle_seconds)
        next_in = max(0, ACTION_THRESHOLD_S - self.idle_seconds) if self.action_count == 0 else max(0, ACTION_REPEAT_S - self.idle_seconds)
        enriched_summary = {}
        if self.last_enriched:
            enriched_summary = {
                "tone": self.last_enriched.get("tone", {}), "intent": self.last_enriched.get("meaning", {}).get("intent", ""),
                "concepts": [c.get("label","") for c in self.last_enriched.get("concepts",[]) if isinstance(c,dict)][:4],
                "retrieval_triggered": self.last_enriched.get("retrieval_triggered", False), "error_mode": self.last_enriched.get("prediction_error", {}).get("error_mode", "cosine"),
            }
        return {
            "latent": self.latent, "latent_prev": self.latent_prev, "free_energy": round(self.free_energy_val, 4),
            "idle_fe": round(self.idle_fe, 4), "idle_seconds": self.idle_seconds, "idle_phase": phase,
            "action_count": self.action_count, "next_action_in": next_in, "current_prediction": self.current_prediction,
            "rolling_error": self.rolling_error, "latent_label": self._latent_label(),
            "efe_distance": round(efe_distance(self.latent, self.phenotype_prior), 4), "is_processing": self.is_processing,
            "enriched": enriched_summary, "topic_summary": self.topic_summary,
            "affective_ema": {"valence": round(self._valence_ema, 3), "arousal": round(self._arousal_ema, 3)},
            "has_prediction_prior": self._predicted_next_vec is not None,
            "has_hypothetical_prior": self._predicted_hyp_vec is not None,
            "interoceptive": self._last_interoceptive,
            "covariance": [round(p, 4) for p in self._P],
            "error_hierarchy": {
                "lexical": round(sum(self._error_lexical) / max(len(self._error_lexical), 1), 3) if self._error_lexical else None,
                "semantic": round(sum(self._error_semantic) / max(len(self._error_semantic), 1), 3) if self._error_semantic else None,
                "pragmatic": round(sum(self._error_pragmatic) / max(len(self._error_pragmatic), 1), 3) if self._error_pragmatic else None,
            },
            "terminal_state": self._terminal_state,
            "explore_exploit_ratio": round(self._explore_exploit_ratio, 3),
            "self_model": {"error_ema": round(self._self_model_error, 3), "last_predicted": self._last_predicted_action, "last_actual": self._last_actual_action},
            "simulation_accuracy": {
                "deliberated_mean": round(sum(self._sim_accuracy_deliberated) / len(self._sim_accuracy_deliberated), 3) if self._sim_accuracy_deliberated else None,
                "hypothetical_mean": round(sum(self._sim_accuracy_hypothetical) / len(self._sim_accuracy_hypothetical), 3) if self._sim_accuracy_hypothetical else None,
                "n_samples": len(self._sim_accuracy_deliberated),
            },
            "output_quality": {
                "valence_alignment": round(self._output_valence_alignment, 3),
                "velocity_alignment": round(self._output_velocity_alignment, 3),
                "context_usage": round(self._output_context_usage, 3),
            },
            "feedback_loops": {
                "deliberation_gain_ema": round(self._deliberation_gain_ema, 3),
                "sim_accuracy_ema": round(self._sim_accuracy_ema, 3),
            },
        }

    def reset_idle(self):
        """Reset idle timers on user input. Captures terminal state before reset for allostatic analysis."""
        # ── Capture terminal state before reset (Critique 7) ──
        self._terminal_state = classify_terminal_state(self.latent)
        self._terminal_valence = self.latent["valence"]
        self._terminal_arousal = self._arousal_ema
        self._terminal_surprise = self.latent["surprise"]
        self.idle_seconds = 0; self.idle_fe = 0.0

    async def restore_from_memory(self, memory) -> bool:
        """Restore brain state from SQLite on startup. Loads latent state, conversation history,
        covariance, phenotype prior, EMAs, and all extended state from previous sessions."""
        try:
            state = await memory.load_brain_state()
            if state:
                self.latent = {"surprise": state.get("surprise", 0.30), "valence": state.get("valence", 0.00), "velocity": state.get("velocity", 0.30)}
                self.latent_prev, self.action_count, self._msg_counter = dict(self.latent), state.get("action_count", 0), state.get("msg_counter", 0)
                self.topic_summary, self.current_prediction = state.get("topic_summary", ""), state.get("current_prediction") or "(awaiting input…)"
                if state.get("phenotype_json"):
                    try: self.phenotype_prior = json.loads(state.get("phenotype_json"))
                    except Exception: pass
            self.conversation = await memory.load_conversation(limit=40)
            self.messages = await memory.load_messages_for_ui(limit=80)
            # ── Restore extended state (Point 3) ──
            ext = await memory.load_extended_state()
            if ext:
                self._valence_ema = ext.get("valence_ema", 0.0)
                self._arousal_ema = ext.get("arousal_ema", 0.5)
                self._explore_exploit_ratio = ext.get("explore_exploit_ratio", 0.5)
                self._self_model_error = ext.get("self_model_error", 0.0)
                self._terminal_state = ext.get("terminal_state", "default_mode")
                if ext.get("covariance_json"): self._P = ext["covariance_json"]
                if ext.get("predicted_next_vec_json"): self._predicted_next_vec = ext["predicted_next_vec_json"]
                self._predicted_next_text = ext.get("predicted_next_text", "")
                if ext.get("predicted_hyp_vec_json"): self._predicted_hyp_vec = ext["predicted_hyp_vec_json"]
                self._predicted_hyp_text = ext.get("predicted_hyp_text", "")
                if ext.get("intent_history_json"): self._intent_history = deque(ext["intent_history_json"], maxlen=6)
                if ext.get("error_lexical_json"): self._error_lexical = deque(ext["error_lexical_json"], maxlen=10)
                if ext.get("error_semantic_json"): self._error_semantic = deque(ext["error_semantic_json"], maxlen=10)
                if ext.get("error_pragmatic_json"): self._error_pragmatic = deque(ext["error_pragmatic_json"], maxlen=10)
                if ext.get("sim_accuracy_delib_json"): self._sim_accuracy_deliberated = deque(ext["sim_accuracy_delib_json"], maxlen=20)
                if ext.get("sim_accuracy_hyp_json"): self._sim_accuracy_hypothetical = deque(ext["sim_accuracy_hyp_json"], maxlen=20)
                if ext.get("conversation_centroid_json"): self._conversation_centroid = ext["conversation_centroid_json"]
                if ext.get("attention_anchor_json"): self._attention_anchor_vec = ext["attention_anchor_json"]
                if ext.get("action_history_json"):
                    restored = ext["action_history_json"]
                    # Restore prototype action counts from serialized form
                    if isinstance(restored, list):
                        for i, counts in enumerate(restored[:self.N_PROTOTYPES]):
                            if isinstance(counts, dict):
                                self._proto_action_counts[i] = defaultdict(int, counts)
                self._output_valence_alignment = ext.get("output_valence_alignment", 0.0)
                self._output_velocity_alignment = ext.get("output_velocity_alignment", 0.0)
                self._output_context_usage = ext.get("output_context_usage", 0.0)
                self._deliberation_gain_ema = ext.get("deliberation_gain_ema", 0.0)
                self._sim_accuracy_ema = ext.get("sim_accuracy_ema", 0.5)
                if ext.get("pipeline_config_json"):
                    try: self._pipeline_config = ext["pipeline_config_json"]
                    except Exception: pass
            return True
        except Exception as e: logger.error(f"State restore failed: {e}")
        return False

    async def _save_state(self, memory) -> None:
        try: await memory.save_brain_state(self)
        except Exception: pass
        try:
            await memory.save_extended_state({
                "valence_ema": self._valence_ema, "arousal_ema": self._arousal_ema,
                "explore_exploit_ratio": self._explore_exploit_ratio, "self_model_error": self._self_model_error,
                "terminal_state": self._terminal_state, "covariance": self._P,
                "predicted_next_vec": self._predicted_next_vec, "predicted_next_text": self._predicted_next_text,
                "predicted_hyp_vec": self._predicted_hyp_vec, "predicted_hyp_text": self._predicted_hyp_text,
                "intent_history": list(self._intent_history),
                "error_lexical": list(self._error_lexical), "error_semantic": list(self._error_semantic),
                "error_pragmatic": list(self._error_pragmatic),
                "sim_accuracy_delib": list(self._sim_accuracy_deliberated),
                "sim_accuracy_hyp": list(self._sim_accuracy_hypothetical),
                "conversation_centroid": self._conversation_centroid,
                "attention_anchor": self._attention_anchor_vec,
                "action_history": [dict(counts) for counts in self._proto_action_counts],
                "output_valence_alignment": self._output_valence_alignment,
                "output_velocity_alignment": self._output_velocity_alignment,
                "output_context_usage": self._output_context_usage,
                "deliberation_gain_ema": self._deliberation_gain_ema,
                "sim_accuracy_ema": self._sim_accuracy_ema,
                "pipeline_config": self._pipeline_config,
            })
        except Exception: pass

    # ── Episodic Memory Encoding ──

    async def _encode_and_store_memory(self, text: str, latent: dict, llm, memory, vectors, broadcast,
                                        mode: str = "input", error: float = 0.5, precision: float = 1.0,
                                        feedback: dict = None, raw_preview: str = "") -> str:
        """Encode text into long-term episodic memory via LLM + vector store.
        
        Pipeline:
          1. LLM encodes text as factual note (recorder role, tight budget)
          2. Embed the note → 384d vector
          3. Check VAD alignment (does the note match the source affect?)
          4. If alignment < 0.35, retry once with fresh LLM call
          5. Compute encoding strength: f(prediction_error, precision, arousal)
          6. Apply repetition suppression (similar recent memories → reduced strength)
          7. Apply primacy effect (first-cycle memories get boosted)
          8. Store in ChromaDB + SQLite with full metadata
        
        Args:
            text: raw text to encode (may include commitment binding context)
            mode: 'input' for user messages, 'output' for assistant replies
            error: current prediction error (drives encoding strength)
            precision: current precision weight (modulates encoding strength)
            raw_preview: original user text for trace display
            
        Returns: the encoded memory text, or empty string on failure.
        """
        
        for attempt in range(2):
            mem_text = await llm.encode_memory(text, latent, mode=mode, feedback=feedback)
            if not mem_text:
                continue
            
            mem_vec = await vectors.embed(mem_text) if vectors else None
            if mem_vec is None:
                break
            
            # VAD alignment check
            alignment = await vectors.score_memory_alignment(mem_text, mem_vec, latent)
            
            if alignment >= 0.35 or attempt == 1:
                obs_arousal = latent.get("surprise", 0.3)
                enc_strength = compute_encoding_strength(error, precision, obs_arousal)
                
                # ── Repetition Suppression (Grill-Spector et al. 2006) ──
                # Adaptive threshold: fast conversation (high velocity) → stricter suppression
                # Fatigued system → stricter suppression (conserves energy)
                vel = latent.get("velocity", 0.3)
                rep_sim_threshold = max(0.65, 0.75 + vel * 0.1 - (1.0 - self._last_interoceptive.get("energy", 1.0)) * 0.05)
                
                suppression = 1.0
                try:
                    if vectors._episodic_col and vectors._episodic_col.count() > 0:
                        raw_vec = await vectors.embed(text)
                        existing = vectors._episodic_col.query(
                            query_embeddings=[raw_vec], n_results=3, include=["distances"])
                        if existing["distances"] and existing["distances"][0]:
                            top_sim = max(1.0 - d for d in existing["distances"][0])
                            if top_sim > rep_sim_threshold:
                                suppression = max(0.02, (1.0 - top_sim) * 2.0)
                                enc_strength *= suppression
                except Exception:
                    pass
                
                # ── Primacy Effect (Murdock 1962) ──
                # First messages in a conversation set the topic — they should always be
                # retrievable. Without this, a low-surprise first turn gets enc_strength ~0.24
                # and becomes unretrievable after a few turns.
                if self._msg_counter <= 2 and mode == "input":
                    enc_strength = max(0.4, enc_strength)
                
                mem_id = f"ep_{mode}_{self._msg_counter}_{ts().replace(' ','_')}"
                meta = {
                    "ts": ts(), "msg_counter": self._msg_counter,
                    "memory_type": mode, "latent": dict(latent),
                    "covariance": list(self._P), "precision_at": precision,
                    "encoding_strength": enc_strength, "alignment_score": alignment,
                    "error_at": error,
                }
                
                # Store in ChromaDB
                await vectors.store_episodic_memory(mem_id, mem_text, mem_vec, meta)
                
                # Store in SQLite
                await memory.save_episodic_memory({
                    **meta, "memory_text": mem_text,
                    "user_text_preview": raw_preview[:200] if mode == "input" else "",
                    "reply_preview": raw_preview[:200] if mode == "output" else "",
                    "surprise_at": latent.get("surprise", 0.3),
                    "valence_at": latent.get("valence", 0.0),
                    "velocity_at": latent.get("velocity", 0.3),
                })
                
                await broadcast({
                    "type": "system_trace", "label": f"memory_encode_{mode}", "duration_ms": 0,
                    "summary": f"[{mode}] align={alignment:.2f} str={enc_strength:.3f}{' SUPPRESSED('+str(round(suppression,2))+')' if suppression < 1.0 else ''} attempt={attempt+1}: {mem_text[:80]}",
                    "details": {
                        "memory_text": mem_text, "memory_type": mode,
                        "alignment_score": alignment, "encoding_strength": enc_strength,
                        "repetition_suppression": round(suppression, 3),
                        "attempt": attempt + 1, "latent_at_encoding": dict(latent),
                        "covariance": list(self._P),
                    }
                })
                
                # ── Attentional Inertia: update anchor on strong, substantive encodings ──
                orig_words = len(text.split())
                if enc_strength >= 0.3 and suppression >= 0.8 and orig_words >= 3 and mem_vec is not None:
                    self._attention_anchor_vec = list(mem_vec) if not isinstance(mem_vec, list) else mem_vec
                
                # ── Track boredom signals (external repetitiveness + information density) ──
                if mode == "input":
                    self._suppression_history.append(suppression)
                    self._input_lengths.append(len(text.split()))
                
                return mem_text
            
            # Alignment too low — will retry
            await broadcast({
                "type": "system_trace", "label": f"memory_encode_{mode}", "duration_ms": 0,
                "summary": f"[{mode}] alignment={alignment:.2f} BELOW THRESHOLD — retrying",
                "details": {"failed_text": mem_text, "alignment": alignment, "attempt": attempt + 1}
            })
        
        return ""

    # ── Memory Context Assembly ──

    async def _build_memory_context(self, current_vec, latent, vectors, broadcast, n=8, boredom=0.0, adapt=None) -> list:
        """Build conversation context with adaptive boredom-driven self-memory boosting.
        
        adapt dict from _adaptive_thresholds() provides:
          self_boost_factor: how much to amplify self-memories when bored
          ext_dampen_factor: how much to suppress external memories when bored
          boredom_threshold: when class weighting activates
        """
        if not vectors or current_vec is None:
            return []
        
        adapt = adapt or {}
        
        retrieved = await vectors.retrieve_episodic_memories(
            current_vec, latent, current_msg_counter=self._msg_counter, n=n, 
            boredom=boredom, adapt=adapt)
        
        # ── Attentional Inertia (always-merge) ──
        anchor_used = False
        new_from_anchor = 0
        
        if self._attention_anchor_vec is not None and len(retrieved) >= 1:
            anchor_retrieved = await vectors.retrieve_episodic_memories(
                self._attention_anchor_vec, latent, current_msg_counter=self._msg_counter, n=n,
                boredom=boredom, adapt=adapt)
            
            if anchor_retrieved:
                # Merge: deduplicate by mem_id, keep higher-scoring version
                seen_ids = {}
                for r in retrieved:
                    seen_ids[r.get("mem_id", r.get("text", "")[:40])] = r
                for r in anchor_retrieved:
                    rid = r.get("mem_id", r.get("text", "")[:40])
                    if rid not in seen_ids:
                        seen_ids[rid] = r
                        new_from_anchor += 1
                    elif r["retrieval_score"] > seen_ids[rid]["retrieval_score"]:
                        seen_ids[rid] = r
                
                if new_from_anchor > 0:
                    # Re-rank merged results and take top n
                    merged = sorted(seen_ids.values(), key=lambda r: -r["retrieval_score"])[:n]
                    retrieved = merged
                    anchor_used = True
        
        if not retrieved:
            return []
        
        # Sort by msg_counter for temporal ordering
        retrieved.sort(key=lambda r: r.get("msg_counter", 0))
        
        # ── State-Dependent Working Memory Compression ──
        # Recent memories → full detail (working memory buffer)
        # Older memories → compressed proportional to state:
        #   High surprise → keep more detail (need context to resolve uncertainty)
        #   Low surprise → compress more (things are settled, save bandwidth)
        #   Low energy → compress more (conserve resources)
        s = latent.get("surprise", 0.3)
        
        context = []
        trace_entries = []
        prev_counter = None
        
        for ep in retrieved:
            role = "user" if ep["memory_type"] == "input" else "assistant"
            text = ep["text"]
            turns_since = ep.get("turns_since", 0)
            
            # Compression: recent items are kept whole, older items fade
            if turns_since <= 1:
                # Working memory: full detail
                compressed_text = text
            else:
                # Fading memory: retention based on recency + state
                # High surprise = need context = keep more
                retention = max(0.25, 0.8 - turns_since * 0.1 + s * 0.3)
                total_words = len(text.split())
                keep_words = max(5, int(total_words * retention))
                
                if keep_words < total_words:
                    words = text.split()
                    compressed_text = " ".join(words[:keep_words])
                else:
                    compressed_text = text
            
            context.append({"role": role, "content": compressed_text, "_order": ep.get("msg_counter", 0)})
            
            mc = ep.get("msg_counter", 0)
            gap = (mc - prev_counter) if prev_counter is not None else 0
            prev_counter = mc
            
            trace_entries.append({
                "text": ep["text"][:80],
                "compressed_to": len(compressed_text.split()),
                "original_words": len(ep["text"].split()),
                "type": ep["memory_type"],
                "msg_counter": mc,
                "turns_since": ep.get("turns_since", 0),
                "temporal_gap": gap if gap > 2 else 0,
                "retrieval_score": ep["retrieval_score"],
                "similarity": ep["similarity"],
                "retrievability": ep["retrievability"],
                "recency": ep["recency"],
                "encoding_strength": ep["encoding_strength"],
                "resonance": ep["resonance"],
                "fidelity": ep["fidelity"],
                "recall_count": ep["recall_count"],
            })
        
        gaps = [t["temporal_gap"] for t in trace_entries if t["temporal_gap"] > 2]
        score_range = f"[{trace_entries[-1]['retrieval_score']:.3f} — {trace_entries[0]['retrieval_score']:.3f}]" if trace_entries else "[]"
        anchor_note = f" | ANCHOR added {new_from_anchor}" if anchor_used else ""
        boredom_note = f" | BOREDOM {boredom:.2f}" if boredom > 0.3 else ""
        
        await broadcast({
            "type": "system_trace", "label": "memory_assembly", "duration_ms": 0,
            "summary": f"Assembled {len(retrieved)} memories | gaps: {len(gaps)} | scores: {score_range}{anchor_note}{boredom_note}",
            "details": {
                "memories": trace_entries, "temporal_gaps": gaps, 
                "total_retrieved": len(retrieved),
                "anchor_used": anchor_used,
                "boredom": boredom,
            }
        })
        
        # Strip internal vectors before returning to caller
        for r in retrieved:
            r.pop("_vec", None)
        
        return context

    def _adaptive_thresholds(self, energy: float = 1.0) -> dict:
        """Compute all adaptive thresholds from current system state.
        
        Inputs:
            energy [0-1]: interoceptive energy
            allostatic_load: rolling error accumulation
            surprise: current latent surprise
            precision: signal confidence
            explore_ratio: exploration tendency
            velocity: conversation pace
            input_lengths: recent user message word counts
            
        Every hardcoded threshold in the system should ultimately come from here.
        """
        load = getattr(self, '_allostatic_load', 0.3)
        s = self.latent.get("surprise", 0.3)
        vel = self.latent.get("velocity", 0.3)
        precision = getattr(self, '_last_precision', 1.0)
        explore = self._explore_exploit_ratio
        
        # ── Fatigue: composite tiredness ──
        # 0.0 = fresh, 1.0 = exhausted
        fatigue = max(0.0, min(1.0, (1.0 - energy) * 0.6 + load * 0.4))
        
        # ── Novelty: how much new information is coming in ──
        # Low surprise + high fatigue = nothing new happening + system is tired
        novelty = max(0.0, min(1.0, s * 0.7 + (1.0 - fatigue) * 0.3))
        
        # ── Boredom threshold: when does the system turn inward? ──
        # Fresh + high novelty = tolerant (0.6), don't turn inward
        # Fatigued + low novelty = turn inward fast (0.1)
        boredom_threshold = round(max(0.1, 0.3 + novelty * 0.3 - fatigue * 0.2), 3)
        
        # ── Self-memory boost: how much to amplify own thoughts when bored ──
        # Low novelty = strong inward turn (2.0×)
        # High novelty = mild boost (1.0×)
        # High precision = confident about own thoughts → stronger boost
        self_boost_factor = round(1.0 + (1.0 - novelty) * 1.0 + (precision - 1.0) * 0.5, 3)
        
        # ── External dampening: how much to suppress external when bored ──
        # High precision = confident about what to ignore → stronger dampening
        ext_dampen_factor = round(max(0.15, 0.45 + (precision - 1.0) * 0.1), 3)
        
        # ── Active inference threshold: when to continue own thought ──
        # Same as boredom threshold — they should fire together
        active_inference_threshold = boredom_threshold
        
        # ── Information density baseline: relative message length ──
        # Mean and std of recent user message lengths
        if self._input_lengths and len(self._input_lengths) >= 3:
            lengths = list(self._input_lengths)
            mean_len = sum(lengths) / len(lengths)
            # Short message = word_count / mean_len < 0.3
            short_msg_ratio = round(max(0.1, 0.3 * (1.0 + mean_len * 0.02)), 3)
        else:
            mean_len = 10.0  # default assumption
            short_msg_ratio = 0.3
        
        return {
            "boredom_threshold": boredom_threshold,
            "active_inference_threshold": active_inference_threshold,
            "self_boost_factor": self_boost_factor,
            "ext_dampen_factor": ext_dampen_factor,
            "mean_input_length": round(mean_len, 1),
            "short_msg_ratio": short_msg_ratio,
            # Memory system thresholds
            "rep_suppression_sim": round(max(0.65, 0.75 + vel * 0.1 - fatigue * 0.1), 3),
            "lateral_inhibition_sim": round(max(0.70, 0.85 - explore * 0.1 - fatigue * 0.05), 3),
            "habituation_fast_s": round(max(30, 120 - fatigue * 90)),
            "habituation_slow_s": round(max(120, 600 - fatigue * 400)),
            # Diagnostics
            "fatigue": round(fatigue, 3),
            "novelty": round(novelty, 3),
        }

    def _next_id(self, role: str) -> str:
        self._msg_counter += 1
        return f"{role}_{self._msg_counter}_{ts().replace(' ','_')}"

    # ── Self-Model: Prototype-Based (Point 9) ──

    def _latent_vec(self) -> list:
        return [self.latent["surprise"], self.latent["valence"], self.latent["velocity"]]

    def _prototype_weights(self, latent_vec: list = None) -> list:
        """Gaussian RBF activation weights across all prototypes."""
        if latent_vec is None: latent_vec = self._latent_vec()
        sigma_sq = self.PROTOTYPE_SIGMA ** 2
        weights = []
        for proto in self._prototypes:
            dist_sq = sum((latent_vec[i] - proto[i]) ** 2 for i in range(3))
            weights.append(math.exp(-dist_sq / (2 * sigma_sq)))
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def predict_own_action(self) -> str:
        """Predict action type from weighted combination of prototype distributions."""
        weights = self._prototype_weights()
        combined = Counter()
        for idx, w in enumerate(weights):
            for action, count in self._proto_action_counts[idx].items():
                combined[action] += count * w
        if not combined:
            return "elaboration"
        return combined.most_common(1)[0][0]

    def update_self_model(self, actual_action: str) -> float:
        """Record action, update nearest prototype, compute self-model error."""
        predicted = self.predict_own_action()
        self._last_predicted_action = predicted
        self._last_actual_action = actual_action
        latent_vec = self._latent_vec()
        weights = self._prototype_weights(latent_vec)

        # Distribute action count to prototypes weighted by activation
        for idx, w in enumerate(weights):
            if w > 0.05:  # only update significantly activated prototypes
                self._proto_action_counts[idx][actual_action] += 1

        # Adapt nearest prototype toward current latent (0.05 learning rate)
        nearest_idx = max(range(len(weights)), key=lambda i: weights[i])
        for d in range(3):
            self._prototypes[nearest_idx][d] = round(
                0.95 * self._prototypes[nearest_idx][d] + 0.05 * latent_vec[d], 4)

        # Compute soft error: probability of actual action under weighted distribution
        combined = Counter()
        for idx, w in enumerate(weights):
            for action, count in self._proto_action_counts[idx].items():
                combined[action] += count * w
        total = sum(combined.values()) or 1.0
        prob = combined.get(actual_action, 0) / total
        error = round(1.0 - prob, 4)

        self._self_model_error = round(0.3 * error + 0.7 * self._self_model_error, 4)
        
        # ── Feedback: self-model error → explore/exploit (Point 10) ──
        # High self-model error = behavior changing = consolidate (exploit)
        # Low self-model error = stable behavior = explore
        self_model_push = (0.5 - self._self_model_error) * 0.15
        self._explore_exploit_ratio = round(max(0.0, min(1.0, self._explore_exploit_ratio + self_model_push)), 4)
        
        return error

    # ── Internal Action EFE (pure physics) ──

    def start_idle_loop(self, llm, memory, broadcast, vectors=None):
        """Start the Default Mode Network (idle loop). Runs Langevin dynamics,
        dream synthesis, memory compression, and attractor analysis when no user input."""
        self._idle_task = asyncio.create_task(self._idle_loop(llm, memory, broadcast, vectors))

    async def _dream_synthesis(self, vectors, llm, memory, broadcast):
        """Dream: find conflicting memories and synthesize novel connections.
        Selection criterion: high similarity + conflicting valence = unresolved tension."""
        if not vectors: return
        try:
            conflicts = await vectors.find_conflicting_memories(n_pairs=2)
            if not conflicts:
                return
            
            for conflict in conflicts:
                dream_text = await llm.generate_dream(conflict["texts"], self.latent)
                if dream_text and len(dream_text) > 5:
                    dream_vec = await vectors.embed(dream_text)
                    # Store as dream memory — low encoding_strength (fragile)
                    dream_id = f"ep_dream_{self._msg_counter}_{ts().replace(' ','_')}"
                    meta = {
                        "ts": ts(), "msg_counter": self._msg_counter,
                        "memory_type": "dream", "latent": dict(self.latent),
                        "covariance": list(self._P), "precision_at": 0.5,
                        "encoding_strength": 0.15,  # dreams are fragile
                        "alignment_score": 0.5, "error_at": 0.0,
                    }
                    await vectors.store_episodic_memory(dream_id, dream_text, dream_vec, meta)
                    
                    await broadcast({
                        "type": "system_trace", "label": "dream_synthesis", "duration_ms": 0,
                        "summary": f"Dream: tension={conflict['tension_score']:.2f} → {dream_text[:80]}",
                        "details": {"source_memories": [t[:60] for t in conflict["texts"]], "tension_score": conflict["tension_score"], "dream": dream_text}
                    })
        except Exception as e:
            logger.debug(f"Dream synthesis failed: {e}")

    async def _compress_old_memories(self, vectors, llm, memory, broadcast):
        """Compression: old weak memories become terser. Strong memories resist compression."""
        if not vectors: return
        try:
            old_mems = await memory.load_episodic_memories(limit=50)
            if not old_mems: return
            
            for mem in old_mems[:5]:  # process up to 5 per cycle
                enc_str = float(mem.get("encoding_strength", 0.5))
                text = mem.get("memory_text", "")
                words = len(text.split())
                
                # Strong memories resist compression
                if enc_str > 0.5 or words <= 6:
                    continue
                
                # Turns since encoding → compression aggressiveness
                turns = max(0, self._msg_counter - int(mem.get("msg_counter", 0)))
                if turns < 20:
                    continue  # too recent
                
                # Word budget shrinks with age and low encoding strength
                target_words = max(4, int(words * (0.3 + enc_str * 0.5)))
                if target_words >= words - 2:
                    continue  # already short enough
                
                compressed = await llm.compress_memory(text, target_words, self.latent)
                if compressed and len(compressed) > 3:
                    comp_vec = await vectors.embed(compressed)
                    orig_vec = await vectors.embed(text)
                    preservation = _cosim(comp_vec, orig_vec) if orig_vec is not None and comp_vec is not None else 0.0
                    
                    if preservation > 0.70:
                        # Good compression — replace
                        mem_id = mem.get("id")
                        if mem_id:
                            await memory.update_episodic_recall(
                                mem_id, compressed, mem.get("valence_at", 0.0), mem.get("valence_drift", 0.0))
                        
                        await broadcast({
                            "type": "system_trace", "label": "memory_compression", "duration_ms": 0,
                            "summary": f"Compressed [{words}→{len(compressed.split())}w] preservation={preservation:.2f}: {compressed[:60]}",
                            "details": {"original": text[:80], "compressed": compressed, "preservation": round(preservation, 3), "encoding_strength": enc_str, "turns_since": turns}
                        })
                        break  # one compression per cycle
        except Exception as e:
            logger.debug(f"Memory compression failed: {e}")

    async def _unresolved_attractor(self, vectors, memory, broadcast):
        """Find unresolved memories and store them as Langevin attractor wells.
        
        Instead of directly manipulating latent state, populates _idle_attractors
        with positions and tensions. The idle_drift Langevin dynamics will then
        naturally pull the state toward these wells with force ∝ tension.
        """
        if not vectors: return
        try:
            mems = await memory.load_episodic_memories(limit=50)
            if not mems: return
            
            attractors = []
            for m in mems:
                enc = float(m.get("encoding_strength", 0.5))
                recalls = int(m.get("recall_count", 0))
                val = float(m.get("valence_at", 0.0))
                words = len(m.get("memory_text", "").split())
                substance = min(1.0, words / 15.0)
                tension = enc * (1.0 / (1.0 + recalls * 0.3)) * (0.5 + abs(val)) * substance
                
                if tension > 0.1:
                    attractors.append({
                        "surprise": float(m.get("surprise_at", 0.3)),
                        "valence": val,
                        "velocity": float(m.get("velocity_at", 0.3)),
                        "tension": round(tension, 3),
                        "text": m.get("memory_text", "")[:60],
                    })
            
            # Keep top 3 attractors by tension
            attractors.sort(key=lambda a: -a["tension"])
            self._idle_attractors = attractors[:3]
            
            if self._idle_attractors:
                top = self._idle_attractors[0]
                await broadcast({
                    "type": "system_trace", "label": "unresolved_attractor", "duration_ms": 0,
                    "summary": f"Langevin wells: {len(self._idle_attractors)} | top: '{top['text']}' (tension={top['tension']:.3f})",
                    "details": {"attractors": self._idle_attractors}
                })
        except Exception as e:
            logger.debug(f"Unresolved attractor failed: {e}")

    async def _idle_loop(self, llm, memory, broadcast, vectors):
        while True:
            await asyncio.sleep(1)
            if self.is_processing: continue
            self.idle_seconds += 1
            phase = idle_phase(self.idle_seconds)
            
            # ── Latent drift via Langevin dynamics + covariance inflation ──
            if self.idle_seconds >= IDLE_PHASES[1]["threshold"]:
                self.latent = idle_drift(
                    self.latent, 
                    phenotype_prior=self.phenotype_prior,
                    free_energy=self.free_energy_val,
                    unresolved_attractors=self._idle_attractors)
                # Genuine FE: covariance inflates via process noise (Kalman predict step)
                self._P = [min(p + q, 1.0) for p, q in zip(self._P, self.PROCESS_NOISE)]
                self.free_energy_val = compute_free_energy(
                    self.rolling_error, self.latent, self.phenotype_prior, self._P)

            # ── Unresolved memory attractor (every 60s after drift) ──
            if self.idle_seconds >= 60 and self.idle_seconds % 60 == 0 and vectors:
                asyncio.create_task(self._unresolved_attractor(vectors, memory, broadcast))

            # ── Dream synthesis (at 120s) ──
            if self.idle_seconds == 120 and vectors and not self.is_processing:
                asyncio.create_task(self._dream_synthesis(vectors, llm, memory, broadcast))

            # ── Memory compression (at 180s) ──
            if self.idle_seconds == 180 and vectors and not self.is_processing:
                asyncio.create_task(self._compress_old_memories(vectors, llm, memory, broadcast))
                
            await broadcast({"type": "idle_tick", "data": self.state_snapshot()})

    async def process_input(self, user_text: str, llm: Any, memory: Any, broadcast: Callable, vectors: Any = None, origin: str = "user"):
        """Main 14-step processing pipeline. See module docstring for full pipeline description.
        
        Each step broadcasts trace events via WebSocket for real-time diagnostics.
        All trace events are also collected by CycleDiagnostic for end-of-cycle analysis.
        
        Args:
            user_text: raw user input
            llm: LLMClient instance (language rendering surface)
            memory: Memory instance (SQLite persistence)
            broadcast: async function to send WebSocket messages
            vectors: EmbeddingStore instance (vector space operations)
            origin: 'user' for real input, 'idle' for DMN-generated input
        """
        if self.is_processing and origin == "user": return
        if origin == "user": self.reset_idle()
        self.is_processing = True
        
        # ── Diagnostic: reset and intercept traces ──
        self._diag_cycle += 1
        self._diag.reset(self._diag_cycle)
        self._diag.begin_cycle(
            self._diag_cycle, user_text, self.latent,
            raw_conversation=self.conversation[-8:],
            prev_prediction=self.current_prediction or "")
        _original_broadcast = broadcast
        async def _traced_broadcast(msg):
            if msg.get("type") in ("system_trace", "llm_trace"):
                self._diag.ingest(msg)
            await _original_broadcast(msg)
        broadcast = _traced_broadcast
        
        llm.reset_cycle_stats(feedback={
            "valence_misalignment": self._output_valence_alignment,
            "velocity_misalignment": self._output_velocity_alignment,
            "sim_accuracy": self._sim_accuracy_ema,
            "deliberation_gain": self._deliberation_gain_ema,
            "budget_accuracy": llm._budget_accuracy_ema,
            "energy": llm.energy,
            "precision": getattr(self, '_last_precision', 1.0),
            "explore_ratio": self._explore_exploit_ratio,
        })
        await broadcast({"type": "processing", "value": True})
        msg_ts = ts()

        if origin == "user":
            await broadcast({"type": "user_message", "message": {"role": "user", "content": user_text, "ts": msg_ts, "error": None, "origin": "user", "enriched": {}}})

        self.conversation.append({"role": "user", "content": user_text})
        self._user_msg_lengths.append(len(user_text.split()))
        # Update running word count baseline (EMA, alpha=0.3)
        user_words = len(user_text.split())
        self._user_word_ema = round(0.3 * user_words + 0.7 * self._user_word_ema, 2)
        await memory.append_message("user", user_text)
        msg_id = self._next_id("user")

        try:
            prior_latent = dict(self.latent)

            # ── 1. ENCODE ──────────
            enriched = await vectors.encode_message(user_text, self.latent, covariance=list(self._P)) if vectors else {"prediction_error":{}}
            enriched["ts"] = msg_ts
            self.last_enriched = enriched
            concepts_extracted = [c["label"] for c in enriched.get("concepts", []) if "label" in c]

            await broadcast({
                "type": "system_trace", "label": "sensory_encode", "duration_ms": 0, "summary": f"Velocity: {enriched.get('velocity', 0.3)} | Intent: {enriched.get('intent', 'unknown')}",
                "details": { "velocity": enriched.get('velocity', 0.3), "affective_valence": enriched.get('tone', {}).get('valence', 0.0), "intent_winner": enriched.get('intent', 'unknown'), "intent_scores": enriched.get('intent_breakdown', {}) }
            })

            # ── 1b. PERCEPTUAL ENRICHMENT ──────────
            msg_vec = enriched.get("vector")
            irony_distance = 0.0

            # ── 1c. UPDATE USER MODEL ──────────
            if vectors:
                obs_v = enriched.get("tone", {}).get("valence", 0.0)
                obs_intent = enriched.get("intent", "other")
                vectors.update_user_model(user_text, obs_v, obs_intent)

            # ── 1d. ENCODE EXPECTATION (for pragmatic surprise) ──────────
            expectation_text = ""
            expectation_vec = None
            if vectors and self.conversation:
                expectation_text = await llm.encode_expectation(self.conversation, self.latent)
                if expectation_text:
                    expectation_vec = await vectors.embed(expectation_text)

            # ── 2. TRUE PREDICTIVE ERROR + SIMULATION ACCURACY ──────────
            semantic_error = None
            deliberated_sim, hypothetical_sim = None, None
            expectation_error = None

            if msg_vec is not None:
                # Expectation error: pragmatic-level surprise (form/intent, not content)
                if expectation_vec is not None:
                    exp_sim = max(-1.0, min(1.0, _cosim(msg_vec, expectation_vec)))
                    expectation_error = round(1.0 - exp_sim, 4)
                    await broadcast({
                        "type": "system_trace", "label": "expectation_error", "duration_ms": 0,
                        "summary": f"Expected: '{expectation_text}' | Pragmatic surprise: {expectation_error:.3f}",
                        "details": {"expectation": expectation_text, "expectation_error": expectation_error}
                    })

            if msg_vec is not None:
                # Deliberated prediction accuracy (primary error signal)
                if self._predicted_next_vec is not None:
                    sim = max(-1.0, min(1.0, _cosim(msg_vec, self._predicted_next_vec)))
                    semantic_error = round(1.0 - sim, 4)
                    deliberated_sim = round(sim, 4)
                    self._sim_accuracy_deliberated.append(deliberated_sim)

                # Hypothetical prediction accuracy (reflex baseline)
                if self._predicted_hyp_vec is not None:
                    sim_hyp = max(-1.0, min(1.0, _cosim(msg_vec, self._predicted_hyp_vec)))
                    hypothetical_sim = round(sim_hyp, 4)
                    self._sim_accuracy_hypothetical.append(hypothetical_sim)

                # Deliberation gain: did enrichment improve prediction?
                deliberation_gain = None
                if deliberated_sim is not None and hypothetical_sim is not None:
                    deliberation_gain = round(deliberated_sim - hypothetical_sim, 4)

                # Running means
                mean_delib = round(sum(self._sim_accuracy_deliberated) / len(self._sim_accuracy_deliberated), 3) if self._sim_accuracy_deliberated else None
                mean_hyp = round(sum(self._sim_accuracy_hypothetical) / len(self._sim_accuracy_hypothetical), 3) if self._sim_accuracy_hypothetical else None

                if semantic_error is not None or hypothetical_sim is not None:
                    await broadcast({
                        "type": "system_trace", "label": "simulation_accuracy", "duration_ms": 0,
                        "summary": f"Deliberated: {deliberated_sim or '?'} | Hypothetical: {hypothetical_sim or '?'} | Gain: {deliberation_gain or '?'}",
                        "details": {
                            "deliberated_prediction": self._predicted_next_text,
                            "hypothetical_prediction": self._predicted_hyp_text,
                            "actual_user_text": user_text[:200],
                            "deliberated_similarity": deliberated_sim,
                            "hypothetical_similarity": hypothetical_sim,
                            "deliberation_gain": deliberation_gain,
                            "semantic_error": semantic_error,
                            "running_mean_deliberated": mean_delib,
                            "running_mean_hypothetical": mean_hyp,
                            "n_samples": len(self._sim_accuracy_deliberated),
                        }
                    })

                # ── Feedback EMA updates (Point 10) ──
                if deliberated_sim is not None:
                    self._sim_accuracy_ema = round(0.2 * deliberated_sim + 0.8 * self._sim_accuracy_ema, 4)
                if deliberation_gain is not None:
                    self._deliberation_gain_ema = round(0.2 * deliberation_gain + 0.8 * self._deliberation_gain_ema, 4)

            # ── 3. AFFECTIVE BLEED (Memory Impact Physics) ──────────
            retrieved = enriched.get("retrieved", [])
            affective_bleed_trace = None
            if retrieved:
                # Compute similarity of recent messages to the retrieved set (for retrieval_surprise)
                recent_sim = enriched.get("top_similarity", 0.5)

                total_weight, weighted_hist_v = 0.0, 0.0
                impact_details = []
                for r in retrieved:
                    sim = r.get("similarity", 0.0)
                    hist_v = r.get("latent", {}).get("valence")
                    if hist_v is None or sim < 0.55:
                        continue
                    
                    enc_str = r.get("encoding_strength", 0.5)
                    impact, breakdown = compute_memory_impact(
                        encoding_strength=enc_str,
                        ts_stored=r.get("ts", ""),
                        latent_stored=r.get("latent", {}),
                        latent_current=prior_latent,
                        similarity_to_recent=recent_sim
                    )
                    
                    weight = sim * impact * 0.25
                    total_weight += weight
                    weighted_hist_v += hist_v * weight
                    impact_details.append({"text": r.get("text", "")[:60], "impact": impact, "weight": round(weight, 4), **breakdown})
                
                if total_weight > 0:
                    avg_pull = weighted_hist_v / total_weight
                    prior_latent["valence"] = round(prior_latent["valence"] * (1 - min(total_weight, 0.8)) + avg_pull * min(total_weight, 0.8), 4)
                    affective_bleed_trace = {"memory_weight": round(total_weight, 3), "historical_valence": round(avg_pull, 3), "new_prior_valence": prior_latent["valence"], "memories": impact_details}

            if affective_bleed_trace: await broadcast({"type": "system_trace", "label": "affective_bleed", "duration_ms": 0, "summary": f"Memory impact: {len(impact_details)} memories, total weight={total_weight:.3f}, pull V={affective_bleed_trace['historical_valence']:+.2f}", "details": affective_bleed_trace})

            # ── 3b. EPISODIC RECALL (6-Rule Dynamics) ──────────
            if vectors and msg_vec is not None:
                try:
                    episodic_recalled = await vectors.retrieve_episodic_memories(msg_vec, prior_latent, n=6)
                    if episodic_recalled:
                        ep_total_weight = 0.0
                        ep_weighted_v = 0.0
                        ep_covariance_inflation = 0.0
                        ep_energy_cost = 0.0
                        any_flip = False
                        ep_details = []
                        
                        for ep in episodic_recalled:
                            rw = ep["recall_weight"]
                            fidelity = ep["fidelity"]
                            stored_v = ep["stored_latent"].get("valence", 0.0)
                            original_v = ep.get("original_valence", stored_v)
                            
                            ep_total_weight += rw
                            ep_weighted_v += stored_v * rw
                            
                            # Valence flip tracking (for diagnostics, no manual injection)
                            if ep["valence_flip"]:
                                any_flip = True
                            
                            # ── Covariance inflation from contradictory recall ──
                            if ep["state_distance"] > 0.5:
                                ep_covariance_inflation += ep["state_distance"] * 0.02 * fidelity
                            
                            # ── Energy cost of recall ──
                            ep_energy_cost += fidelity * 0.05
                            
                            # ── Reconsolidation (fidelity-gated) ──
                            reconsolidated = False
                            if fidelity >= 0.2 and vectors:
                                try:
                                    # Determine source material by fidelity
                                    if fidelity > 0.5:
                                        # High fidelity: re-encode from raw text (vivid recollection)
                                        # Try to get raw from SQLite
                                        raw_rows = await memory.load_episodic_memories(limit=100)
                                        raw_text = None
                                        for row in raw_rows:
                                            if row.get("memory_text") == ep["text"]:
                                                raw_text = row.get("user_text_preview") or row.get("reply_preview") or ep["text"]
                                                break
                                        source_text = raw_text if raw_text else ep["text"]
                                    else:
                                        # Medium fidelity: re-encode existing memory (remembering your memory)
                                        source_text = ep["text"]
                                    
                                    # Word budget scales with fidelity
                                    base_budget = max(8, int(15 + prior_latent.get("surprise", 0.3) * 35))
                                    re_budget = int(base_budget * (0.5 + fidelity * 1.5))
                                    
                                    mode = ep["memory_type"] if ep["memory_type"] in ("input", "output") else "input"
                                    new_mem_text = await llm.encode_memory(source_text, prior_latent, mode=mode)
                                    
                                    if new_mem_text and len(new_mem_text) > 5:
                                        new_vec = await vectors.embed(new_mem_text)
                                        new_valence = prior_latent.get("valence", 0.0)
                                        
                                        # ── Rule 4: Valence erosion tracking ──
                                        new_drift = round(original_v - new_valence, 4)
                                        new_count = ep["recall_count"] + 1
                                        
                                        # Update ChromaDB
                                        await vectors.reconsolidate_memory(
                                            ep["mem_id"], new_mem_text, new_vec,
                                            new_valence, new_drift, new_count)
                                        
                                        # Update SQLite
                                        # Find matching row by text
                                        for row in raw_rows if fidelity > 0.5 else []:
                                            if row.get("memory_text") == ep["text"] and row.get("id"):
                                                await memory.update_episodic_recall(
                                                    row["id"], new_mem_text, new_valence, new_drift)
                                                break
                                        else:
                                            # Low-fidelity: just bump count
                                            for row in (await memory.load_episodic_memories(limit=100)):
                                                if row.get("memory_text") == ep["text"] and row.get("id"):
                                                    await memory.bump_recall_count(row["id"])
                                                    break
                                        
                                        reconsolidated = True
                                        
                                        # ── Semantic Diff: track reconsolidation bias ──
                                        try:
                                            diff_text = await llm.generate_reconsolidation_diff(ep["text"], new_mem_text)
                                            if diff_text:
                                                await vectors.track_reconsolidation_delta(diff_text)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    logger.debug(f"Reconsolidation failed: {e}")
                            
                            ep_details.append({
                                "text": ep["text"][:80], "type": ep["memory_type"],
                                "recall_weight": rw, "fidelity": fidelity,
                                "resonance": ep["resonance"], "state_distance": ep["state_distance"],
                                "valence_flip": ep["valence_flip"], "similarity": ep["similarity"],
                                "recall_count": ep["recall_count"],
                                "frequency_amp": ep["frequency_amplifier"],
                                "habituation": ep["habituation"],
                                "valence_drift": ep["valence_drift"],
                                "reconsolidated": reconsolidated,
                            })
                        
                        if ep_total_weight > 0.05:
                            ep_avg_v = ep_weighted_v / ep_total_weight
                            
                            # ── Bayesian Surprise Integration ──
                            # Treat recalled memories as an observation with precision
                            # proportional to total recall weight. The Kalman filter
                            # naturally absorbs the shock: contradictory memories produce
                            # large innovation → large surprise increase. No manual injection.
                            mem_obs_surprise = prior_latent["surprise"]  # no direct surprise observation from memory
                            mem_obs_valence = ep_avg_v  # weighted average recalled valence
                            mem_obs_velocity = prior_latent["velocity"]  # memory doesn't change velocity
                            mem_precision = min(2.0, ep_total_weight * 2.0)  # weight → precision
                            
                            # If there's a valence flip, the innovation will be mathematically large
                            # because ep_avg_v diverges from prior_latent["valence"]
                            mem_latent, mem_P, mem_gain, _ = update_latent(
                                prior_latent, mem_obs_surprise, mem_obs_valence, mem_obs_velocity,
                                precision=mem_precision, P=list(self._P), Q=[0, 0, 0]  # no process noise for memory update
                            )
                            prior_latent["valence"] = mem_latent["valence"]
                            prior_latent["surprise"] = mem_latent["surprise"]
                            # Don't update P from memory recall — that happens in the main Kalman step
                            
                            # ── Covariance inflation from contradictory recall ──
                            if ep_covariance_inflation > 0:
                                self._P = [round(p + ep_covariance_inflation, 4) for p in self._P]
                            
                            # ── Energy cost ──
                            if ep_energy_cost > 0:
                                llm._budget_overshoot_ema = round(
                                    llm._budget_overshoot_ema + ep_energy_cost * 0.3, 4)
                            
                            await broadcast({
                                "type": "system_trace", "label": "episodic_recall", "duration_ms": 0,
                                "summary": f"Recalled {len(episodic_recalled)} | weight={ep_total_weight:.3f} | V pull={ep_avg_v:+.2f} | flip={'YES' if any_flip else 'no'} | Kalman gain={mem_gain:.3f} | P inflate={ep_covariance_inflation:.3f}",
                                "details": {
                                    "total_weight": round(ep_total_weight, 3),
                                    "avg_valence_pull": round(ep_avg_v, 3),
                                    "kalman_gain": round(mem_gain, 3),
                                    "mem_precision": round(mem_precision, 3),
                                    "covariance_inflation": round(ep_covariance_inflation, 4),
                                    "energy_cost": round(ep_energy_cost, 3),
                                    "any_valence_flip": any_flip,
                                    "episodes": ep_details,
                                }
                            })
                except Exception as e:
                    logger.debug(f"Episodic recall failed: {e}")

            # ── 4. AFFECTIVE PREDICTION ERROR (Critique 4) ──────────
            obs_valence = enriched.get("tone", {}).get("valence", 0.0)
            obs_arousal = enriched.get("tone", {}).get("arousal", 0.5)
            obs_velocity = enriched.get("velocity", 0.3)

            # ── Precision-Weighted Perception: free energy modulates affective input ──
            # High FE (anxious) → amplify arousal (hypervigilance)
            # Low FE (bored) → dampen valence (emotional flattening)
            fe = self.free_energy_val
            if fe > 0.5:
                obs_arousal = min(1.0, obs_arousal * (1.0 + (fe - 0.5) * 0.6))  # amplify
            elif fe < 0.2:
                obs_valence = obs_valence * (0.5 + fe * 2.5)  # dampen toward zero

            # Error computed against the EMA (the system's affective prediction)
            predicted_valence = self._valence_ema
            predicted_arousal = self._arousal_ema
            affective_error = round(abs(predicted_valence - obs_valence), 4)
            arousal_error = round(abs(predicted_arousal - obs_arousal), 4)
            
            # Update EMAs after computing error
            alpha = self._affective_alpha
            self._valence_ema = round(alpha * obs_valence + (1 - alpha) * self._valence_ema, 4)
            self._arousal_ema = round(alpha * obs_arousal + (1 - alpha) * self._arousal_ema, 4)

            await broadcast({
                "type": "system_trace", "label": "affective_prediction_error", "duration_ms": 0,
                "summary": f"Affective Error: V={affective_error:.3f} A={arousal_error:.3f} | Predicted V={predicted_valence:+.2f} → Observed V={obs_valence:+.2f}",
                "details": {"predicted_valence": round(predicted_valence, 3), "observed_valence": obs_valence, "valence_error": affective_error, "predicted_arousal": round(predicted_arousal, 3), "observed_arousal": obs_arousal, "arousal_error": arousal_error, "updated_ema_valence": self._valence_ema, "updated_ema_arousal": self._arousal_ema, "ema_alpha": alpha}
            })

            # ── 5. LEXICAL ERROR (Perplexity) ──────────
            context_for_perplexity = self.conversation[:-1][-6:]
            perplexity_result = await llm.compute_perplexity(user_text, context_for_perplexity)
            lexical_error = perplexity_result["surprise"] if perplexity_result else None
            logprob_variance = perplexity_result["logprob_variance"] if perplexity_result else None

            # ── 6. ERROR COMPOSITION ──────────
            # Hierarchical: semantic (system's own prediction) > lexical (LLM prior) > fallback (cosine)
            pred_err_dict = enriched.get("prediction_error", {})
            
            if semantic_error is not None:
                # Primary: system's own prediction error
                primary_error = semantic_error
                error_mode = "predictive"
            elif lexical_error is not None:
                primary_error = lexical_error
                error_mode = "perplexity"
            else:
                primary_error = pred_err_dict.get("error", 0.5)
                error_mode = pred_err_dict.get("error_mode", "cosine")

            # ── Pragmatic error (Critique 1): intent shift from predicted pattern ──
            observed_intent = enriched.get("intent", "other")
            pragmatic_error = 0.0
            if self._intent_history:
                # How unexpected is this intent given recent history?
                match_rate = sum(1 for i in self._intent_history if i == observed_intent) / len(self._intent_history)
                pragmatic_error = round(1.0 - match_rate, 4)

            # ── Hierarchical error logging (Critique 1) ──
            if lexical_error is not None: self._error_lexical.append(lexical_error)
            if semantic_error is not None: self._error_semantic.append(semantic_error)
            self._error_pragmatic.append(pragmatic_error)

            # Blend in affective error — emotional surprise modulates the composite
            affective_weight = 0.25 if affective_error > 0.3 else 0.10
            current_error = round(primary_error * (1 - affective_weight) + affective_error * affective_weight, 4)

            if pred_err_dict.get("retrieval_triggered"):
                current_error = round(current_error * enriched.get("prediction_error", {}).get("dampener", 0.85), 4)
            if concepts_extracted:
                await memory.update_concept_uncertainty(concepts_extracted, current_error)

            await broadcast({
                "type": "system_trace", "label": "error_composition", "duration_ms": 0,
                "summary": f"Composite Error = {current_error:.3f} ({error_mode}) | Pragmatic Δ = {pragmatic_error:.2f} | Affective Δ = {affective_error:.3f}",
                "details": {"semantic_error": semantic_error, "lexical_error": lexical_error, "pragmatic_error": pragmatic_error, "affective_error": affective_error, "affective_weight": affective_weight, "composite_error": current_error, "error_mode": error_mode,
                            "hierarchy_means": {"lexical": round(sum(self._error_lexical)/max(len(self._error_lexical),1), 3) if self._error_lexical else None, "semantic": round(sum(self._error_semantic)/max(len(self._error_semantic),1), 3) if self._error_semantic else None, "pragmatic": round(sum(self._error_pragmatic)/max(len(self._error_pragmatic),1), 3)}}
            })

            # ── 7. PRECISION-WEIGHTING (Critique 2) ──────────
            # observed_intent already computed in step 6
            self._intent_history.append(observed_intent)

            precision, precision_breakdown = compute_precision(
                msg_vec=msg_vec if msg_vec is not None else [],
                recent_vecs=list(self._recent_msg_vecs),
                observed_intent=observed_intent,
                intent_history=list(self._intent_history),
                logprob_variance=logprob_variance
            )
            self._last_precision = precision  # stored for next cycle's feedback
            
            # ── Irony/subtext attenuates precision on literal content ──
            if irony_distance > 0.2:
                sincerity_factor = max(0.5, 1.0 - irony_distance * 0.5)
                precision = round(precision * sincerity_factor, 4)
                precision_breakdown["sincerity_factor"] = round(sincerity_factor, 3)
                precision_breakdown["irony_distance"] = irony_distance
            
            # Add current vec to window AFTER precision is computed
            if msg_vec is not None:
                self._recent_msg_vecs.append(msg_vec)
                # Update conversation centroid with user messages too (Critique 8)
                if self._conversation_centroid is None:
                    self._conversation_centroid = list(msg_vec)
                else:
                    alpha_c = 0.15
                    self._conversation_centroid = [
                        alpha_c * msg_vec[i] + (1 - alpha_c) * self._conversation_centroid[i]
                        for i in range(len(msg_vec))
                    ]

            await broadcast({
                "type": "system_trace", "label": "precision_weighting", "duration_ms": 0,
                "summary": f"Precision = {precision:.3f} ({'amplify' if precision > 1.0 else 'attenuate' if precision < 0.8 else 'normal'})",
                "details": precision_breakdown
            })

            # ── 8. KALMAN UPDATE (proper covariance, Point 4) ──────────
            new_latent, new_P, effective_gain, gain_breakdown = update_latent(
                prior_latent, current_error, obs_valence, obs_velocity,
                precision=precision, P=self._P, Q=self.PROCESS_NOISE
            )
            self.latent, self.latent_prev = new_latent, prior_latent
            self._P = new_P
            self._error_history.append(current_error)
            
            # ── Proper Free Energy (Point 5) ──
            self.free_energy_val = compute_free_energy(current_error, new_latent, self.phenotype_prior, self._P)

            await broadcast({
                "type": "system_trace", "label": "kalman_update", "duration_ms": 0,
                "summary": f"Effective Gain: {effective_gain:.3f} | P: [{self._P[0]:.3f}, {self._P[1]:.3f}, {self._P[2]:.3f}] | FE: {self.free_energy_val:.3f}",
                "details": {**gain_breakdown, "observation": {"surprise": current_error, "valence": obs_valence, "velocity": obs_velocity}, "prior_state": prior_latent, "posterior_state": new_latent, "free_energy": self.free_energy_val}
            })

            # ── 9. ALLOSTATIC LOAD ──────────
            stress_level = self.rolling_error
            allostasis_trace = None
            if stress_level > 0.55:
                self.phenotype_prior["valence"] = round((self.phenotype_prior["valence"] * 0.98) + (-0.20 * 0.02), 4)
                self.phenotype_prior["surprise"] = round((self.phenotype_prior["surprise"] * 0.98) + (0.60 * 0.02), 4)
                allostasis_trace = "Chronic stress: Baseline sinking."
            elif stress_level < 0.35:
                self.phenotype_prior["valence"] = round((self.phenotype_prior["valence"] * 0.98) + (BASE_PRIOR["valence"] * 0.02), 4)
                self.phenotype_prior["surprise"] = round((self.phenotype_prior["surprise"] * 0.98) + (BASE_PRIOR["surprise"] * 0.02), 4)
                allostasis_trace = "Safe environment: Healing toward genotype."

            if allostasis_trace: await broadcast({"type": "system_trace", "label": "allostatic_load", "duration_ms": 0, "summary": allostasis_trace, "details": {"rolling_error": stress_level, "new_phenotype_prior": self.phenotype_prior}})

            # ── 9b. INTEROCEPTIVE STATE (Critique 6) ──────────
            self._last_interoceptive = llm.interoceptive_state
            energy = llm.energy
            if energy < 0.3:
                await broadcast({"type": "system_trace", "label": "interoceptive_fatigue", "duration_ms": 0,
                    "summary": f"Energy depleted: {energy:.2f} — conserving resources",
                    "details": self._last_interoceptive})

            dominant = pred_err_dict.get("dominant_state", self._latent_label())
            pred_err = {"error": current_error, "type": pred_err_dict.get("type", "semantic"), "dominant_state": dominant, "explanation": pred_err_dict.get("explanation", ""), "error_mode": error_mode, "perplexity_used": lexical_error is not None, "precision": precision, "semantic_error": semantic_error, "affective_error": affective_error, "pragmatic_error": pragmatic_error}
            await memory.log_latent(new_latent)
            await memory.log_prediction(self.current_prediction, user_text, current_error, pred_err["type"], dominant, pred_err["explanation"], False)
            await self._save_state(memory)
            if vectors and enriched.get("vector"):
                enc_strength = compute_encoding_strength(current_error, precision, obs_arousal)
                await vectors.store_message(msg_id, user_text, enriched["vector"], "user", new_latent, enriched, encoding_strength=enc_strength)

            # ── FEEDBACK DICT (used by memory encoding and output) ──
            feedback = {
                "valence_misalignment": self._output_valence_alignment,
                "velocity_misalignment": self._output_velocity_alignment,
                "sim_accuracy": self._sim_accuracy_ema,
                "deliberation_gain": self._deliberation_gain_ema,
                "budget_accuracy": llm._budget_accuracy_ema,
                "energy": llm.energy,
                "precision": precision,
                "explore_ratio": self._explore_exploit_ratio,
            }

            # ── 9c. LONG-TERM MEMORY ENCODING ──────────
            # Long-term: LLM interprets the raw text for permanent storage.
            # Working memory (raw conversation) handles short-term context separately.
            # Commitment binding still detects confirmations.
            encode_text = user_text
            is_commitment = False
            
            if (lexical_error is not None and lexical_error < 0.15 
                and self._last_causal_impact > 0.5
                and len(self.conversation) >= 2):
                prev_asst = None
                for turn in reversed(self.conversation[:-1]):
                    if turn.get("role") == "assistant" and turn.get("content", "").strip():
                        prev_asst = turn["content"]
                        break
                if prev_asst:
                    # Keep the user's ACTUAL words. Add a brief context note about
                    # what they're confirming. Never replace user text with assistant text.
                    prev_summary = prev_asst[:80].split('\n')[0].strip()
                    encode_text = f'User said: "{user_text}" (confirming: {prev_summary})'
                    is_commitment = True
                    await broadcast({
                        "type": "system_trace", "label": "commitment_binding", "duration_ms": 0,
                        "summary": f"COMMITMENT: perplexity={lexical_error:.3f} causal_impact={self._last_causal_impact:.3f}",
                        "details": {"perplexity": lexical_error, "causal_impact": self._last_causal_impact,
                                   "user_text": user_text, "confirming": prev_summary}
                    })
            
            input_memory_text = ""
            if vectors:
                input_memory_text = await self._encode_and_store_memory(
                    encode_text, new_latent, llm, memory, vectors, broadcast,
                    mode="input", error=current_error, precision=precision,
                    feedback=feedback, raw_preview=user_text)
                self._diag.set_input_memory(input_memory_text or "")
                
                # ── HyDE Validation for long-term memory ──
                # Ask the LLM "what would you expect about this topic?"
                # If its expectation aligns with the memory → boost encoding strength
                if input_memory_text and len(input_memory_text) > 10:
                    try:
                        hyde_hypothesis = await llm.hyde_validate(input_memory_text, latent=new_latent)
                        if hyde_hypothesis:
                            hyde_vec = await vectors.embed(hyde_hypothesis)
                            mem_vec = await vectors.embed(input_memory_text)
                            hyde_sim = max(0.0, _cosim(hyde_vec, mem_vec))
                            # High sim = aligns with LLM knowledge → boost
                            # Low sim = novel/contradictory → don't punish (yet)
                            if hyde_sim > 0.5:
                                # Memory aligns with world knowledge — mark as validated
                                await broadcast({
                                    "type": "system_trace", "label": "hyde_validation", "duration_ms": 0,
                                    "summary": f"HyDE: sim={hyde_sim:.3f} — memory aligns with LLM knowledge",
                                    "details": {"hyde_sim": round(hyde_sim, 3), "hypothesis": hyde_hypothesis[:80],
                                               "memory": input_memory_text[:80]}
                                })
                    except Exception:
                        pass

            # ── 10. BUILD CONTEXT (Working Memory + Long-Term Retrieval) ──────────
            adapt = self._adaptive_thresholds(energy=energy)
            
            # Boredom computation (unchanged)
            rep_boredom = 0.0
            if self._suppression_history:
                rep_boredom = 1.0 - sum(self._suppression_history) / len(self._suppression_history)
            brev_boredom = 0.0
            current_words = len(user_text.split())
            if self._user_word_ema > 2:
                brev_boredom = max(0.0, 1.0 - current_words / self._user_word_ema)
            nov_boredom = 0.0
            if self._conversation_centroid is not None and msg_vec is not None:
                centroid_sim = max(0.0, _cosim(msg_vec, self._conversation_centroid))
                nov_boredom = centroid_sim
            boredom = round(min(1.0, rep_boredom * 0.5 + brev_boredom * 0.3 + nov_boredom * 0.2), 3)
            
            # ── Long-term retrieval (semantic search in ChromaDB) ──
            lt_context = await self._build_memory_context(
                msg_vec, new_latent, vectors, broadcast, n=8, 
                boredom=boredom, adapt=adapt)
            
            # ── Populate cycle data sources for pipeline resolver ──
            encoded_inputs = await self._get_recent_encoded_memories(vectors, n=10, mode="input")
            encoded_outputs = await self._get_recent_encoded_memories(vectors, n=10, mode="output")
            
            # Tag raw conversation with _order for chronological interleaving
            raw_conv_ordered = []
            base_counter = max(0, self._msg_counter - len(self.conversation[-12:]))
            for i, m in enumerate(self.conversation[-12:]):
                raw_conv_ordered.append({
                    "role": m.get("role", "user"), 
                    "content": m.get("content", ""),
                    "_order": base_counter + i
                })
            
            self._cycle_data = {
                "raw_conversation": raw_conv_ordered,
                "current_message": user_text,
                "encoded_inputs": encoded_inputs,
                "encoded_outputs": encoded_outputs,
                "lt_memories": lt_context,
            }
            
            # ── Working Memory Compression (sources from pipeline config) ──
            s = new_latent.get("surprise", 0.3)
            wm_budget = max(40, min(150, int(60 + s * 120)))
            wm_sources = self.resolve_context("working_memory")
            wm_text = await llm.compress_working_memory(
                wm_sources, new_latent, word_budget=wm_budget,
                state_conditioning=self.stage_uses_state("working_memory"))
            self._cycle_data["working_memory"] = wm_text
            
            # ── Assemble output context from pipeline config ──
            _mc = self.resolve_context("output")
            
            # ── Active Inference (DMN continuation) ──
            active_inference = False
            if boredom > 0.0 and len(_mc) >= 2:
                last = _mc[-1]
                last_words = len(last.get("content", "").split())
                brevity_ratio = last_words / max(self._user_word_ema, 1)
                if last.get("role") == "user" and brevity_ratio < adapt["short_msg_ratio"]:
                    penultimate = _mc[-2]
                    pen_words = len(penultimate.get("content", "").split())
                    if penultimate.get("role") == "assistant" and pen_words > last_words * 3:
                        if boredom > adapt["active_inference_threshold"]:
                            _mc = _mc[:-1]
                            active_inference = True
            
            if active_inference:
                await broadcast({
                    "type": "system_trace", "label": "active_inference", "duration_ms": 0,
                    "summary": f"ACTIVE INFERENCE: boredom={boredom:.2f} > threshold={adapt['active_inference_threshold']:.2f}",
                    "details": {"boredom": boredom, "rep_boredom": round(rep_boredom, 3),
                                "brev_boredom": round(brev_boredom, 3), "nov_boredom": round(nov_boredom, 3),
                                "adaptive": adapt, "context_len": len(_mc)}
                })
            
            self._diag.set_memory_context(_mc)

            # ── 11. PRE-ACTION TRAJECTORY ──
            predict_context = self.resolve_context("predict_pre")
            pre_action_pred = await llm.predict_trajectory(
                predict_context, new_latent, user_model=vectors.get_user_model() if vectors else {},
                state_conditioning=self.stage_uses_state("predict_pre"))
            vec_pre = await vectors.embed(pre_action_pred) if vectors else None
            self._predicted_hyp_vec = vec_pre
            self._predicted_hyp_text = pre_action_pred

            # ── 11b. POLICY SELECTION (Active Inference via Expected Free Energy) ──
            # Evaluate candidate policies: respond, think, recall, silence
            # Think = feed hypothetical as imagined Kalman observation (no new LLM calls)
            # Recall = forage episodic memory for unresolved tensions
            # Silence = skip output, let covariance inflate
            # Respond = full output generation (current default)
            # The loop allows chaining: think → re-evaluate → maybe think again → respond
            
            MAX_THINK_ITERS = 3
            selected_policy = "respond"  # default if policy eval fails
            policy_trace = []
            
            for think_iter in range(MAX_THINK_ITERS + 1):
                policy = self.evaluate_policies(energy=energy, current_error=current_error)
                selected_policy = policy["selected"]
                policy_trace.append({"iter": think_iter, "selected": selected_policy, **policy})
                
                await broadcast({
                    "type": "system_trace", "label": "policy_selection", "duration_ms": 0,
                    "summary": f"Policy: {selected_policy.upper()} (iter {think_iter}) | G: {' '.join(f'{k}={v:.3f}' for k,v in policy['G'].items())}",
                    "details": policy
                })
                
                if selected_policy == "think" and think_iter < MAX_THINK_ITERS:
                    # ── THINK: feed hypothetical as imagined observation ──
                    # Reuses the hypothetical already generated in step 11
                    # No new LLM calls — pure Kalman update with reduced precision
                    if self._predicted_hyp_text and self._predicted_hyp_vec:
                        think_result = await self._execute_think(
                            self._predicted_hyp_text, self._predicted_hyp_vec,
                            new_latent, vectors, broadcast)
                        new_latent = think_result["latent"]
                        self.latent = new_latent
                        self.free_energy_val = think_result["fe"]
                    else:
                        break  # no hypothetical to think about
                    
                elif selected_policy == "recall" and think_iter < MAX_THINK_ITERS:
                    # ── RECALL: forage episodic memory ──
                    found = await self._execute_recall(vectors, memory, broadcast)
                    if not found:
                        break  # nothing to recall, fall through to respond
                    
                else:
                    # respond or silence — exit loop
                    break

            # ── 12. OUTPUT (or silence) ──────────
            reply = ""
            
            if selected_policy == "silence":
                # System chose not to respond — energy recovery, P will inflate next idle tick
                await broadcast({
                    "type": "system_trace", "label": "policy_silence", "duration_ms": 0,
                    "summary": f"System chose SILENCE — EFE favored waiting",
                    "details": {"policy_trace": policy_trace}
                })
                
            else:
                # respond (or fallback from think/recall that didn't resolve)
                reply = await llm.output_module(
                    conversation=_mc, 
                    latent=new_latent, 
                    dynamic_axes=self.dynamic_axes, 
                    error_result=pred_err, 
                    enriched=enriched, 
                    topic_summary=self.topic_summary, 
                    feedback=feedback,
                    user_model=vectors.get_user_model() if vectors else {},
                    state_conditioning=self.stage_uses_state("output"),
                )

            # ── 12a. ENCODE OUTPUT MEMORY ──────────
            output_memory = ""
            if not reply or len(reply.strip()) < 3:
                # Empty/failed response — encode failure so system has memory of it
                reply = ""
                if vectors:
                    try:
                        failure_text = "(System produced no response this turn)"
                        fail_vec = await vectors.embed(failure_text)
                        fail_id = f"ep_output_{self._msg_counter}_{ts().replace(' ','_')}"
                        await vectors.store_episodic_memory(fail_id, failure_text, fail_vec, {
                            "ts": ts(), "msg_counter": self._msg_counter,
                            "memory_type": "output", "latent": dict(new_latent),
                            "covariance": list(self._P), "precision_at": precision,
                            "encoding_strength": 0.6,  # failures are salient
                            "alignment_score": 1.0, "error_at": current_error,
                        })
                        output_memory = failure_text
                    except Exception:
                        pass
            elif vectors:
                try:
                    # Output encoding strength uses INFORMATION DENSITY, not prediction error.
                    # The system isn't surprised by its own output (low error → crushed strength).
                    # Instead: how much new information does this reply contribute?
                    reply_words = len(reply.split())
                    info_density = min(1.0, reply_words / max(self._user_word_ema * 2, 10))
                    output_error = max(current_error, info_density)  # never below info density
                    
                    output_memory = await self._encode_and_store_memory(
                        reply, new_latent, llm, memory, vectors, broadcast,
                        mode="output", error=output_error, precision=precision,
                        feedback=feedback, raw_preview=reply)
                except Exception as e:
                    logger.debug(f"Output memory encoding failed: {e}")

            self._diag.set_output(reply, output_memory or "", posterior_latent=new_latent)

            # ── 12b. POST-ACTION TRAJECTORY & CAUSAL IMPACT ──
            reply_for_prediction = output_memory if output_memory else reply
            prediction_text = await llm.predicted_input_module(
                self.resolve_context("predict_post"), new_latent, reply_for_prediction,
                user_model=vectors.get_user_model() if vectors else {},
                state_conditioning=self.stage_uses_state("predict_post"))
            if prediction_text: 
                self.current_prediction = prediction_text
                vec_post = await vectors.embed(prediction_text) if vectors else None
                
                self._predicted_next_vec = vec_post
                self._predicted_next_text = prediction_text
                
                if vec_pre is not None and vec_post is not None:
                    causal_impact = 1.0 - _cosim(vec_pre, vec_post)
                    self._last_causal_impact = round(causal_impact, 4)
                    await broadcast({
                        "type": "system_trace", "label": "causal_impact", "duration_ms": 0, "summary": f"Action altered user trajectory by {causal_impact:.2f}",
                        "details": { "expected_trajectory_before_action": pre_action_pred, "expected_trajectory_after_action": prediction_text, "causal_impact_score": round(causal_impact, 3) }
                    })

            # ── 12c. COUNTERFACTUAL DEPTH ──────────
            if vectors:
                try:
                    reply_vec = await vectors.embed(reply)
                    
                    if self._conversation_centroid is None:
                        self._conversation_centroid = reply_vec
                    else:
                        alpha_c = 0.15
                        self._conversation_centroid = [
                            alpha_c * reply_vec[i] + (1 - alpha_c) * self._conversation_centroid[i]
                            for i in range(len(reply_vec))
                        ]
                    
                    centroid_dist = round(1.0 - _cosim(reply_vec, self._conversation_centroid), 4)
                    self._explore_exploit_ratio = round(
                        0.3 * min(1.0, centroid_dist / 0.8) + 0.7 * self._explore_exploit_ratio, 4)
                    
                    action_type = "explore" if self._explore_exploit_ratio > 0.55 else "exploit" if self._explore_exploit_ratio < 0.35 else "balanced"
                    await broadcast({
                        "type": "system_trace", "label": "counterfactual_depth", "duration_ms": 0,
                        "summary": f"Action type: {action_type} | Centroid dist = {centroid_dist:.3f} | Ratio = {self._explore_exploit_ratio:.3f}",
                        "details": {"centroid_distance": centroid_dist, "explore_exploit_ratio": self._explore_exploit_ratio, "action_type": action_type}
                    })
                    
                    # ── 12d. SELF-MODEL UPDATE ──────────
                    actual_action, action_scores = await vectors.classify_action(reply_vec)
                    sm_error = self.update_self_model(actual_action)
                    await broadcast({
                        "type": "system_trace", "label": "self_model", "duration_ms": 0,
                        "summary": f"Predicted: {self._last_predicted_action} → Actual: {actual_action} | Self-error = {sm_error:.2f} (EMA = {self._self_model_error:.2f})",
                        "details": {"predicted_action": self._last_predicted_action, "actual_action": actual_action, "self_model_error": sm_error, "self_model_ema": self._self_model_error, "action_scores": action_scores}
                    })

                    # ── 12e. OUTPUT QUALITY MEASUREMENT ──────────
                    reply_valence, reply_arousal, _ = await vectors._score_vad_text(reply, reply_vec)
                    valence_miss = abs(reply_valence - new_latent["valence"])
                    self._output_valence_alignment = round(0.3 * valence_miss + 0.7 * self._output_valence_alignment, 4)
                    
                    reply_words = len(reply.split())
                    budget_stats = llm._last_budget_stats
                    if budget_stats:
                        velocity_miss = min(1.0, abs(1.0 - budget_stats.get("ratio", 1.0)))
                        word_budget = budget_stats.get("budgeted_words", 0)
                    else:
                        s_val = new_latent.get("surprise", 0.3)
                        base_max_tokens = int(50 + (s_val * 200))
                        word_budget = int(base_max_tokens * 0.75)
                        velocity_miss = min(1.0, abs(reply_words - word_budget) / max(word_budget, 1))
                    self._output_velocity_alignment = round(0.3 * velocity_miss + 0.7 * self._output_velocity_alignment, 4)
                    self._output_context_usage = 0.0  # TODO: measure against retrieved memories

                    await broadcast({
                        "type": "system_trace", "label": "output_quality", "duration_ms": 0,
                        "summary": f"Words: {reply_words}/{word_budget} | Budget acc: {llm._budget_accuracy_ema:.2f} | V miss: {valence_miss:.2f}",
                        "details": {
                            "reply_valence": round(reply_valence, 3), "target_valence": new_latent["valence"],
                            "valence_miss_ema": self._output_valence_alignment,
                            "reply_words": reply_words, "word_budget": word_budget,
                            "budget_accuracy_ema": round(llm._budget_accuracy_ema, 3),
                            "velocity_miss_ema": self._output_velocity_alignment,
                        }
                    })
                except Exception:
                    pass

            self.conversation.append({"role": "assistant", "content": reply})
            await memory.append_message("assistant", reply)
            if len(self.conversation) > 32: self.conversation = self.conversation[-32:]

            # ── JEPA + GMM Data Logging ──
            # Log (V_t, action_vec, V_{t+1}) tuples for future predictor training
            # Log output embeddings for GMM action space clustering
            if vectors and reply:
                try:
                    reply_vec = await vectors.embed(reply)
                    if reply_vec is not None:
                        jepa_entry = {
                            "v_t": [prior_latent.get("surprise", 0), prior_latent.get("valence", 0), prior_latent.get("velocity", 0)],
                            "action_vec": reply_vec[:32],  # truncate to 32 dims for storage efficiency
                            "v_t1": [new_latent.get("surprise", 0), new_latent.get("valence", 0), new_latent.get("velocity", 0)],
                            "msg_counter": self._msg_counter,
                        }
                        if not hasattr(self, '_jepa_log'):
                            self._jepa_log = []
                        self._jepa_log.append(jepa_entry)
                        
                        # Full action embedding for GMM clustering
                        if not hasattr(self, '_action_embeddings'):
                            self._action_embeddings = []
                        self._action_embeddings.append({
                            "vec": reply_vec,
                            "latent": [new_latent.get("surprise", 0), new_latent.get("valence", 0), new_latent.get("velocity", 0)],
                            "msg_counter": self._msg_counter,
                        })
                        
                        # Persist to disk every 10 cycles
                        if len(self._jepa_log) % 10 == 0:
                            try:
                                import json as _json
                                with open("jepa_training_data.jsonl", "a") as f:
                                    for entry in self._jepa_log[-10:]:
                                        f.write(_json.dumps(entry) + "\n")
                                with open("action_embeddings.jsonl", "a") as f:
                                    for entry in self._action_embeddings[-10:]:
                                        f.write(_json.dumps({"vec": entry["vec"], "latent": entry["latent"], "mc": entry["msg_counter"]}) + "\n")
                            except Exception:
                                pass
                except Exception:
                    pass

            # ── 13. CAUSAL RELEVANCE SCORING ──────────
            # Did the memories we used actually help predict what happened?
            # If prediction error went DOWN compared to rolling average → memories helped
            # If prediction error went UP → memories were noise/distraction
            # This feedback loop shapes which long-term memories survive
            if vectors and lt_context and current_error is not None:
                try:
                    relevance = max(0.0, 1.0 - current_error)  # low error = memories helped
                    # Compare to baseline: did we do better than usual?
                    baseline_error = self.rolling_error
                    relevance_delta = baseline_error - current_error  # positive = better than baseline
                    
                    if relevance_delta > 0.1:
                        # Memories HELPED — boost encoding strength of retrieved memories
                        boost = min(0.1, relevance_delta * 0.2)
                        for mem_msg in lt_context[:4]:
                            content = mem_msg.get("content", "")[:80]
                            # Find and boost in ChromaDB (best effort)
                            try:
                                results = vectors._episodic_col.query(
                                    query_embeddings=[await vectors.embed(content)],
                                    n_results=1, include=["metadatas"])
                                if results["ids"] and results["ids"][0]:
                                    mid = results["ids"][0][0]
                                    meta = results["metadatas"][0][0]
                                    old_str = float(meta.get("encoding_strength", 0.5))
                                    meta["encoding_strength"] = round(min(1.0, old_str + boost), 4)
                                    vectors._episodic_col.update(ids=[mid], metadatas=[meta])
                            except Exception:
                                pass
                    elif relevance_delta < -0.15:
                        # Memories HURT — degrade encoding strength
                        penalty = min(0.08, abs(relevance_delta) * 0.15)
                        for mem_msg in lt_context[:4]:
                            content = mem_msg.get("content", "")[:80]
                            try:
                                results = vectors._episodic_col.query(
                                    query_embeddings=[await vectors.embed(content)],
                                    n_results=1, include=["metadatas"])
                                if results["ids"] and results["ids"][0]:
                                    mid = results["ids"][0][0]
                                    meta = results["metadatas"][0][0]
                                    old_str = float(meta.get("encoding_strength", 0.5))
                                    meta["encoding_strength"] = round(max(0.01, old_str - penalty), 4)
                                    vectors._episodic_col.update(ids=[mid], metadatas=[meta])
                            except Exception:
                                pass
                    
                    await broadcast({
                        "type": "system_trace", "label": "causal_relevance", "duration_ms": 0,
                        "summary": f"Relevance Δ={relevance_delta:+.3f} | {'BOOST' if relevance_delta > 0.1 else 'DEGRADE' if relevance_delta < -0.15 else 'neutral'} | error={current_error:.3f} baseline={baseline_error:.3f}",
                        "details": {"relevance_delta": round(relevance_delta, 3), 
                                   "current_error": round(current_error, 3),
                                   "baseline_error": round(baseline_error, 3),
                                   "n_lt_memories": len(lt_context)}
                    })
                except Exception as e:
                    logger.debug(f"Causal relevance scoring failed: {e}")

            # ── 14. BACKGROUND TASKS ──────────
            self._summary_counter += 1
            if self._summary_counter % 5 == 0 and len(self.conversation) >= 4:
                asyncio.create_task(llm.topic_summary_module(_mc[-8:], self.latent))
                async def update_axes():
                    if vectors:
                        new_axes = await vectors.compute_dynamic_axes(self.conversation[-8:], self.latent)
                        if new_axes: self.dynamic_axes = new_axes
                asyncio.create_task(update_axes())

            assistant_msg = {"role": "assistant", "content": reply, "ts": ts(), "error": round(current_error, 3), "origin": origin, "enriched": enriched, "prediction_texts": [self.current_prediction]}
            self.messages.append(assistant_msg)
            await broadcast({"type": "assistant_message", "message": assistant_msg, "error_result": pred_err, "data": self.state_snapshot()})

            # ── 14. CYCLE DIAGNOSTIC (module-based) ──────────
            async def run_diagnostic():
                try:
                    report = self._diag.build_report()
                    analysis = await self._diag.analyze(llm)
                    await _original_broadcast({
                        "type": "system_trace", "label": "diagnostic",
                        "duration_ms": 0,
                        "summary": analysis if analysis else "No LLM analysis available",
                        "details": {"report": report, "cycle": self._diag_cycle},
                    })
                except Exception as e:
                    logger.debug(f"Diagnostic failed: {e}")
            asyncio.create_task(run_diagnostic())

        except Exception as e:
            logger.error(f"process_input error: {e}", exc_info=True)
            await _original_broadcast({"type": "error", "message": str(e)})
        finally:
            self.is_processing = False
            await _original_broadcast({"type": "processing", "value": False})