"""
embeddings.py — Full vector encoding layer. v1.1 (Empirical Physics)
"""

import asyncio
import json
import logging
import math
import sqlite3
import hashlib
from collections import deque
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

EMBED_MODEL           = "all-MiniLM-L6-v2"
VAD_MODEL_NAME        = "j-hartmann/emotion-english-distilroberta-base"
CHROMA_PATH           = "./chroma_db"
FTS_DB_PATH           = "./brain.db"
BASE_RETRIEVAL_THRESH = 0.72
RETRIEVAL_N_BASE      = 4
RETRIEVAL_CONCEPT_TYPES = {"factual", "reference", "definition", "event", "person", "place"}
RECALL_SIGNAL_WORDS = set()  # DEPRECATED: epistemic foraging driven by covariance

# ── ACTION ANCHORS (Self-Model, no LLM) ──
ACTION_ANCHORS = {
    "question":     ["What do you think about", "How does that work", "Can you tell me more", "Why do you feel that way", "What would happen if"],
    "reassurance":  ["That makes sense", "I understand", "That's completely valid", "You're on the right track", "I hear you"],
    "challenge":    ["Have you considered", "But what about", "That contradicts", "I'm not sure that follows", "What if the opposite is true"],
    "pivot":        ["Speaking of which", "That reminds me", "On a different note", "Let me shift to", "This connects to"],
    "elaboration":  ["To expand on that", "More specifically", "What I mean is", "The deeper point is", "In other words"],
    "reflection":   ["It sounds like", "What I notice is", "The pattern here", "Looking at this from above", "Stepping back"],
}

# ── Memory Impact Physics ──
def compute_encoding_strength(prediction_error: float, precision: float, arousal: float) -> float:
    """Encoding strength at storage time. High error × high precision × high arousal = deep engram."""
    return round(prediction_error * precision * (0.5 + arousal * 0.5), 4)

def compute_memory_impact(encoding_strength: float, ts_stored: str, latent_stored: dict,
                          latent_current: dict, similarity_to_recent: float) -> tuple[float, dict]:
    """Compute how much a retrieved memory should move the current latent state.
    Content is lossless; impact decays.
    
    Returns (impact, breakdown_dict)."""
    # Temporal decay: Ebbinghaus curve (logarithmic)
    try:
        from datetime import datetime as dt
        stored = dt.strptime(ts_stored, "%Y-%m-%d %H:%M:%S")
        hours_since = max(0.001, (dt.now() - stored).total_seconds() / 3600.0)
    except Exception:
        hours_since = 1.0
    temporal_decay = round(1.0 / (1.0 + math.log1p(hours_since) * 0.15), 4)

    # State resonance: Gaussian falloff on latent distance
    lat_dist = sum((latent_current.get(k, 0) - latent_stored.get(k, 0)) ** 2 for k in ("surprise", "valence", "velocity")) ** 0.5
    state_resonance = round(math.exp(-lat_dist * 2.0), 4)

    # Retrieval surprise: unexpected memories hit harder
    retrieval_surprise = round(max(0.0, 1.0 - similarity_to_recent), 4)

    impact = round(encoding_strength * temporal_decay * state_resonance * (0.5 + retrieval_surprise * 0.5), 4)

    breakdown = {
        "encoding_strength": encoding_strength,
        "temporal_decay": temporal_decay,
        "hours_since": round(hours_since, 1),
        "state_resonance": state_resonance,
        "latent_distance": round(lat_dist, 3),
        "retrieval_surprise": retrieval_surprise,
        "impact": impact,
    }
    return impact, breakdown

# ── PURE VECTOR PROJECTION AXES ──
EMPIRICAL_AXES = {
    "abstraction": {
        "low": ["apple", "car", "walking", "desk", "blue", "yesterday", "code", "invoice", "dog"],
        "high": ["epistemology", "transcendence", "metaphysics", "ontology", "existential", "theoretical"]
    },
    "objectivity": {
        "low": ["I feel", "I hate", "my opinion", "beautiful", "terrible", "I wish", "sadly", "love"],
        "high": ["data shows", "therefore", "mathematically", "evidence", "proves", "statistically"]
    }
}

VAD_ANCHORS = {
    "valence_pos": ["I love this", "this is wonderful", "excellent work", "I am delighted", "this is brilliant", "perfect solution", "I agree completely", "this is beautiful", "outstanding", "I am thrilled"],
    "valence_neg": ["I hate this", "this is terrible", "I am furious", "this is wrong", "I am deeply upset", "this is a disaster", "I am disgusted", "this is awful", "I cannot stand this", "this makes me angry"],
    "arousal_high": ["This is urgent!", "I need this NOW!", "Critical alert!", "Emergency situation!", "Act immediately!", "Crisis mode!", "Extremely excited!", "Can't stop thinking about this!"],
    "arousal_low": ["I am completely relaxed.", "Feeling very calm today.", "No urgency at all.", "Taking things slowly.", "At peace.", "Drowsy and tired.", "Very sleepy.", "Drifting off."]
}
INTENT_ANCHORS = {
    "question": ["what is", "how does", "can you explain", "why does", "tell me about", "I wonder", "do you know"],
    "command": ["do this", "write a", "generate", "create", "make", "build", "execute", "run", "implement", "fix this"],
    "statement": ["I think", "I believe", "it seems", "in my opinion", "I feel", "I know", "I understand", "I see"],
    "acknowledgment": ["okay", "understood", "got it", "I see", "thanks", "thank you", "alright", "sure", "makes sense"],
    "recall": ["what did you say", "you mentioned", "earlier you said", "what was that", "remind me", "you told me"],
    "challenge": ["but why", "I disagree", "that's wrong", "that doesn't make sense", "actually no", "I don't think so"],
    "other": ["hmm", "interesting", "yes", "no", "maybe", "perhaps", "definitely", "absolutely"],
}

class RunningStats:
    def __init__(self, min_samples: int = 15, default: float = BASE_RETRIEVAL_THRESH):
        self.n, self._mean, self._M2, self.min_samples, self._default = 0, 0.0, 0.0, min_samples, default
    def update(self, x: float):
        if x <= 0.0: return
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        self._M2 += delta * (x - self._mean)
    @property
    def std(self) -> float: return math.sqrt(self._M2 / (self.n - 1)) if self.n >= 2 else 0.0
    def threshold(self) -> float: return max(0.58, min(0.90, self._mean + 0.5 * self.std)) if self.n >= self.min_samples else self._default

def _cosim(a: list, b: list) -> float: return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
def _dominant_state(error: float, valence: float, arousal: float) -> str:
    if error < 0.20: return "engagement"
    if error < 0.40: return "engagement" if valence > 0 else "curiosity"
    if error > 0.70: return "confusion" if arousal > 0.55 else "boredom" if valence < -0.25 else "curiosity"
    return "exploration" if arousal > 0.60 else "curiosity" if valence > 0.25 else "exploration"
def _error_type(error, pred_valence, actual_valence, pred_arousal, actual_arousal) -> str:
    if abs(actual_valence - pred_valence) > 0.40: return "emotional"
    if abs(actual_arousal - pred_arousal) < 0.18 and error > 0.55: return "topical"
    return "semantic" if error > 0.40 else "structural"
def _explanation(error: float, intent: str, dominant: str, valence: float) -> str:
    if error < 0.20: return "prediction accurate"
    if error > 0.75: return f"{'negative' if valence < -0.2 else 'positive' if valence > 0.2 else 'neutral'} {intent} far from prediction"
    return f"{intent} shifted toward {dominant}"
def _tone_label(valence: float, arousal: float) -> str:
    if valence > 0.50 and arousal > 0.55: return "excited"
    if valence > 0.50 and arousal < 0.35: return "content"
    if valence > 0.20: return "positive"
    if valence < -0.55 and arousal > 0.55: return "hostile"
    if valence < -0.55 and arousal < 0.35: return "defeated"
    if valence < -0.20: return "negative"
    if valence < -0.10 and arousal > 0.55: return "dismissive"
    return "activated" if arousal > 0.65 else "flat" if arousal < 0.28 else "neutral"

def _has_recall_signal(text: str) -> bool: return any(w in text.lower() for w in RECALL_SIGNAL_WORDS)
def _latent_threshold(base: float, latent: dict) -> float: return max(0.55, min(0.90, base - latent.get("surprise", 0.3) * 0.12 - latent.get("velocity", 0.3) * 0.06 + (0.08 if latent.get("valence", 0.0) < -0.3 else 0)))
def _latent_n(base: int, latent: dict) -> int: return max(2, min(9, base + int(latent.get("surprise", 0.3) * 3) + int(latent.get("velocity", 0.3) * 2) - (2 if latent.get("valence", 0.0) < -0.3 else 0)))
def _retrieval_dampener(retrieved: list) -> float: return max(0.70, 1.0 - (sum(r.get("similarity", 0) for r in retrieved) / len(retrieved)) * 0.30) if retrieved else 1.0

EMOTION_VAD = { "joy": (0.88, 0.72), "surprise": (0.08, 0.78), "neutral": (0.00, 0.20), "sadness": (-0.80, 0.18), "disgust": (-0.68, 0.30), "fear": (-0.62, 0.74), "anger": (-0.70, 0.84) }

class VADModel:
    def __init__(self, model_name: str = VAD_MODEL_NAME):
        self._model_name, self._pipe, self._available = model_name, None, False
    def load(self):
        if not _TRANSFORMERS_AVAILABLE: return
        try:
            self._pipe = hf_pipeline("text-classification", model=self._model_name, top_k=None, truncation=True, max_length=128)
            self._available = True
        except Exception as e:
            logger.warning(f"VAD model failed ({e}) — using anchor fallback.")
            self._available = False
    def score(self, text: str):
        if not self._available or self._pipe is None: return None
        try:
            results = self._pipe(text[:512])
            probs = {r["label"].lower(): r["score"] for r in results[0]}
            valence = arousal = 0.0
            for emotion, prob in probs.items():
                if emotion in EMOTION_VAD:
                    v, a = EMOTION_VAD[emotion]
                    valence += prob * v
                    arousal += prob * a
            return round(valence, 3), round(arousal, 3)
        except Exception: return None

class EmbeddingStore:
    def __init__(self, embed_model=EMBED_MODEL, chroma_path=CHROMA_PATH):
        self._embed_model_name = embed_model
        self._chroma_path = chroma_path
        self._vad_model = VADModel()
        self._client = self._messages_col = self._concepts_col = self._beliefs_col = None
        self._vad_vecs, self._intent_vecs, self._empirical_vecs = {}, {}, {}
        self._action_vecs = {}  # Self-model action anchors
        self._intent_classifier = None  # sklearn LogisticRegression
        self._action_classifier = None  # sklearn LogisticRegression for action types
        self._sim_stats = RunningStats()
        self.concept_mode = "vector"
        self._embed_window = deque(maxlen=6)
        # ── Shadow User Model (Theory of Mind) ──
        self._user_valence_history: deque = deque(maxlen=10)
        self._user_msg_lengths: deque = deque(maxlen=10)
        self._user_intent_history: deque = deque(maxlen=10)
        self._user_question_count: int = 0
        self._user_correction_count: int = 0
        self._user_msg_count: int = 0
        # ── Reconsolidation Bias Tracking ──
        self._recon_delta_vecs: deque = deque(maxlen=30)  # recent diff embeddings
        self._recon_bias_vec: Optional[list] = None  # centroid of deltas = systematic bias

    def calculate_empirical_complexity(self, text: str) -> dict:
        """Pure algorithmic measurement of cognitive load and expected reading time."""
        words = text.split()
        if not words: return {"cognitive_load": 0.1, "complexity_score": 0.1, "reading_time_s": 5, "delay_s": 15}
        
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / word_count
        
        # English avg is ~4.7 chars.
        complexity_score = min(1.0, max(0.1, (avg_word_len - 3.5) / 4.0))
        cognitive_load = min(1.0, (word_count / 50.0) * 0.5 + (complexity_score * 0.5))
        
        # Base read speed ~4 words per second, multiplied by structural complexity.
        base_time = max(5, int(word_count / 4))
        expected_reading_time = int(base_time * (1.0 + complexity_score))
        expected_delay_s = expected_reading_time + max(10, int(cognitive_load * 30))
        
        if _has_recall_signal(text) or any(w in text.lower() for w in ["wait", "brb", "pause", "hold on", "sec"]):
            expected_delay_s = 300
            
        return { "cognitive_load": round(cognitive_load, 3), "complexity_score": round(complexity_score, 3), "reading_time_s": expected_reading_time, "delay_s": expected_delay_s }

    async def init(self):
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, lambda: SentenceTransformer(self._embed_model_name))
        await loop.run_in_executor(None, self._vad_model.load)

        self._client = chromadb.PersistentClient(path=self._chroma_path, settings=Settings(anonymized_telemetry=False))
        self._messages_col = self._client.get_or_create_collection("messages", metadata={"hnsw:space": "cosine"})
        self._concepts_col = self._client.get_or_create_collection("concepts", metadata={"hnsw:space": "cosine"})
        self._beliefs_col  = self._client.get_or_create_collection("core_beliefs", metadata={"hnsw:space": "cosine"})
        self._episodic_col = self._client.get_or_create_collection("episodic_memories", metadata={"hnsw:space": "cosine"})

        for key, phrases in VAD_ANCHORS.items(): self._vad_vecs[key] = await loop.run_in_executor(None, lambda p=phrases: self._model.encode(p, normalize_embeddings=True).tolist())
        for intent, phrases in INTENT_ANCHORS.items(): self._intent_vecs[intent] = await loop.run_in_executor(None, lambda p=phrases: self._model.encode(p, normalize_embeddings=True).tolist())
        
        for axis_name, poles in EMPIRICAL_AXES.items():
            self._empirical_vecs[axis_name] = {
                "low": await loop.run_in_executor(None, lambda p=poles["low"]: self._model.encode(p, normalize_embeddings=True).tolist()),
                "high": await loop.run_in_executor(None, lambda p=poles["high"]: self._model.encode(p, normalize_embeddings=True).tolist())
            }

        # Self-model action anchors
        for action, phrases in ACTION_ANCHORS.items():
            if phrases:
                self._action_vecs[action] = await loop.run_in_executor(None, lambda p=phrases: self._model.encode(p, normalize_embeddings=True).tolist())

        # ── Train sklearn classifiers (Option B: dedicated classifiers) ──
        if _SKLEARN_AVAILABLE:
            def _train_classifier(anchor_dict):
                X, y = [], []
                for label, phrases in anchor_dict.items():
                    if not phrases: continue
                    vecs = self._model.encode(phrases, normalize_embeddings=True).tolist()
                    X.extend(vecs)
                    y.extend([label] * len(vecs))
                if len(set(y)) < 2: return None
                clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
                clf.fit(np.array(X), y)
                return clf
            self._intent_classifier = await loop.run_in_executor(None, lambda: _train_classifier(INTENT_ANCHORS))
            self._action_classifier = await loop.run_in_executor(None, lambda: _train_classifier(ACTION_ANCHORS))
            if self._intent_classifier: logger.info("Intent classifier trained (sklearn).")
            if self._action_classifier: logger.info("Action classifier trained (sklearn).")

        await self._init_fts()
        logger.info("Vector store ready.")

    async def _init_fts(self):
        def sync_fts():
            conn = sqlite3.connect(FTS_DB_PATH)
            conn.executescript("CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(content, content=\"conversation_history\", content_rowid=\"id\"); INSERT INTO conversation_fts(conversation_fts) VALUES('rebuild');")
            conn.commit(); conn.close()
        await asyncio.get_event_loop().run_in_executor(None, sync_fts)

    async def _fts_search(self, query: str, n: int = 4) -> list[dict]:
        def sync_search():
            tokens = [w for w in query.lower().split() if w not in RECALL_SIGNAL_WORDS and len(w) > 2]
            if not tokens: return []
            try:
                conn = sqlite3.connect(FTS_DB_PATH)
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT ch.id, ch.ts, ch.role, ch.content FROM conversation_fts fts JOIN conversation_history ch ON ch.id = fts.rowid WHERE conversation_fts MATCH ? ORDER BY rank LIMIT ?", (" OR ".join(tokens), n)).fetchall()
                conn.close()
                return [{"text": r["content"], "role": r["role"], "ts": r["ts"], "similarity": 1.0, "source": "exact", "latent": {}} for r in rows]
            except Exception: return []
        return await asyncio.get_event_loop().run_in_executor(None, sync_search)

    async def embed(self, text: str) -> list:
        return await asyncio.get_event_loop().run_in_executor(None, lambda: self._model.encode(text, normalize_embeddings=True).tolist())

    def semantic_velocity(self, new_vec: list) -> float:
        window = list(self._embed_window)
        if len(window) < 2:
            self._embed_window.append(new_vec)
            return 0.3 
        dists = [1.0 - _cosim(window[i], window[i+1]) for i in range(len(window) - 1)]
        velocity = min(1.0, (sum(dists) / len(dists)) / 0.45)
        self._embed_window.append(new_vec)
        return round(velocity, 4)

    async def _score_vad_text(self, text: str, vec: list) -> tuple[float, float, str]:
        """VAD scoring: distilroberta exclusively. Falls back to neutral if unavailable."""
        res = self._vad_model.score(text)
        if res is not None:
            return res[0], res[1], "affective"
        return 0.0, 0.5, "unavailable"

    async def _classify_intent(self, vec: list) -> dict:
        """Intent classification: sklearn LogisticRegression with proper decision boundaries."""
        if self._intent_classifier and _SKLEARN_AVAILABLE:
            try:
                probs = self._intent_classifier.predict_proba(np.array([vec]))[0]
                classes = self._intent_classifier.classes_
                scored = sorted(zip(classes, probs), key=lambda x: -x[1])
                best_label, best_prob = scored[0]
                return {"label": str(best_label), "best_score": round(float(best_prob), 3),
                        "breakdown": {str(k): round(float(v), 3) for k, v in scored}}
            except Exception:
                pass
        # Fallback to cosine anchors if sklearn unavailable
        scores = {intent: sum(_cosim(vec, v) for v in vecs) / len(vecs) for intent, vecs in self._intent_vecs.items() if vecs}
        if not scores: return {"label": "other", "breakdown": {}}
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        best_intent, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        label = "other" if (best_score - second_score) < 0.003 else best_intent
        return {"label": label, "best_score": round(best_score, 3), "breakdown": {k: round(v, 3) for k, v in sorted_scores}}

    async def store_belief(self, belief_text: str):
        vec = await self.embed(belief_text)
        b_id = hashlib.md5(belief_text.encode()).hexdigest()
        try: self._beliefs_col.upsert(ids=[b_id], embeddings=[vec], documents=[belief_text])
        except Exception: pass

    def belief_redundancy_count(self) -> int:
        """Count approximate number of redundant belief pairs (sim > 0.80).
        Used by internal action EFE to determine replay urgency."""
        if not self._beliefs_col or self._beliefs_col.count() < 2:
            return 0
        try:
            all_b = self._beliefs_col.get(include=["embeddings"])
            vecs = all_b["embeddings"]
            count = 0
            for i in range(min(len(vecs), 20)):  # cap at 20 to avoid O(n²) blowup
                for j in range(i + 1, min(len(vecs), 20)):
                    if _cosim(vecs[i], vecs[j]) > 0.80:
                        count += 1
            return count
        except Exception:
            return 0

    def max_concept_uncertainty(self) -> float:
        """Quick check: how uncertain is the most uncertain concept?
        Used by internal action EFE for forage urgency."""
        # This delegates to memory.get_most_uncertain_concept, but we can
        # provide a sync estimate from the concept collection's metadata
        return 0.5  # fallback; actual check happens via memory in brain.py

    async def _extract_concepts(self, vec: list, n: int = 5) -> list[dict]:
        if self._concepts_col.count() == 0: return []
        try:
            results = self._concepts_col.query(query_embeddings=[vec], n_results=min(n, self._concepts_col.count()), include=["documents", "metadatas", "distances"])
            return [{"label": doc, "type": meta.get("type","unknown"), "similarity": round(1.0 - dist, 3), "source": "seed" if meta.get("message_id") == "seed" else "vector"} for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]) if (1.0 - dist) > 0.55]
        except Exception: return []

    async def _top_similarity(self, vec: list) -> float:
        if self._messages_col.count() == 0: return 0.0
        try:
            try: r = self._messages_col.query(query_embeddings=[vec], n_results=1, where={"role": "user"}, include=["distances"])
            except Exception: r = self._messages_col.query(query_embeddings=[vec], n_results=1, include=["distances"])
            if r["distances"] and r["distances"][0]:
                sim = round(1.0 - r["distances"][0][0], 3)
                self._sim_stats.update(sim)
                return sim
        except Exception:
            pass  # HNSW index not ready yet
        return 0.0

    async def _semantic_retrieve(self, vec: list, n: int, latent: dict) -> list[dict]:
        if self._messages_col.count() == 0: return []
        try:
            try: results = self._messages_col.query(query_embeddings=[vec], n_results=min(n * 2, self._messages_col.count()), where={"role": "user"}, include=["documents", "metadatas", "distances"])
            except Exception: results = self._messages_col.query(query_embeddings=[vec], n_results=min(n, self._messages_col.count()), include=["documents", "metadatas", "distances"])
            retrieved = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                retrieved.append({
                    "text": doc, "ts": meta.get("ts",""), "role": meta.get("role","user"),
                    "similarity": round(1.0 - dist, 3),
                    "latent": json.loads(meta.get("latent_json", "{}")),
                    "encoding_strength": float(meta.get("encoding_strength", 0.5)),
                    "source": "semantic"
                })
            return sorted(retrieved, key=lambda r: r.get("ts", ""))[:n] if (latent.get("surprise", 0.5) < 0.25 and latent.get("velocity", 0.5) < 0.25) else sorted(retrieved, key=lambda r: -r["similarity"])[:n]
        except Exception:
            return []  # HNSW index not ready yet

    async def classify_action(self, reply_vec: list) -> tuple[str, dict]:
        """Classify a system reply into an action type. sklearn with cosine fallback."""
        if self._action_classifier and _SKLEARN_AVAILABLE:
            try:
                probs = self._action_classifier.predict_proba(np.array([reply_vec]))[0]
                classes = self._action_classifier.classes_
                scored = sorted(zip(classes, probs), key=lambda x: -x[1])
                return str(scored[0][0]), {str(k): round(float(v), 3) for k, v in scored}
            except Exception:
                pass
        scores = {}
        for action, vecs in self._action_vecs.items():
            if vecs:
                scores[action] = sum(_cosim(reply_vec, v) for v in vecs) / len(vecs)
        if not scores:
            return "other", {}
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_scores[0][0], {k: round(v, 3) for k, v in sorted_scores}

    # ── Shadow User Model (Theory of Mind) ──

    def update_user_model(self, text: str, valence: float, intent: str):
        """Update user state tracking from observable signals. No LLM inference."""
        self._user_msg_count += 1
        self._user_valence_history.append(valence)
        self._user_msg_lengths.append(len(text.split()))
        self._user_intent_history.append(intent)
        if intent in ("question", "recall"): self._user_question_count += 1
        if intent in ("challenge", "correction"): self._user_correction_count += 1

    def get_user_model(self) -> dict:
        """Computed user state from observables."""
        n = max(self._user_msg_count, 1)
        vals = list(self._user_valence_history)
        lengths = list(self._user_msg_lengths)
        intents = list(self._user_intent_history)
        
        # Valence slope (linear regression over last 5)
        valence_slope = 0.0
        if len(vals) >= 3:
            recent = vals[-5:]
            x_mean = (len(recent) - 1) / 2
            y_mean = sum(recent) / len(recent)
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
            den = sum((i - x_mean) ** 2 for i in range(len(recent)))
            valence_slope = round(num / den, 4) if den > 0 else 0.0
        
        # Length trend
        length_trend = 0.0
        if len(lengths) >= 3:
            recent_l = lengths[-5:]
            length_trend = round((recent_l[-1] - recent_l[0]) / max(len(recent_l), 1), 2)
        
        # Intent entropy (high = scattered)
        intent_entropy = 0.0
        if intents:
            from collections import Counter as C
            counts = C(intents)
            total = len(intents)
            intent_entropy = round(-sum((c/total) * math.log(c/total + 1e-10) for c in counts.values()), 3)
        
        # Semantic consistency (from embed_window)
        semantic_consistency = 0.5
        window = list(self._embed_window)
        if len(window) >= 2:
            sims = [_cosim(window[i], window[i+1]) for i in range(len(window)-1)]
            semantic_consistency = round(sum(sims) / len(sims), 3)
        
        # Engagement = length_trend_positive × consistency × (1 - entropy)
        engagement = round(max(0.0, min(1.0, 
            (1.0 if length_trend >= 0 else 0.5) * semantic_consistency * (1.0 - min(intent_entropy / 2.0, 1.0))
        )), 3)
        
        # Confusion = high question rate + dropping complexity + low consistency
        question_rate = round(self._user_question_count / n, 3)
        confusion = round(max(0.0, min(1.0, question_rate * 0.5 + (1.0 - semantic_consistency) * 0.3 + min(intent_entropy / 2.0, 0.2))), 3)
        
        return {
            "valence_current": round(vals[-1], 3) if vals else 0.0,
            "valence_slope": valence_slope,
            "engagement": engagement,
            "confusion": confusion,
            "question_rate": question_rate,
            "correction_rate": round(self._user_correction_count / n, 3),
            "length_trend": length_trend,
            "intent_entropy": intent_entropy,
            "semantic_consistency": semantic_consistency,
            "n_messages": self._user_msg_count,
            "_lengths": list(self._user_msg_lengths),  # raw word counts for boredom computation
        }

    # ── Conflict Detection (for Dreams) ──

    async def find_conflicting_memories(self, n_pairs: int = 3) -> list:
        """Find memory pairs with high semantic similarity but conflicting valence.
        These represent unresolved tensions — ideal targets for dream synthesis."""
        if not self._episodic_col or self._episodic_col.count() < 4:
            return []
        try:
            all_mems = self._episodic_col.get(
                include=["documents", "metadatas", "embeddings"],
                limit=50  # cap to avoid O(n²) blowup
            )
            docs = all_mems.get("documents", [])
            metas = all_mems.get("metadatas", [])
            vecs = all_mems.get("embeddings", [])
            if not docs or len(docs) < 4:
                return []
            
            conflicts = []
            for i in range(len(docs)):
                latent_i = json.loads(metas[i].get("latent_json", "{}"))
                v_i = latent_i.get("valence", 0.0)
                for j in range(i + 1, min(len(docs), 30)):
                    sim = _cosim(vecs[i], vecs[j])
                    if sim < 0.6:
                        continue
                    latent_j = json.loads(metas[j].get("latent_json", "{}"))
                    v_j = latent_j.get("valence", 0.0)
                    valence_conflict = abs(v_i - v_j)
                    if valence_conflict > 0.3:
                        conflicts.append({
                            "texts": [docs[i], docs[j]],
                            "similarity": round(sim, 3),
                            "valence_conflict": round(valence_conflict, 3),
                            "tension_score": round(sim * valence_conflict, 4),
                        })
            
            conflicts.sort(key=lambda c: -c["tension_score"])
            return conflicts[:n_pairs]
        except Exception:
            return []

    # ── Reconsolidation Bias Tracking ──

    async def track_reconsolidation_delta(self, diff_text: str):
        """Embed a reconsolidation diff and accumulate into the bias vector."""
        if not diff_text:
            return
        diff_vec = await self.embed(diff_text)
        self._recon_delta_vecs.append(diff_vec)
        # Update centroid
        if self._recon_delta_vecs:
            n = len(self._recon_delta_vecs)
            dim = len(diff_vec)
            self._recon_bias_vec = [
                sum(v[d] for v in self._recon_delta_vecs) / n for d in range(dim)
            ]

    def get_reconsolidation_bias(self) -> dict:
        """Return the current systematic reinterpretation bias."""
        if not self._recon_bias_vec or len(self._recon_delta_vecs) < 3:
            return {"direction": "none", "magnitude": 0.0, "n_samples": len(self._recon_delta_vecs)}
        # Score the bias vector against VAD anchors to determine direction
        bias_valence = 0.0
        if self._vad_vecs.get("valence_pos") and self._vad_vecs.get("valence_neg"):
            pos_sim = sum(_cosim(self._recon_bias_vec, v) for v in self._vad_vecs["valence_pos"]) / len(self._vad_vecs["valence_pos"])
            neg_sim = sum(_cosim(self._recon_bias_vec, v) for v in self._vad_vecs["valence_neg"]) / len(self._vad_vecs["valence_neg"])
            bias_valence = round(pos_sim - neg_sim, 3)
        magnitude = round(math.sqrt(sum(v*v for v in self._recon_bias_vec)), 4)
        direction = "positive" if bias_valence > 0.05 else "negative" if bias_valence < -0.05 else "neutral"
        return {"direction": direction, "valence": bias_valence, "magnitude": magnitude, "n_samples": len(self._recon_delta_vecs)}

    # ── Episodic Memory System ──

    async def score_memory_alignment(self, memory_text: str, memory_vec: list, target_latent: dict) -> float:
        """Check if the memory text's emotional content matches the system state at encoding.
        Returns alignment score [0, 1]. Below 0.35 = misencoded."""
        mem_valence, mem_arousal, _ = await self._score_vad_text(memory_text, memory_vec)
        target_v = target_latent.get("valence", 0.0)
        target_s = target_latent.get("surprise", 0.3)
        # Valence alignment (60% weight) and arousal/surprise alignment (40%)
        valence_err = abs(mem_valence - target_v)
        arousal_err = abs(mem_arousal - target_s)
        alignment = max(0.0, 1.0 - (0.6 * valence_err + 0.4 * arousal_err))
        return round(alignment, 3)

    async def store_episodic_memory(self, mem_id: str, text: str, vector: list, metadata: dict):
        """Store episodic memory with entropy-based decay of overlapping memories.
        
        When a new memory is encoded near an old one in embedding space, the old
        memory's encoding_strength (precision) degrades. This replaces temporal
        Ebbinghaus decay with interference-based decay:
        
        Π_old = Π_old - cosine_sim(new, old) × enc_strength_new
        
        A memory from an hour ago remains sharp if nothing similar overwrites it.
        "Continue" encoded 5 times crushes all previous "Continue" memories.
        """
        try:
            new_enc_strength = float(metadata.get("encoding_strength", 0.5))
            
            # ── Entropy decay: degrade overlapping memories ──
            if self._episodic_col.count() > 0:
                try:
                    nearby = self._episodic_col.query(
                        query_embeddings=[vector], n_results=min(5, self._episodic_col.count()),
                        include=["metadatas", "distances"])
                    if nearby["ids"] and nearby["ids"][0]:
                        for nid, nmeta, ndist in zip(nearby["ids"][0], nearby["metadatas"][0], nearby["distances"][0]):
                            if nid == mem_id:
                                continue  # don't decay self
                            sim = max(0.0, 1.0 - ndist)
                            if sim > 0.5:  # only degrade substantially overlapping memories
                                old_str = float(nmeta.get("encoding_strength", 0.5))
                                decay = sim * new_enc_strength * 0.3  # interference proportional to overlap × new strength
                                new_str = round(max(0.01, old_str - decay), 4)
                                if new_str < old_str:
                                    nmeta["encoding_strength"] = new_str
                                    self._episodic_col.update(ids=[nid], metadatas=[nmeta])
                except Exception:
                    pass  # decay is best-effort, never block encoding
            
            chroma_meta = {
                "ts": metadata.get("ts", ""),
                "msg_counter": int(metadata.get("msg_counter", 0)),
                "memory_type": metadata.get("memory_type", "input"),
                "latent_json": json.dumps(metadata.get("latent", {})),
                "covariance_json": json.dumps(metadata.get("covariance", [])),
                "encoding_strength": new_enc_strength,
                "alignment_score": float(metadata.get("alignment_score", 0.0)),
                "precision_at": float(metadata.get("precision_at", 1.0)),
                "error_at": float(metadata.get("error_at", 0.5)),
                "recall_count": 0,
                "last_recall_ts": "",
                "original_valence": float(metadata.get("latent", {}).get("valence", 0.0)),
                "valence_drift": 0.0,
            }
            self._episodic_col.upsert(ids=[mem_id], embeddings=[vector], documents=[text], metadatas=[chroma_meta])
        except Exception as e:
            logger.warning(f"Episodic store failed: {e}")

    async def reconsolidate_memory(self, mem_id: str, new_text: str, new_vector: list, 
                                    new_valence: float, valence_drift: float, recall_count: int):
        """Update a memory in ChromaDB after reconsolidation."""
        try:
            # Get existing metadata
            existing = self._episodic_col.get(ids=[mem_id], include=["metadatas"])
            if existing["metadatas"]:
                meta = existing["metadatas"][0]
                meta["recall_count"] = recall_count
                meta["last_recall_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                meta["valence_drift"] = float(valence_drift)
                # Update latent valence to reconsolidated value
                latent = json.loads(meta.get("latent_json", "{}"))
                latent["valence"] = new_valence
                meta["latent_json"] = json.dumps(latent)
                self._episodic_col.update(ids=[mem_id], embeddings=[new_vector], documents=[new_text], metadatas=[meta])
        except Exception as e:
            logger.debug(f"Reconsolidation ChromaDB update failed: {e}")

    async def retrieve_episodic_memories(self, query_vec: list, current_latent: dict, 
                                         current_msg_counter: int = 0, n: int = 8,
                                         boredom: float = 0.0, adapt: dict = None) -> list:
        """Unified episodic retrieval with adaptive boredom-driven class weighting.
        
        adapt dict provides dynamic factors from system state:
          self_boost_factor: multiplier for self-memories when bored (from precision/novelty)
          ext_dampen_factor: dampening rate for external memories (from precision)
          boredom_threshold: when weighting activates (from fatigue/novelty)
          lateral_inhibition_sim: adaptive similarity threshold
          habituation_fast_s / habituation_slow_s: adaptive habituation windows
        """
        if not self._episodic_col or self._episodic_col.count() == 0:
            return []
        try:
            count = self._episodic_col.count()
            
            # ── Track 1: Semantic retrieval (cosine similarity to query) ──
            results = self._episodic_col.query(
                query_embeddings=[query_vec],
                n_results=min(n * 4, count),
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            if not results["documents"] or not results["documents"][0]:
                return []
            
            # ── Track 2: Temporal retrieval (most recent memories regardless of similarity) ──
            # Ensures conversation continuity even when topics shift.
            # Get ALL memories, sort by msg_counter, inject recent ones into results.
            semantic_ids = set(results["ids"][0]) if results["ids"] else set()
            n_temporal = min(4, count)  # inject up to 4 recent memories
            
            try:
                all_mems = self._episodic_col.get(
                    include=["documents", "metadatas", "embeddings"])
                if all_mems and all_mems["ids"]:
                    # Sort by msg_counter descending
                    temporal_items = sorted(
                        zip(all_mems["ids"], all_mems["documents"], 
                            all_mems["metadatas"], all_mems["embeddings"] or [None]*len(all_mems["ids"])),
                        key=lambda x: -int(x[2].get("msg_counter", 0)))
                    
                    # Inject temporal items not already in semantic results
                    injected = 0
                    for tid, tdoc, tmeta, temb in temporal_items:
                        if injected >= n_temporal:
                            break
                        if tid not in semantic_ids:
                            results["ids"][0].append(tid)
                            results["documents"][0].append(tdoc)
                            results["metadatas"][0].append(tmeta)
                            # Compute actual distance for injected items
                            if temb is not None and query_vec is not None:
                                dist = 1.0 - sum(float(a)*float(b) for a,b in zip(query_vec, temb))
                            else:
                                dist = 1.0  # max distance if no embedding
                            results["distances"][0].append(max(0, dist))
                            if results.get("embeddings") and results["embeddings"][0] is not None:
                                results["embeddings"][0].append(temb)
                            injected += 1
            except Exception:
                pass  # temporal track is best-effort
        except Exception:
            return []
        
        cur_s = current_latent.get("surprise", 0.3)
        cur_v = current_latent.get("valence", 0.0)
        cur_vel = current_latent.get("velocity", 0.3)
        now = datetime.now()
        adapt = adapt or {}
        
        # Adaptive thresholds (with defaults for when adapt is empty)
        boredom_threshold = adapt.get("boredom_threshold", 0.3)
        self_boost = adapt.get("self_boost_factor", 1.5)
        ext_dampen = adapt.get("ext_dampen_factor", 0.45)
        hab_fast_s = adapt.get("habituation_fast_s", 120)
        hab_slow_s = adapt.get("habituation_slow_s", 600)
        
        scored = []
        raw_embeddings = results.get("embeddings", [[]])[0]
        embeddings_list = raw_embeddings if raw_embeddings is not None and len(raw_embeddings) > 0 else [None] * len(results["documents"][0])
        for doc, meta, dist, mid, emb in zip(
            results["documents"][0], results["metadatas"][0], 
            results["distances"][0], results["ids"][0],
            embeddings_list
        ):
            sim = round(1.0 - dist, 3)
            if sim < 0.25:
                continue
            
            stored_latent = json.loads(meta.get("latent_json", "{}"))
            stored_s = stored_latent.get("surprise", 0.3)
            stored_v = stored_latent.get("valence", 0.0)
            stored_vel = stored_latent.get("velocity", 0.3)
            stored_cov = json.loads(meta.get("covariance_json", "[]")) or [0.5, 0.5, 0.5]
            stored_msg_counter = int(meta.get("msg_counter", 0))
            
            # ── Retrievability: encoding_strength × recency × precision ──
            enc_strength = float(meta.get("encoding_strength", 0.5))
            precision_at = float(meta.get("precision_at", 1.0))
            turns_since = max(0, current_msg_counter - stored_msg_counter)
            recency = round(math.exp(-turns_since * 0.3), 4)  # 1 turn=0.74, 3=0.41, 6=0.17
            precision_factor = min(1.0, precision_at * 0.8)
            retrievability = enc_strength * recency * precision_factor
            
            # ── State distance + resonance ──
            state_dist = math.sqrt(
                (cur_s - stored_s) ** 2 + (cur_v - stored_v) ** 2 + (cur_vel - stored_vel) ** 2
            )
            resonance = math.exp(-state_dist ** 2 / (2 * 0.3 ** 2))
            
            # ── Contrast ──
            valence_delta = cur_v - stored_v
            valence_flip = abs(valence_delta) > 0.4 and (cur_v * stored_v < 0)
            contrast_boost = 0.3 if valence_flip else 0.0
            
            # ── Recall dynamics ──
            recall_count = int(meta.get("recall_count", 0))
            frequency_amplifier = 1.0 + math.log1p(recall_count) * 0.15
            
            habituation = 1.0
            last_recall_str = meta.get("last_recall_ts", "")
            if last_recall_str:
                try:
                    last_recall = datetime.strptime(last_recall_str, "%Y-%m-%d %H:%M:%S")
                    seconds_since = (now - last_recall).total_seconds()
                    if seconds_since < hab_fast_s:     habituation = 0.3
                    elif seconds_since < hab_slow_s:   habituation = 0.7
                except Exception:
                    pass
            
            # ── Boredom-driven class weighting (DMN) ──
            # Uses adaptive factors from system state:
            #   self_boost: amplification for own thoughts (driven by precision/novelty)
            #   ext_dampen: dampening for external input (driven by precision)
            #   boredom_threshold: when weighting activates (driven by fatigue/novelty)
            memory_type = meta.get("memory_type", "input")
            if boredom > boredom_threshold:
                if memory_type == "output":
                    class_weight = 1.0 + boredom * self_boost
                else:
                    class_weight = max(0.3, 1.0 - boredom * ext_dampen)
            else:
                class_weight = 1.0
            
            # ── Final retrieval score ──
            retrieval_score = round(
                sim * retrievability * (resonance + contrast_boost) * frequency_amplifier * habituation * class_weight, 4)
            
            fidelity = min(1.0, retrieval_score)
            
            # Encoding confidence (for affective bleed)
            mean_cov = sum(stored_cov) / max(len(stored_cov), 1) if stored_cov else 0.5
            encoding_confidence = 1.0 / (1.0 + mean_cov)
            
            original_valence = float(meta.get("original_valence", stored_v))
            valence_drift_stored = float(meta.get("valence_drift", 0.0))
            
            scored.append({
                "mem_id": mid,
                "text": doc,
                "memory_type": meta.get("memory_type", "input"),
                "msg_counter": stored_msg_counter,
                "similarity": sim,
                "retrievability": round(retrievability, 4),
                "recency": round(recency, 3),
                "turns_since": turns_since,
                "retrieval_score": retrieval_score,
                "fidelity": round(fidelity, 3),
                "state_distance": round(state_dist, 3),
                "resonance": round(resonance, 3),
                "valence_flip": valence_flip,
                "contrast_boost": contrast_boost,
                "encoding_strength": enc_strength,
                "encoding_confidence": round(encoding_confidence, 3),
                "stored_latent": stored_latent,
                "valence_delta": round(valence_delta, 3),
                "recall_count": recall_count,
                "frequency_amplifier": round(frequency_amplifier, 3),
                "habituation": round(habituation, 3),
                "original_valence": original_valence,
                "valence_drift": valence_drift_stored,
                "precision_at": round(precision_at, 3),
                "class_weight": round(class_weight, 3),
                "ts": meta.get("ts", ""),
                "_vec": emb,  # stored for lateral inhibition, stripped before return
            })
        
        # ── Lateral Inhibition (Hartline 1949) ──
        # Adaptive threshold: exploring system → lower threshold → more diversity
        lat_inhib_sim = adapt.get("lateral_inhibition_sim", 0.85)
        scored.sort(key=lambda r: -r["retrieval_score"])
        selected = []
        for candidate in scored:
            if len(selected) >= n:
                break
            c_vec = candidate.get("_vec")
            c_text = candidate.get("text", "")
            
            inhibited = False
            for already in selected:
                a_vec = already.get("_vec")
                if c_vec is not None and a_vec is not None:
                    try:
                        sim_to_selected = sum(float(a) * float(b) for a, b in zip(c_vec, a_vec))
                        if sim_to_selected > lat_inhib_sim:
                            inhibited = True
                            break
                    except Exception:
                        pass
                if not inhibited and c_text and already.get("text", ""):
                    if c_text[:40] == already["text"][:40]:
                        inhibited = True
                        break
            if not inhibited:
                selected.append(candidate)
        
        # _vec kept for downstream spread computation (attentional inertia)
        return selected

    async def store_message(self, message_id: str, text: str, vector: list, role: str, latent: dict, enriched: dict, encoding_strength: float = 0.5):
        tone, concepts = enriched.get("tone", {}), enriched.get("concepts", [])
        metadata = {
            "ts": enriched.get("ts",""), "role": role, "latent_json": json.dumps(latent),
            "concepts_json": json.dumps([c.get("label","") for c in concepts if isinstance(c,dict)]),
            "tone_label": tone.get("label",""), "tone_valence": float(tone.get("valence", 0.0)), "tone_arousal": float(tone.get("arousal", 0.5)), "intent": enriched.get("intent",""),
            "encoding_strength": float(encoding_strength),
        }
        try: self._messages_col.upsert(ids=[message_id], embeddings=[vector], documents=[text], metadatas=[metadata])
        except Exception: pass

    async def compute_dynamic_axes(self, recent_messages: list, latent: dict) -> dict:
        """
        Replace LLM-based axis induction with vector-space computation.
        1. Start with empirical seed axes (abstraction, objectivity).
        2. Evolve by projecting recent concept clusters as emergent poles.
        """
        if not recent_messages:
            return {}

        # ── Seed axes from EMPIRICAL_AXES ──
        recent_vecs = []
        for m in recent_messages[-8:]:
            try:
                v = await self.embed(m["content"][:300])
                recent_vecs.append(v)
            except Exception:
                continue
        if not recent_vecs:
            return {}

        centroid = [sum(v[i] for i in range(len(v))) / len(recent_vecs) 
                    for v in zip(*recent_vecs)]
        
        axes = {}
        for axis_name, pole_vecs in self._empirical_vecs.items():
            sim_low  = sum(_cosim(centroid, v) for v in pole_vecs["low"])  / len(pole_vecs["low"])
            sim_high = sum(_cosim(centroid, v) for v in pole_vecs["high"]) / len(pole_vecs["high"])
            axes[axis_name] = round(math.tanh((sim_high - sim_low) * 10), 3)

        # ── Emergent axes from concept clusters ──
        if self._concepts_col and self._concepts_col.count() > 0:
            try:
                results = self._concepts_col.query(
                    query_embeddings=[centroid],
                    n_results=min(10, self._concepts_col.count()),
                    include=["documents", "embeddings", "distances"]
                )
                concept_vecs = results.get("embeddings", [[]])[0]
                concept_labels = results.get("documents", [[]])[0]
                concept_dists = results.get("distances", [[]])[0]

                # Filter to relevant concepts (similarity > 0.45)
                active = [(label, vec, 1.0 - dist) 
                          for label, vec, dist in zip(concept_labels, concept_vecs, concept_dists)
                          if (1.0 - dist) > 0.45]

                if len(active) >= 4:
                    # Split into two clusters by projecting onto the first principal direction
                    # (difference between most and least similar to conversation centroid)
                    active.sort(key=lambda x: x[2])
                    pole_a = active[:len(active)//2]   # distant concepts
                    pole_b = active[len(active)//2:]   # proximal concepts

                    vec_a = [sum(v[i] for _, v, _ in pole_a) / len(pole_a) 
                             for i in range(len(pole_a[0][1]))]
                    vec_b = [sum(v[i] for _, v, _ in pole_b) / len(pole_b) 
                             for i in range(len(pole_b[0][1]))]

                    # Name the axis from the pole labels
                    label_a = pole_a[-1][0]  # closest in the distant group
                    label_b = pole_b[0][0]   # closest in the proximal group
                    axis_key = f"{label_a}_vs_{label_b}"

                    # Score: where does the conversation centroid sit between poles?
                    sim_a = _cosim(centroid, vec_a)
                    sim_b = _cosim(centroid, vec_b)
                    axes[axis_key] = round(math.tanh((sim_b - sim_a) * 8), 3)

            except Exception:
                pass

        return axes

    async def compute_epistemic_value(self, candidate_vec: list, recent_assistant_vecs: list) -> float:
        """
        Epistemic value = how far this candidate pulls from the centroid
        of recent assistant outputs. High distance = high novelty.
        """
        if not recent_assistant_vecs:
            return 0.5
        centroid = [sum(v[i] for v in recent_assistant_vecs) / len(recent_assistant_vecs)
                    for i in range(len(recent_assistant_vecs[0]))]
        return round(min(1.0, (1.0 - _cosim(candidate_vec, centroid)) / 0.5), 4)

    async def encode_message(self, text: str, latent: Optional[dict] = None, covariance: list = None) -> dict:
        if latent is None: latent = {"surprise": 0.3, "valence": 0.0, "velocity": 0.3}

        vec = await self.embed(text)
        velocity = self.semantic_velocity(vec)
        valence, arousal, vad_source = await self._score_vad_text(text, vec)
        
        intent_data = await self._classify_intent(vec)
        intent = intent_data["label"]
        concepts = await self._extract_concepts(vec, n=5) if self.concept_mode in ("vector", "both") else []

        # Contextual Belief Retrieval
        relevant_beliefs = []
        if self._beliefs_col and self._beliefs_col.count() > 0:
            try:
                b_results = self._beliefs_col.query(query_embeddings=[vec], n_results=3, include=["documents", "distances"])
                for doc, dist in zip(b_results["documents"][0], b_results["distances"][0]):
                    if (1.0 - dist) > 0.55: relevant_beliefs.append(doc)
            except Exception: pass

        top_sim = await self._top_similarity(vec)
        
        # ── Epistemic Foraging: covariance-driven, not string-matching ──
        # When the Kalman covariance trace(P) is high, the system is uncertain
        # and should forage memories to resolve uncertainty.
        P = covariance or [0.5, 0.5, 0.5]
        trace_P = sum(P) / len(P)  # mean covariance = epistemic uncertainty
        epistemic_drive = trace_P > 0.35  # uncertain → forage
        
        retrieve = top_sim >= _latent_threshold(self._sim_stats.threshold(), latent) or epistemic_drive
        
        retrieved, citation_required = [], False
        if retrieve:
            # Deeper retrieval when epistemically driven (more candidates)
            n_retrieve = _latent_n(RETRIEVAL_N_BASE, latent)
            if epistemic_drive:
                n_retrieve = max(n_retrieve, 6)  # forage wider when uncertain
            retrieved = await self._semantic_retrieve(vec, n=n_retrieve, latent=latent)

        # Empirical Physics
        complexity_stats = self.calculate_empirical_complexity(text)
        empirical_axes = {}
        for axis_name, vecs in self._empirical_vecs.items():
            sim_low = sum(_cosim(vec, v) for v in vecs["low"]) / len(vecs["low"])
            sim_high = sum(_cosim(vec, v) for v in vecs["high"]) / len(vecs["high"])
            empirical_axes[axis_name] = round(math.tanh((sim_high - sim_low) * 10), 3)

        return {
            "vector": vec, "velocity": velocity, 
            "tone": {"label": _tone_label(valence, arousal), "valence": valence, "arousal": arousal, "source": vad_source},
            "intent": intent, "intent_breakdown": intent_data.get("breakdown", {}),
            "meaning": {"intent": intent, "summary": ""}, "concepts": concepts, "concept_source": "vector",
            "prediction_error": {"retrieval_triggered": retrieve, "dampener": _retrieval_dampener(retrieved)},
            "top_similarity": top_sim, "retrieval_triggered": retrieve, "retrieval_reason": "high_sim" if retrieve else "none",
            "retrieved": retrieved, "citation_required": citation_required, "relevant_beliefs": relevant_beliefs,
            "complexity": complexity_stats, "empirical_axes": empirical_axes
        }