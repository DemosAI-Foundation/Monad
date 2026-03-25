"""
memory.py — SQLite persistence layer.

Stores all durable state across sessions:
  - Conversation history (user + assistant messages)
  - Episodic memories (with encoding strength, valence, recall count)
  - Brain state (latent, covariance, phenotype prior, EMAs)
  - Predictions and their accuracy
  - Concept uncertainty tracking
  - Latent state log for visualization
"""

import aiosqlite
import json
import logging
from datetime import datetime

logger  = logging.getLogger(__name__)
DB_PATH = "brain.db"

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Memory:
    async def init(self):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS brain_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    action_count INTEGER DEFAULT 0,
                    current_prediction TEXT,
                    msg_counter INTEGER DEFAULT 0,
                    topic_summary TEXT DEFAULT '',
                    surprise REAL DEFAULT 0.3,
                    valence REAL DEFAULT 0.0,
                    velocity REAL DEFAULT 0.3,
                    phenotype_json TEXT
                );

                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS latent_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    surprise REAL NOT NULL, valence REAL NOT NULL, velocity REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS core_beliefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    belief_text TEXT NOT NULL, trigger_error REAL, trigger_valence REAL
                );
                
                CREATE TABLE IF NOT EXISTS concept_uncertainty (
                    concept TEXT PRIMARY KEY,
                    uncertainty REAL DEFAULT 0.5,
                    last_foraged TEXT
                );

                CREATE TABLE IF NOT EXISTS prediction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    predicted TEXT, actual TEXT, error REAL, error_type TEXT,
                    dominant_state TEXT, explanation TEXT, is_silence INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS error_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    error REAL NOT NULL, source TEXT
                );

                CREATE TABLE IF NOT EXISTS silence_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    idle_duration INTEGER, dominant_state TEXT, error REAL, reply TEXT
                );
                
                CREATE TABLE IF NOT EXISTS enriched_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                    message_id TEXT, tone_label TEXT, tone_valence REAL, tone_arousal REAL,
                    intent TEXT, meaning_summary TEXT, concepts_json TEXT, concept_source TEXT,
                    top_similarity REAL, retrieval_triggered INTEGER DEFAULT 0,
                    retrieval_reason TEXT, retrieved_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    msg_counter INTEGER,
                    memory_type TEXT NOT NULL,
                    memory_text TEXT NOT NULL,
                    user_text_preview TEXT,
                    reply_preview TEXT,
                    surprise_at REAL, valence_at REAL, velocity_at REAL,
                    covariance_json TEXT,
                    precision_at REAL,
                    encoding_strength REAL,
                    alignment_score REAL,
                    error_at REAL,
                    recall_count INTEGER DEFAULT 0,
                    last_recall_ts TEXT,
                    valence_drift REAL DEFAULT 0.0,
                    original_valence REAL
                );

                CREATE TABLE IF NOT EXISTS brain_extended_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    ts TEXT NOT NULL,
                    valence_ema REAL DEFAULT 0.0,
                    arousal_ema REAL DEFAULT 0.5,
                    explore_exploit_ratio REAL DEFAULT 0.5,
                    self_model_error REAL DEFAULT 0.0,
                    terminal_state TEXT DEFAULT 'default_mode',
                    covariance_json TEXT,
                    predicted_next_vec_json TEXT,
                    predicted_next_text TEXT,
                    predicted_hyp_vec_json TEXT,
                    predicted_hyp_text TEXT,
                    intent_history_json TEXT,
                    error_lexical_json TEXT,
                    error_semantic_json TEXT,
                    error_pragmatic_json TEXT,
                    sim_accuracy_delib_json TEXT,
                    sim_accuracy_hyp_json TEXT,
                    conversation_centroid_json TEXT,
                    action_history_json TEXT,
                    output_valence_alignment REAL DEFAULT 0.0,
                    output_velocity_alignment REAL DEFAULT 0.0,
                    output_context_usage REAL DEFAULT 0.0,
                    deliberation_gain_ema REAL DEFAULT 0.0,
                    sim_accuracy_ema REAL DEFAULT 0.5
                );
            """)
            try: await db.execute("ALTER TABLE brain_state ADD COLUMN phenotype_json TEXT")
            except Exception: pass
            await db.commit()
        logger.info("Database initialised.")

    async def save_brain_state(self, brain) -> None:
        lat = getattr(brain, "latent", {"surprise": 0.3, "valence": 0.0, "velocity": 0.3})
        ptype = getattr(brain, "phenotype_prior", {})
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO brain_state (ts, action_count, current_prediction, msg_counter, topic_summary, surprise, valence, velocity, phenotype_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts(), brain.action_count, brain.current_prediction, brain._msg_counter, getattr(brain, "topic_summary", ""),
                 lat.get("surprise", 0.3), lat.get("valence",  0.0), lat.get("velocity", 0.3), json.dumps(ptype))
            )
            await db.commit()

    async def load_brain_state(self) -> dict | None:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM brain_state ORDER BY id DESC LIMIT 1") as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def append_message(self, role: str, content: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO conversation_history (ts, role, content) VALUES (?,?,?)", (ts(), role, content))
            await db.commit()

    async def load_conversation(self, limit: int = 40) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?", (limit,)) as cur:
                rows = await cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def load_messages_for_ui(self, limit: int = 80) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT ts, role, content FROM conversation_history ORDER BY id DESC LIMIT ?", (limit,)) as cur:
                rows = await cur.fetchall()
        return [{"role": r["role"], "content": r["content"], "ts": r["ts"], "error": None, "origin": r["role"], "enriched": {}} for r in reversed(rows)]

    # ── Core Beliefs (Semantic Consolidation) ──
    async def add_core_belief(self, belief_text: str, error: float, valence: float):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO core_beliefs (ts, belief_text, trigger_error, trigger_valence) VALUES (?,?,?,?)", (ts(), belief_text, error, valence))
            await db.commit()

    async def get_core_beliefs(self, limit: int = 30) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT belief_text FROM core_beliefs ORDER BY id DESC LIMIT ?", (limit,)) as cur:
                rows = await cur.fetchall()
        return [r["belief_text"] for r in rows]

    # ── Episodic Memory ──
    async def save_episodic_memory(self, mem: dict) -> int:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("""
                INSERT INTO episodic_memory (ts, msg_counter, memory_type, memory_text,
                    user_text_preview, reply_preview,
                    surprise_at, valence_at, velocity_at, covariance_json,
                    precision_at, encoding_strength, alignment_score, error_at,
                    recall_count, last_recall_ts, valence_drift, original_valence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,NULL,0.0,?)
            """, (
                ts(), mem.get("msg_counter", 0), mem["memory_type"], mem["memory_text"],
                mem.get("user_text_preview", ""), mem.get("reply_preview", ""),
                mem.get("surprise_at", 0.3), mem.get("valence_at", 0.0), mem.get("velocity_at", 0.3),
                json.dumps(mem.get("covariance", [])),
                mem.get("precision_at", 1.0), mem.get("encoding_strength", 0.5),
                mem.get("alignment_score", 0.0), mem.get("error_at", 0.5),
                mem.get("valence_at", 0.0),  # original_valence = valence at first encoding
            ))
            await db.commit()
            return cur.lastrowid

    async def update_episodic_recall(self, mem_id: int, new_text: str, new_valence: float, valence_drift: float):
        """Update a memory after reconsolidation: new text, updated recall stats."""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                UPDATE episodic_memory SET
                    memory_text = ?, valence_at = ?, valence_drift = ?,
                    recall_count = recall_count + 1, last_recall_ts = ?
                WHERE id = ?
            """, (new_text, new_valence, valence_drift, ts(), mem_id))
            await db.commit()

    async def bump_recall_count(self, mem_id: int):
        """Increment recall count without reconsolidation (low-fidelity familiarity)."""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE episodic_memory SET recall_count = recall_count + 1, last_recall_ts = ? WHERE id = ?",
                (ts(), mem_id))
            await db.commit()

    async def get_recall_history(self, limit: int = 20) -> list:
        """Get most-recalled memories for diagnostic monitoring."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, memory_type, memory_text, recall_count, last_recall_ts,
                       valence_at, original_valence, valence_drift, encoding_strength
                FROM episodic_memory WHERE recall_count > 0
                ORDER BY recall_count DESC LIMIT ?
            """, (limit,)) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def load_episodic_memories(self, limit: int = 40) -> list:
        """Load recent episodic memories for conversation reconstruction."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM episodic_memory ORDER BY id DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Extended State Persistence ──
    async def save_extended_state(self, state: dict) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO brain_extended_state (id, ts, valence_ema, arousal_ema, explore_exploit_ratio,
                    self_model_error, terminal_state, covariance_json,
                    predicted_next_vec_json, predicted_next_text, predicted_hyp_vec_json, predicted_hyp_text,
                    intent_history_json, error_lexical_json, error_semantic_json, error_pragmatic_json,
                    sim_accuracy_delib_json, sim_accuracy_hyp_json, conversation_centroid_json, action_history_json,
                    output_valence_alignment, output_velocity_alignment, output_context_usage,
                    deliberation_gain_ema, sim_accuracy_ema)
                VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    ts=excluded.ts, valence_ema=excluded.valence_ema, arousal_ema=excluded.arousal_ema,
                    explore_exploit_ratio=excluded.explore_exploit_ratio, self_model_error=excluded.self_model_error,
                    terminal_state=excluded.terminal_state, covariance_json=excluded.covariance_json,
                    predicted_next_vec_json=excluded.predicted_next_vec_json, predicted_next_text=excluded.predicted_next_text,
                    predicted_hyp_vec_json=excluded.predicted_hyp_vec_json, predicted_hyp_text=excluded.predicted_hyp_text,
                    intent_history_json=excluded.intent_history_json,
                    error_lexical_json=excluded.error_lexical_json, error_semantic_json=excluded.error_semantic_json,
                    error_pragmatic_json=excluded.error_pragmatic_json,
                    sim_accuracy_delib_json=excluded.sim_accuracy_delib_json, sim_accuracy_hyp_json=excluded.sim_accuracy_hyp_json,
                    conversation_centroid_json=excluded.conversation_centroid_json, action_history_json=excluded.action_history_json,
                    output_valence_alignment=excluded.output_valence_alignment, output_velocity_alignment=excluded.output_velocity_alignment,
                    output_context_usage=excluded.output_context_usage,
                    deliberation_gain_ema=excluded.deliberation_gain_ema, sim_accuracy_ema=excluded.sim_accuracy_ema
            """, (
                ts(), state.get("valence_ema", 0.0), state.get("arousal_ema", 0.5),
                state.get("explore_exploit_ratio", 0.5), state.get("self_model_error", 0.0),
                state.get("terminal_state", "default_mode"), json.dumps(state.get("covariance")),
                json.dumps(state.get("predicted_next_vec")), state.get("predicted_next_text", ""),
                json.dumps(state.get("predicted_hyp_vec")), state.get("predicted_hyp_text", ""),
                json.dumps(state.get("intent_history", [])),
                json.dumps(state.get("error_lexical", [])), json.dumps(state.get("error_semantic", [])),
                json.dumps(state.get("error_pragmatic", [])),
                json.dumps(state.get("sim_accuracy_delib", [])), json.dumps(state.get("sim_accuracy_hyp", [])),
                json.dumps(state.get("conversation_centroid")), json.dumps(state.get("action_history", {})),
                state.get("output_valence_alignment", 0.0), state.get("output_velocity_alignment", 0.0),
                state.get("output_context_usage", 0.0),
                state.get("deliberation_gain_ema", 0.0), state.get("sim_accuracy_ema", 0.5),
            ))
            await db.commit()

    async def load_extended_state(self) -> dict | None:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM brain_extended_state WHERE id = 1") as cur:
                row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        # Deserialize JSON fields
        for key in ("covariance_json", "predicted_next_vec_json", "predicted_hyp_vec_json",
                     "intent_history_json", "error_lexical_json", "error_semantic_json",
                     "error_pragmatic_json", "sim_accuracy_delib_json", "sim_accuracy_hyp_json",
                     "conversation_centroid_json", "action_history_json"):
            try: d[key] = json.loads(d.get(key) or "null")
            except Exception: d[key] = None
        return d

    # ── Concept Uncertainty (Epistemic Foraging) ──
    async def update_concept_uncertainty(self, concepts: list, surprise: float):
        if not concepts: return
        async with aiosqlite.connect(DB_PATH) as db:
            for c in concepts:
                await db.execute("""
                    INSERT INTO concept_uncertainty (concept, uncertainty, last_foraged)
                    VALUES (?, ?, '')
                    ON CONFLICT(concept) DO UPDATE SET uncertainty = (uncertainty * 0.7) + (? * 0.3)
                """, (c, surprise, surprise))
            await db.commit()

    async def get_most_uncertain_concept(self) -> str:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT concept FROM concept_uncertainty WHERE uncertainty > 0.6 ORDER BY uncertainty DESC LIMIT 1") as cur:
                row = await cur.fetchone()
        return row["concept"] if row else None

    async def mark_concept_foraged(self, concept: str):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE concept_uncertainty SET uncertainty = 0.1, last_foraged = ? WHERE concept = ?", (ts(), concept))
            await db.commit()

    # ── Logging & UI Fetching ──
    async def log_latent(self, latent: dict) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO latent_log (ts, surprise, valence, velocity) VALUES (?,?,?,?)", (ts(), latent.get("surprise", 0.3), latent.get("valence", 0.0), latent.get("velocity", 0.3)))
            await db.commit()

    async def log_prediction(self, predicted, actual, error, error_type, dominant, explanation, is_silence=False):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO prediction_log (ts,predicted,actual,error,error_type,dominant_state,explanation,is_silence) VALUES (?,?,?,?,?,?,?,?)", (ts(), predicted, actual, error, error_type, dominant, explanation, int(is_silence)))
            await db.commit()

    async def log_error_stat(self, error: float, source: str = "reactive"):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO error_stats (ts, error, source) VALUES (?,?,?)", (ts(), error, source))
            await db.commit()

    async def log_silence(self, idle_duration, dominant_state, error, reply):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO silence_log (ts,idle_duration,dominant_state,error,reply) VALUES (?,?,?,?,?)", (ts(), idle_duration, dominant_state, error, reply))
            await db.commit()

    async def log_enriched(self, message_id: str, enriched: dict):
        tone, meaning = enriched.get("tone", {}), enriched.get("meaning", {})
        labels = [c.get("label","") for c in enriched.get("concepts",[]) if isinstance(c, dict)]
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO enriched_log (ts,message_id,tone_label,tone_valence,tone_arousal,intent,meaning_summary,concepts_json,concept_source,top_similarity,retrieval_triggered,retrieval_reason,retrieved_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (enriched.get("ts", ts()), message_id, tone.get("label", ""), float(tone.get("valence", 0.0)), float(tone.get("arousal", 0.5)), meaning.get("intent", ""), meaning.get("summary", "")[:500], json.dumps(labels), enriched.get("concept_source", "none"), float(enriched.get("top_similarity", 0.0)), int(enriched.get("retrieval_triggered", False)), enriched.get("retrieval_reason", "none"), len(enriched.get("retrieved", [])))
            )
            await db.commit()

    async def get_predictions(self, limit: int = 50) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM prediction_log ORDER BY id DESC LIMIT ?", (limit,)) as cur: return [dict(r) for r in await cur.fetchall()]
    async def get_latent_log(self, limit: int = 100) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM latent_log ORDER BY id DESC LIMIT ?", (limit,)) as cur: return [dict(r) for r in await cur.fetchall()]
    async def get_silence_log(self, limit: int = 30) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM silence_log ORDER BY id DESC LIMIT ?", (limit,)) as cur: return [dict(r) for r in await cur.fetchall()]
    async def get_error_stats(self) -> dict:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT AVG(error) as mean, COUNT(*) as count FROM error_stats") as cur: agg = dict(await cur.fetchone())
            async with db.execute("SELECT error FROM error_stats ORDER BY id DESC LIMIT 50") as cur: rows = await cur.fetchall()
        errors = [r["error"] for r in rows]
        mean = agg.get("mean") or 0.0
        variance = sum((e - mean)**2 for e in errors) / max(len(errors)-1, 1) if len(errors) > 1 else 0.0
        return {"mean": round(mean, 4), "variance": round(variance, 4), "count": agg.get("count") or 0, "history": [{"error": r["error"]} for r in reversed(rows)]}
    async def get_enriched_log(self, limit: int = 30) -> list:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM enriched_log ORDER BY id DESC LIMIT ?", (limit,)) as cur: rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try: d["concepts_json"] = json.loads(d.get("concepts_json") or "[]")
            except Exception: d["concepts_json"] = []
            result.append(d)
        return result