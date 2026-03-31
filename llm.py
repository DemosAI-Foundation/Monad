"""
llm.py — LLM client: language rendering surface for the Bayesian brain.

The LLM generates text when asked. This module controls HOW it generates:
  - 10 task profiles, each producing 8 physics-driven sampling parameters
  - Universal word budget tracking with continuous energy consequences
  - Streaming with sentence-boundary abort
  - EOS logit bias that scales with overshoot history
  - Perplexity measurement (3 strategies)
  - HyDE validation for memory encoding
  - Working memory compression

Every LLM call is measured: actual_words / budgeted_words → energy consequences.
Overshoot drains energy (quadratic). Undershoot tightens future budgets (linear).
No thresholds — all consequences scale continuously.
"""

import asyncio
import logging
import math
import time
import json
import re
from typing import Optional, Callable
import aiohttp

logger = logging.getLogger(__name__)

PREFERRED_STATE = {"surprise": 0.25, "valence": 0.30, "velocity": 0.40}

def _compute_manifold(latent: dict, task: str = "output", feedback: dict = None) -> dict:
    """Map latent state → all 8 sampling parameters. No hardcoded overrides.
    
    Every task type produces all 8 params derived from the latent state.
    Task type changes the BASE sensitivity and SCALING, not the physics.
    
    8 parameters:
        temperature       ← surprise (exploration)
        top_p             ← surprise (nucleus sampling width)
        min_p             ← precision (token quality floor)
        presence_penalty  ← explore_ratio (novelty pressure)
        frequency_penalty ← velocity (anti-repetition)
        repetition_penalty ← velocity (harder anti-repetition)
        dynatemp_range    ← surprise (temperature uncertainty)
        max_tokens        ← surprise × energy (output budget)
    """
    s = latent.get("surprise", 0.3)
    v = latent.get("valence", 0.0)
    vel = latent.get("velocity", 0.3)
    
    fb = feedback or {}
    precision = fb.get("precision", 1.0)
    explore_ratio = fb.get("explore_ratio", 0.5)
    energy = fb.get("energy", 1.0)
    budget_accuracy = fb.get("budget_accuracy", 1.0)
    budget_comp = max(0.5, min(1.3, 2.0 - budget_accuracy))
    energy_scale = max(0.4, 0.5 + energy * 0.5)
    valence_corr = max(0.7, 1.0 - fb.get("valence_misalignment", 0.0) * 0.3)
    velocity_corr = max(0.4, 1.0 - fb.get("velocity_misalignment", 0.0) * 0.5)
    sim_accuracy = fb.get("sim_accuracy", 0.5)
    sim_mod = round(1.0 + (0.5 - sim_accuracy) * 0.4, 3)
    
    # ── Task-specific base scaling ──
    # Each task sets a sensitivity multiplier [0-1] for how much state affects params
    # and a base offset for each param
    profiles = {
        "output":       {"t_base": 0.15, "t_scale": 0.85, "p_base": 0.50, "p_scale": 0.45, "budget_base": 50, "budget_scale": 200},
        "hypothetical": {"t_base": 0.15, "t_scale": 0.85, "p_base": 0.50, "p_scale": 0.45, "budget_base": 50, "budget_scale": 200},
        "predict":      {"t_base": 0.30, "t_scale": 0.50, "p_base": 0.80, "p_scale": 0.15, "budget_base": 15, "budget_scale": 30},
        "internal":     {"t_base": 0.10, "t_scale": 0.40, "p_base": 0.40, "p_scale": 0.30, "budget_base": 20, "budget_scale": 80},
        "creative":     {"t_base": 0.40, "t_scale": 0.55, "p_base": 0.75, "p_scale": 0.20, "budget_base": 20, "budget_scale": 60},
        "compression":  {"t_base": 0.10, "t_scale": 0.30, "p_base": 0.50, "p_scale": 0.25, "budget_base": 30, "budget_scale": 100},
        "summary":      {"t_base": 0.15, "t_scale": 0.30, "p_base": 0.55, "p_scale": 0.25, "budget_base": 30, "budget_scale": 80},
        "diagnostic":   {"t_base": 0.10, "t_scale": 0.20, "p_base": 0.50, "p_scale": 0.20, "budget_base": 80, "budget_scale": 120},
        "label":        {"t_base": 0.10, "t_scale": 0.20, "p_base": 0.40, "p_scale": 0.15, "budget_base": 5,  "budget_scale": 10},
        "recorder":     {"t_base": 0.10, "t_scale": 0.30, "p_base": 0.40, "p_scale": 0.20, "budget_base": 10, "budget_scale": 30},
    }
    p = profiles.get(task, profiles["internal"])
    
    # ── Compute all 8 params from state ──
    temp = round((p["t_base"] + s * p["t_scale"]) * valence_corr, 3)
    top_p = round(p["p_base"] + s * p["p_scale"], 3)
    min_p = round(max(0.01, min(0.15, precision * 0.05)), 3)
    presence_pen = round(max(0.0, explore_ratio * 0.4), 2)
    freq_pen = round(max(0.0, vel * 0.3), 2)
    rep_pen = round(1.0 + vel * 0.25, 3)
    dynatemp = round(max(0.0, s * 0.3), 2)
    
    max_tokens = int((p["budget_base"] + s * p["budget_scale"]) * budget_comp * energy_scale)
    if v < -0.5:
        max_tokens = min(max_tokens, 100)
    max_tokens = int(max_tokens * velocity_corr)
    
    # Task-specific adjustments (still physics-driven)
    if task == "predict":
        temp = round(temp * sim_mod, 3)
    
    return {
        "temperature": temp,
        "top_p": top_p,
        "min_p": min_p,
        "presence_penalty": presence_pen,
        "frequency_penalty": freq_pen,
        "repetition_penalty": rep_pen,
        "dynatemp_range": dynatemp,
        "max_tokens": max_tokens,
    }

def _state_conditioning(latent: dict) -> str:
    s  = latent.get("surprise", 0.3)
    v  = latent.get("valence", 0.0)
    vel = latent.get("velocity", 0.3)
    return (
        f"--- STATE CONDITIONING ---\n"
        f"Surprise {s:.2f} / range [0,1]\n"
        f"Valence {v:+.2f} / range [-1,+1]\n"
        f"Velocity {vel:.2f} / range [0,1]"
    )

def _user_response_budget(conversation: list) -> tuple[int, int]:
    """Measure average user message length from conversation history.
    Returns (word_budget, token_budget) for prediction clamping."""
    user_msgs = [m["content"] for m in conversation if m.get("role") == "user"]
    if not user_msgs:
        return 20, 30
    word_counts = [len(m.split()) for m in user_msgs]
    avg_words = sum(word_counts) / len(word_counts)
    # Weight recent messages more heavily (users may shift style)
    if len(word_counts) >= 3:
        recent_avg = sum(word_counts[-3:]) / 3
        avg_words = avg_words * 0.4 + recent_avg * 0.6
    word_budget = max(5, min(120, int(avg_words * 1.3)))  # slight overshoot for headroom
    token_budget = max(10, min(200, int(word_budget * 1.4)))  # rough word->token ratio
    return word_budget, token_budget

class LLMClient:
    """Language rendering surface for the Bayesian brain.
    
    The LLM generates text when asked. This class controls:
      - HOW it generates: 8 sampling params from _compute_manifold(latent, task)
      - HOW MUCH it generates: word budget injected into system prompt + EOS bias
      - WHAT HAPPENS after: universal word count tracking with energy consequences
    
    Every call routes through _call(), which:
      1. Injects word budget into system prompt
      2. Computes EOS logit bias (scales with overshoot history)
      3. Selects streaming (output) or non-streaming (internal)
      4. Measures actual vs budgeted words
      5. Applies continuous energy drain/penalty
    
    10 task profiles: output, hypothetical, predict, internal, creative,
    compression, summary, diagnostic, label, recorder.
    """
    def __init__(self, endpoint: str = ""):
        self.endpoint, self.trace_callback = endpoint, None
        # ── Interoceptive State (Critique 6) ──
        self._cycle_calls: int = 0
        self._cycle_latency_ms: float = 0
        self._total_calls: int = 0
        self._total_latency_ms: float = 0
        # ── Feedback from brain (Point 10) ──
        self._feedback: dict = {}
        # ── Word Budget Tracking ──
        self._budget_accuracy_ema: float = 1.0
        self._budget_overshoot_ema: float = 0.0
        self._last_budget_stats: dict = {}
        self._energy_offset: float = 0.0  # drain from overshoot, recovers each cycle
        self._budget_penalty: float = 0.0  # reduction from undershoot, recovers slowly
        # ── Vocabulary Pruning (Semantic Aphasia) ──
        self._complex_token_ids: list = []  # populated at connection time
        self._vocab_ready: bool = False
        self._eos_token_id: Optional[int] = None  # discovered at connection time

    def reset_cycle_stats(self, feedback: dict = None):
        """Called at start of each process_input cycle."""
        self._cycle_calls = 0
        self._cycle_latency_ms = 0
        # Partially recover energy each cycle (metabolic recovery)
        self._energy_offset = min(0.0, self._energy_offset + 0.02)
        if feedback: self._feedback = feedback

    @property
    def energy(self) -> float:
        """Interoceptive energy [0, 1]. Decays with calls, latency, budget overshoot,
        and direct energy drain from overshooting word budgets."""
        call_cost = min(1.0, self._cycle_calls / 12.0)
        latency_cost = min(1.0, self._cycle_latency_ms / 30000)
        overshoot_cost = min(0.3, self._budget_overshoot_ema * 0.4)
        offset = getattr(self, '_energy_offset', 0.0)
        return round(max(0.0, min(1.0, 1.0 - (call_cost * 0.5 + latency_cost * 0.3 + overshoot_cost) + offset)), 4)

    @property
    def interoceptive_state(self) -> dict:
        """Current metabolic state: energy, call counts, latency, budget accuracy."""
        return {
            "energy": self.energy,
            "cycle_calls": self._cycle_calls,
            "cycle_latency_ms": round(self._cycle_latency_ms),
            "total_calls": self._total_calls,
            "budget_accuracy": round(self._budget_accuracy_ema, 3),
            "budget_overshoot": round(self._budget_overshoot_ema, 3),
            "last_budget": self._last_budget_stats,
        }

    async def test_connection(self) -> bool:
        """Check if LLM endpoint is reachable. Initializes EOS token and vocab pruning on first success."""
        if not self.endpoint: return False
        for path in ["/health", "/v1/models"]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.endpoint}{path}", timeout=aiohttp.ClientTimeout(total=4)) as resp:
                        if resp.status in (200, 404):
                            if not self._vocab_ready:
                                asyncio.create_task(self._init_vocab_pruning())
                            if not self._eos_token_id:
                                asyncio.create_task(self._init_eos_token())
                            return True
            except Exception: continue
        return False

    async def _init_eos_token(self):
        """Discover EOS token ID from llama.cpp /props endpoint or by tokenizing known markers."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try /props first — it may expose default_generation_settings with stop tokens
                try:
                    async with session.get(f"{self.endpoint}/props", timeout=aiohttp.ClientTimeout(total=4)) as resp:
                        if resp.ok:
                            data = await resp.json()
                            # Some builds expose eos_token_id directly
                            if "default_generation_settings" in data:
                                eos = data["default_generation_settings"].get("eos_token_id")
                                if eos:
                                    self._eos_token_id = int(eos)
                                    logger.info(f"EOS token ID from /props: {self._eos_token_id}")
                                    return
                except Exception:
                    pass
                
                # Fallback: tokenize known EOS markers and use the shortest result
                for marker in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"]:
                    try:
                        async with session.post(f"{self.endpoint}/tokenize",
                            json={"content": marker, "add_special": False},
                            timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.ok:
                                data = await resp.json()
                                tokens = data.get("tokens", [])
                                if len(tokens) == 1:  # single token = likely an actual special token
                                    self._eos_token_id = tokens[0]
                                    logger.info(f"EOS token ID from tokenize('{marker}'): {self._eos_token_id}")
                                    return
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"EOS token discovery failed: {e}")

    async def _init_vocab_pruning(self):
        """Tokenize complex words via llama.cpp /tokenize endpoint.
        Maps sophisticated vocabulary to token IDs for energy-based logit_bias."""
        COMPLEX_WORDS = [
            "nevertheless", "furthermore", "consequently", "notwithstanding", "ameliorate",
            "juxtaposition", "quintessential", "extrapolate", "paradigmatic", "epistemological",
            "phenomenological", "hermeneutic", "ontological", "concatenate", "obfuscate",
            "confabulate", "verisimilitude", "perspicacious", "sesquipedalian", "antidisestablishmentarian",
            "circumlocution", "prognosticate", "unequivocally", "disproportionate", "indistinguishable",
            "simultaneously", "characteristically", "disenfranchise", "compartmentalize",
            "interconnectedness", "multifaceted", "predominantly", "comprehensive", "infrastructure",
            "implementation", "sophisticated", "fundamentally", "subsequently", "approximately",
            "conceptualize", "contextualize", "instrumentalize", "operationalize", "systematically",
        ]
        try:
            async with aiohttp.ClientSession() as session:
                token_ids = set()
                for word in COMPLEX_WORDS:
                    try:
                        async with session.post(f"{self.endpoint}/tokenize",
                            json={"content": word}, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                            if resp.ok:
                                data = await resp.json()
                                tokens = data.get("tokens", [])
                                token_ids.update(tokens)
                    except Exception:
                        continue
                self._complex_token_ids = list(token_ids)
                self._vocab_ready = bool(self._complex_token_ids)
                if self._vocab_ready:
                    logger.info(f"Vocabulary pruning ready: {len(self._complex_token_ids)} tokens mapped from {len(COMPLEX_WORDS)} complex words.")
        except Exception as e:
            logger.debug(f"Vocabulary pruning init failed: {e}")

    def _compute_vocab_logit_bias(self) -> dict:
        """Compute logit_bias penalties from interoceptive energy.
        Low energy → complex words get negative bias → simpler output."""
        if not self._complex_token_ids or not self._vocab_ready:
            return {}
        energy = self.energy
        if energy > 0.5:
            return {}
        penalty = round(-5.0 * (1.0 - energy * 2.0), 1)
        return {str(tid): penalty for tid in self._complex_token_ids}

    def _compute_eos_bias(self, word_budget: int) -> dict:
        """Compute EOS token logit bias — continuous scaling from energy, budget, and history.
        
        No thresholds. Alpha scales continuously with overshoot history:
        perfect accuracy (1.0) → alpha 0.3 (gentle)
        poor accuracy (0.5) → alpha 0.8 (firm)
        terrible accuracy (0.0) → alpha 1.3 (strong)
        """
        if not self._eos_token_id:
            return {}
        energy = max(0.2, self.energy)
        budget_ratio = max(0.5, word_budget / 50.0)
        
        # Continuous alpha: scales inversely with budget accuracy
        alpha = round(0.3 + (1.0 - self._budget_accuracy_ema) * 1.0, 2)
        
        eos_bias = round(min(2.5, alpha / budget_ratio / energy), 2)
        return {str(self._eos_token_id): eos_bias}

    def _sanitize(self, messages: list) -> list:
        """Merge consecutive same-role messages. Chat APIs require alternating roles."""
        result = []
        for msg in messages:
            if result and result[-1]["role"] == msg["role"]: result[-1]["content"] += "\n" + msg["content"]
            else: result.append(dict(msg))
        return result

    async def _call(self, messages: list, manifold: dict, label: str = "unknown") -> str:
        """Central LLM call. Every call goes through here. No exceptions.
        
        Before calling:
          1. Compute word budget (with undershoot penalty from history)
          2. Inject budget into system prompt: "Limit: N words. Do not mention this limit."
          3. Compute EOS logit bias (scales with overshoot history)
          4. Select streaming (output/hypothetical) or non-streaming (everything else)
          5. Set max_tokens = budget × 2 (generous but bounded)
        
        After calling:
          6. Measure actual_words / budgeted_words
          7. Apply continuous energy consequences:
             - Overshoot: energy_drain = overshoot² × 0.04 (quadratic, cumulative)
             - Undershoot: penalty_delta = undershoot × 0.1 - 0.02 (linear, recovers)
          8. Update accuracy/overshoot EMAs
          9. Broadcast trace with sampling params + budget stats
        
        Streaming stops at sentence boundary (.!?\\n) after exceeding word budget.
        Hard abort at 2× budget.
        """
        if not self.endpoint: return ""
        messages = self._sanitize(messages)
        t0 = time.monotonic()
        
        energy = self.energy
        # Bypass energy temp damping for output and diagnostic calls
        if energy < 0.5 and label not in ("output", "diagnostic"):
            manifold = dict(manifold)
            manifold["temperature"] = round(manifold.get("temperature", 0.7) * (0.6 + energy * 0.4), 3)

        # ── Compute word budget with penalty for chronic undershoot ──
        base_budget = int(manifold.get("max_tokens", 150) * 0.75)
        penalty = getattr(self, '_budget_penalty', 0.0)
        word_budget = max(5, int(base_budget * max(0.5, 1.0 - penalty)))
        
        # ── Inject word budget into system prompt ──
        # Every call gets a budget. The LLM knows how much it can say.
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"].rstrip() + f" Limit: {word_budget} words. Do not mention this limit."
                break

        # ── Logit biases: vocabulary pruning + EOS bias (universal) ──
        vocab_bias = self._compute_vocab_logit_bias()
        eos_bias = self._compute_eos_bias(word_budget)
        combined_bias = {**vocab_bias, **eos_bias}

        # ── Determine if streaming (output calls only) ──
        use_stream = label in ("output", "predict_hypothetical_reply")

        # max_tokens from manifold (word budget → token estimate at ~1.5 tokens/word)
        manifold_budget = manifold.get("max_tokens", 150)
        api_max_tokens = max(30, int(manifold_budget * 2))  # generous but not infinite
        
        payload = {
            "model": "local", "messages": messages,
            "temperature": manifold.get("temperature", 0.7),
            "top_p": manifold.get("top_p", 0.9),
            "min_p": manifold.get("min_p", 0.05),
            "repetition_penalty": manifold.get("repetition_penalty", 1.0),
            "presence_penalty": manifold.get("presence_penalty", 0.0),
            "frequency_penalty": manifold.get("frequency_penalty", 0.0),
            "max_tokens": api_max_tokens,
            "stream": use_stream,
        }
        
        dynatemp = manifold.get("dynatemp_range", 0.0)
        if dynatemp > 0:
            payload["dynatemp_range"] = dynatemp
        if combined_bias:
            payload["logit_bias"] = combined_bias

        reply = ""
        
        if use_stream:
            # ── Streaming: accumulate tokens, stop at sentence boundary after budget ──
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.endpoint}/v1/chat/completions", json=payload,
                                          timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if not resp.ok:
                            return ""
                        past_budget = False
                        async for line in resp.content:
                            line = line.decode("utf-8", errors="ignore").strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token_text = delta.get("content", "")
                                if token_text:
                                    reply += token_text
                                    word_count = len(reply.split())
                                    
                                    if not past_budget and word_count > word_budget:
                                        past_budget = True  # start looking for sentence boundary
                                    
                                    if past_budget:
                                        # Stop at next sentence boundary
                                        stripped = reply.rstrip()
                                        if stripped and stripped[-1] in '.!?\n':
                                            break
                                        # Hard abort at 2x budget (absolute safety)
                                        if word_count > word_budget * 2:
                                            break
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
            except Exception as e:
                logger.warning(f"Streaming call failed: {e}")
                # Fallback to non-streaming on error
                try:
                    payload["stream"] = False
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{self.endpoint}/v1/chat/completions", json=payload,
                                              timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            if resp.ok:
                                data = await resp.json()
                                reply = data["choices"][0]["message"]["content"]
                except Exception:
                    pass
        else:
            # ── Non-streaming: standard request/response ──
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.endpoint}/v1/chat/completions", json=payload,
                                      timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if not resp.ok: return ""
                    data = await resp.json()
                    reply = data["choices"][0]["message"]["content"]
                
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        self._cycle_calls += 1
        self._cycle_latency_ms += elapsed_ms
        self._total_calls += 1
        self._total_latency_ms += elapsed_ms
        
        # ── Universal Word Budget Tracking + Continuous Energy Consequences ──
        actual_words = len(reply.split())
        budgeted_words = int(manifold.get("max_tokens", 150) * 0.75)
        ratio = actual_words / max(budgeted_words, 1)
        accuracy = max(0.0, 1.0 - abs(1.0 - ratio))
        overshoot = max(0.0, ratio - 1.0)
        undershoot = max(0.0, 1.0 - ratio)
        
        # Continuous energy consequences — no thresholds, pure scaling:
        # 1. Overshoot → energy drain proportional to how far over (quadratic)
        #    ratio 1.0 → 0 drain, ratio 1.5 → 0.01, ratio 2.0 → 0.04, ratio 3.0 → 0.16
        energy_drain = overshoot ** 2 * 0.04
        self._energy_offset = max(-0.3, getattr(self, '_energy_offset', 0.0) - energy_drain)
        
        # 2. Undershoot → budget penalty proportional to how short (linear)
        #    ratio 1.0 → 0 penalty growth, ratio 0.5 → 0.05, ratio 0.1 → 0.09
        penalty_delta = undershoot * 0.1 - 0.02  # grows when short, shrinks when close/over
        self._budget_penalty = max(0.0, min(0.5, getattr(self, '_budget_penalty', 0.0) + penalty_delta))
        
        # 3. Close match → accuracy naturally high → prediction error lower → inherent reward
        
        self._budget_accuracy_ema = round(0.3 * accuracy + 0.7 * self._budget_accuracy_ema, 4)
        self._budget_overshoot_ema = round(0.3 * overshoot + 0.7 * self._budget_overshoot_ema, 4)
        
        budget_stats = {
            "actual_words": actual_words, "budgeted_words": budgeted_words,
            "ratio": round(ratio, 3), "accuracy": round(accuracy, 3),
            "overshoot": round(overshoot, 3),
            "accuracy_ema": round(self._budget_accuracy_ema, 3),
            "overshoot_ema": round(self._budget_overshoot_ema, 3),
        }
        self._last_budget_stats = budget_stats

        if self.trace_callback:
            trace_data = {
                "type": "llm_trace", "label": label, "duration_ms": elapsed_ms, "energy": energy,
                "sampling": {
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "min_p": payload.get("min_p", 0.05),
                    "presence_penalty": payload.get("presence_penalty", 0.0),
                    "frequency_penalty": payload.get("frequency_penalty", 0.0),
                    "repetition_penalty": payload.get("repetition_penalty", 1.0),
                    "dynatemp_range": payload.get("dynatemp_range", 0.0),
                    "eos_bias": eos_bias.get(str(self._eos_token_id), 0.0) if eos_bias else 0.0,
                    "streaming": use_stream,
                },
                "messages": messages,
                "response": reply,
            }
            if budget_stats:
                trace_data["budget"] = budget_stats
            try: await self.trace_callback(trace_data)
            except Exception: pass
        return reply

    async def compute_perplexity(self, text: str, context_messages: list) -> Optional[dict]:
        """Measure how surprising the input was to the model.
        
        Strategy 1: /v1/chat/completions with logprobs=true, max_tokens=5
            Generate a few tokens and measure average logprob confidence.
            High average logprob = model was confident = low surprise.
            Low average logprob = model was uncertain = high surprise.
            
        Strategy 2: /completion (native) with n_predict=5, n_probs=5
            Same idea via native endpoint.
            
        Strategy 3: /v1/completions with echo=true (if supported)
            Full prompt token logprobs — most precise but least supported.
        """
        if not self.endpoint: return None
        t0 = time.monotonic()
        
        # Build messages for chat-format perplexity measurement
        msgs = [{"role": m["role"], "content": m["content"][:300]} for m in context_messages[-4:]]
        msgs.append({"role": "user", "content": text})
        
        log_probs = None
        method_used = "none"
        n_total_tokens = 0

        try:
            async with aiohttp.ClientSession() as session:
                # ── Strategy 1: /v1/chat/completions with logprobs ──
                try:
                    payload = {
                        "model": "local", "messages": msgs,
                        "max_tokens": 5, "temperature": 0,
                        "logprobs": True, "top_logprobs": 5
                    }
                    async with session.post(f"{self.endpoint}/v1/chat/completions", json=payload,
                                          timeout=aiohttp.ClientTimeout(total=20)) as resp:
                        if resp.ok:
                            data = await resp.json()
                            choices = data.get("choices", [])
                            if choices:
                                lp_content = choices[0].get("logprobs", {}).get("content", [])
                                if lp_content:
                                    extracted = [item["logprob"] for item in lp_content if "logprob" in item]
                                    n_total_tokens = len(extracted)
                                    if len(extracted) >= 1:
                                        log_probs = extracted
                                        method_used = "/v1/chat/completions"
                except Exception as e:
                    logger.debug(f"Perplexity chat/completions failed: {e}")

                # ── Strategy 2: /completion (native) with n_probs ──
                if not log_probs:
                    try:
                        context_lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in context_messages[-4:]]
                        full_prompt = ("\n".join(context_lines) + "\nUser: " + text) if context_lines else ("User: " + text)
                        payload_native = {
                            "prompt": full_prompt, "n_probs": 5, "n_predict": 5,
                            "temperature": 0, "cache_prompt": True
                        }
                        async with session.post(f"{self.endpoint}/completion", json=payload_native,
                                              timeout=aiohttp.ClientTimeout(total=20)) as resp:
                            if resp.ok:
                                data = await resp.json()
                                probs_list = data.get("completion_probabilities", [])
                                if probs_list:
                                    extracted = []
                                    for tp in probs_list:
                                        if "probs" in tp and tp["probs"]:
                                            top_prob = tp["probs"][0].get("prob", 0)
                                            if top_prob > 0:
                                                extracted.append(math.log(max(top_prob, 1e-10)))
                                    n_total_tokens = len(extracted)
                                    if extracted:
                                        log_probs = extracted
                                        method_used = "/completion"
                    except Exception as e:
                        logger.debug(f"Perplexity /completion failed: {e}")

                # ── Strategy 3: /v1/completions with echo (least likely to work) ──
                if not log_probs:
                    try:
                        context_lines = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in context_messages[-4:]]
                        full_prompt = ("\n".join(context_lines) + "\nUser: " + text) if context_lines else ("User: " + text)
                        payload_v1 = {
                            "prompt": full_prompt, "max_tokens": 1,
                            "echo": True, "logprobs": 5, "temperature": 0
                        }
                        async with session.post(f"{self.endpoint}/v1/completions", json=payload_v1,
                                              timeout=aiohttp.ClientTimeout(total=20)) as resp:
                            if resp.ok:
                                data = await resp.json()
                                choices = data.get("choices", [])
                                if choices:
                                    lp_data = choices[0].get("logprobs", {})
                                    token_lps = lp_data.get("token_logprobs", [])
                                    all_lps = [lp for lp in token_lps if lp is not None]
                                    n_total_tokens = len(all_lps)
                                    if all_lps:
                                        log_probs = all_lps
                                        method_used = "/v1/completions+echo"
                    except Exception as e:
                        logger.debug(f"Perplexity /v1/completions failed: {e}")

            if not log_probs:
                if self.trace_callback:
                    await self.trace_callback({
                        "type": "system_trace", "label": "calc_perplexity",
                        "duration_ms": round((time.monotonic() - t0) * 1000),
                        "summary": f"FAILED: 0 tokens scored via all 3 strategies",
                        "details": {"endpoint": self.endpoint}
                    })
                return None

            mean_lp = sum(log_probs) / len(log_probs)
            surprise = round(min(1.0, (-mean_lp) / 5.0), 4)
            logprob_variance = round(sum((lp - mean_lp) ** 2 for lp in log_probs) / max(len(log_probs) - 1, 1), 4) if len(log_probs) > 1 else 0.0
            
            if self.trace_callback:
                await self.trace_callback({
                    "type": "system_trace", "label": "calc_perplexity",
                    "duration_ms": round((time.monotonic() - t0) * 1000),
                    "summary": f"Surprise = {surprise} | LP Var = {logprob_variance:.3f} ({len(log_probs)} tokens via {method_used})",
                    "details": {
                        "target_text": text[:100], "normalized_surprise": surprise,
                        "logprob_variance": logprob_variance, "n_tokens_scored": len(log_probs),
                        "mean_logprob": round(mean_lp, 4), "method": method_used,
                    }
                })
            return {"surprise": surprise, "logprob_variance": logprob_variance}
            
        except Exception as e:
            logger.warning(f"Perplexity error: {e}")
            return None

    # ── LOGICAL / BACKGROUND TASKS (STRICT SYSTEM PROMPTING) ──

    async def reconcile_epistemic_conflict(self, concept: str, web_data: str, latent: dict, vectors) -> dict:
        """Reconcile conflict between the LLM's innate knowledge and external web data.
        Computes semantic + statistical conflict, then merges/overrides/balances accordingly."""
        t0 = time.monotonic()
        sys_man = _compute_manifold(latent, "internal")
        
        sys_prompt = f"{_state_conditioning(latent)}\nYou are the Epistemic Reconciliation Module. Define the concept accurately."
        prior_text = (await self._call([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Define exactly what '{concept}' is in one brief sentence. No preamble."}
        ], manifold=sys_man, label="extract_innate_prior")).strip()
        
        vec_prior = await vectors.embed(prior_text)
        vec_web   = await vectors.embed(web_data)
        cos_sim = max(-1.0, min(1.0, sum(x * y for x, y in zip(vec_prior, vec_web))))
        semantic_conflict = round(1.0 - cos_sim, 4)
        
        perplexity_result = await self.compute_perplexity(web_data, context_messages=[])
        statistical_conflict = perplexity_result["surprise"] if perplexity_result else None
        if statistical_conflict is None: statistical_conflict = 0.5 
        total_conflict = round((semantic_conflict + statistical_conflict) / 2.0, 4)

        # ── Code-branched reconciliation: decision is computed, LLM only synthesizes text ──
        sys_reconcile = f"{_state_conditioning(latent)}\nYou are the Epistemic Reconciliation Module."
        if total_conflict < 0.4:
            task = f"These two sources agree (conflict {total_conflict:.2f} / range [0,1]). Merge them into one concise fact.\nSource A: {prior_text}\nSource B: {web_data[:500]}\n\nReturn ONLY valid JSON:\n{{\"reconciled_fact\": \"The merged fact.\"}}"
        elif total_conflict > 0.6:
            task = f"External data overrides prior (conflict {total_conflict:.2f} / range [0,1]). Restate this web data as a concise storable belief.\nConcept: {concept}\nWeb Data: {web_data[:500]}\n\nReturn ONLY valid JSON:\n{{\"reconciled_fact\": \"The fact from web data.\"}}"
        else:
            task = f"Partial conflict detected (conflict {total_conflict:.2f} / range [0,1]). Produce a balanced fact that weights both sources.\nSource A (innate): {prior_text}\nSource B (web): {web_data[:500]}\n\nReturn ONLY valid JSON:\n{{\"reconciled_fact\": \"The balanced fact.\"}}"
        try:
            res = await self._call([{"role": "system", "content": sys_reconcile}, {"role": "user", "content": task}], manifold=sys_man, label="epistemic_compression")
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return { "total_conflict": total_conflict, "semantic_conflict": semantic_conflict, "statistical_conflict": statistical_conflict, "innate_prior": prior_text, "reconciled_fact": str(data.get("reconciled_fact", web_data)) }
        except Exception: pass
        return { "total_conflict": total_conflict, "semantic_conflict": semantic_conflict, "statistical_conflict": statistical_conflict, "innate_prior": prior_text, "reconciled_fact": web_data }

    # ── EPISODIC MEMORY ENCODING ──

    async def encode_memory(self, text: str, latent: dict, mode: str = "input", feedback: dict = None) -> str:
        """Encode text as a factual memory note via the LLM (memory recorder role).
        
        The LLM acts as a memory recorder — it writes a third-person factual note
        about what was said. It does NOT answer questions or perform requests.
        
        Modes:
          'input': record what the user said
          'output': record what the assistant said
        
        Word budget is capped at 3× the original text length to prevent inflation.
        Uses 'recorder' task profile: low temperature, tight budget.
        """
        manifold = _compute_manifold(latent, "recorder", feedback=feedback)
        s = latent.get("surprise", 0.3)
        v = latent.get("valence", 0.0)
        vel = latent.get("velocity", 0.3)
        
        # Word budget scales with surprise, modulated by energy
        # BUT capped by input length — a 1-word input cannot become a 30-word memory
        fb = feedback or {}
        energy_scale = max(0.4, 0.5 + fb.get("energy", 1.0) * 0.5)
        base_budget = int((15 + s * 35) * energy_scale)
        orig_words = len(text.split())
        # Cap: memory can be at most 3x the original, minimum 4 words
        max_budget = max(4, orig_words * 3)
        word_budget = min(base_budget, max_budget)
        
        if mode == "input":
            sys_prompt = (
                "You are a memory recorder. Your ONLY job is to write a short factual note about what the user said. "
                "Do NOT answer their question. Do NOT perform their request. Do NOT add information they didn't say. "
                "Just record what they said in third person, like a note."
            )
            user_prompt = f'Record this: "{text}"'
        else:
            sys_prompt = (
                "You are a memory recorder. Write a short factual note about what the assistant said. "
                "Do NOT add new information. Just summarize what was said, in third person."
            )
            user_prompt = f'Record this: "{text[:300]}"'
        
        # Use output manifold for temperature/top_p
        mem_manifold = dict(manifold)
        
        try:
            result = (await self._call(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                manifold=mem_manifold, label=f"encode_memory_{mode}"
            )).strip().strip('"').strip()
            return result if len(result) > 5 else ""
        except Exception:
            return ""

    # ── PERCEPTUAL TRANSDUCTIONS ──

    async def parse_subtext(self, text: str, latent: dict) -> str:
        """Pragmatic parser: extract what was MEANT beneath the words.
        'I guess that makes sense' → 'Reluctant agreement with doubt'.
        The LLM translates social language to direct language. Physics uses the embedding difference."""
        s, v = latent.get("surprise", 0.3), latent.get("valence", 0.0)
        sys_prompt = (
            "Restate what the person actually means beneath the words. 8 words max. "
            "Strip politeness, sarcasm, deflection. Be direct.\n"
            f"System state: Surprise {s:.2f}, Valence {v:+.2f}"
        )
        try:
            return (await self._call(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f'They said: "{text}"'}],
                manifold=_compute_manifold(latent, "internal", feedback=self._feedback),
                label="parse_subtext"
            )).strip().strip('"').strip()
        except Exception:
            return ""

    async def elaborate_perception(self, text: str, latent: dict) -> str:
        """Sensory elaborator: enrich raw input with interpreted meaning.
        State conditioning happens through temperature (surprise-driven), not prompt values."""
        s = latent.get("surprise", 0.3)
        try:
            return (await self._call(
                [{"role": "system", "content": "Restate the core meaning briefly."},
                 {"role": "user", "content": f'"{text}"'}],
                manifold=_compute_manifold(latent, "internal", feedback=self._feedback),
                label="elaborate_perception"
            )).strip().strip('"').strip()
        except Exception:
            return ""

    async def encode_expectation(self, conversation: list, latent: dict) -> str:
        """Prediction verbalizer: 5-word label of the expected response TYPE.
        Not content — form. 'Technical follow-up question' or 'Emotional pushback'.
        Creates an embedding for pragmatic-level surprise measurement."""
        s, v = latent.get("surprise", 0.3), latent.get("valence", 0.0)
        recent = "\n".join([f"{'User' if m['role']=='user' else 'Asst'}: {m['content'][:150]}" for m in conversation[-3:]])
        try:
            return (await self._call(
                [{"role": "system", "content": (
                    "Read the conversation and predict the TYPE of the user's next message. "
                    "Output a short label like: 'follow-up question', 'agreement', 'topic change', 'clarification request', 'emotional response'. "
                    "Output ONLY the label."
                )},
                 {"role": "user", "content": f"Conversation:\n{recent}"}],
                manifold=_compute_manifold(latent, "label", feedback=self._feedback),
                label="encode_expectation"
            )).strip().strip('"').strip()
        except Exception:
            return ""

    # ── OFFLINE TRANSDUCTIONS (idle/dream/compression) ──

    async def generate_dream(self, memories: list, latent: dict) -> str:
        """Dream generator: synthesize conflicting memories into a novel connection.
        Not a summary — a recombination. Creates new vector positions in embedding space."""
        s = latent.get("surprise", 0.3)
        mem_texts = "\n".join([f"- {m[:150]}" for m in memories[:3]])
        sys_prompt = (
            "These are separate memories. Find what connects them. "
            "Write ONE sentence that captures the hidden link."
        )
        try:
            return (await self._call(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Memories:\n{mem_texts}"}],
                manifold=_compute_manifold(latent, "creative", feedback=self._feedback),
                label="dream_synthesis"
            )).strip().strip('"').strip()
        except Exception:
            return ""

    async def compress_memory(self, memory_text: str, word_budget: int, latent: dict) -> str:
        """Compression codec: re-encode a memory at a tighter budget.
        Old memories naturally become terser and more schematic over time.
        EOS bias controls length mechanically."""
        try:
            result = (await self._call(
                [{"role": "system", "content": "Compress this memory. Preserve the core meaning."},
                 {"role": "user", "content": memory_text}],
                manifold=_compute_manifold(latent, "compression", feedback=self._feedback),
                label="compress_memory"
            )).strip().strip('"').strip()
            return result if len(result) > 3 else ""
        except Exception:
            return ""

    async def generate_reconsolidation_diff(self, old_text: str, new_text: str, latent: dict = None) -> str:
        """Semantic diff: one-line description of what changed during reconsolidation.
        Produces a vector that captures the DIRECTION of memory drift."""
        if latent is None: latent = {"surprise": 0.3, "valence": 0.0, "velocity": 0.3}
        try:
            return (await self._call(
                [{"role": "system", "content": "Describe in one short phrase how the memory changed. What shifted?"},
                 {"role": "user", "content": f"Before: {old_text}\nAfter: {new_text}\n\nWhat changed:"}],
                manifold=_compute_manifold(latent, "internal", feedback=self._feedback),
                label="reconsolidation_diff"
            )).strip().strip('"').strip()
        except Exception:
            return ""

    async def topic_summary_module(self, recent_turns: list, latent: dict) -> str:
        """Generate a 1-2 sentence topic summary of the recent conversation for context."""
        sys_prompt = f"{_state_conditioning(latent)}\nYou are the Context Summarization Module."
        turns_text = "\n".join([f"{t['role'].upper()}: {t['content'][:200]}" for t in recent_turns[-4:]])
        user_prompt = f"Recent conversation:\n{turns_text}\n\nIn 1-2 sentences, describe what this conversation has been about. Plain language."
        return (await self._call([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}], manifold=_compute_manifold(latent, "summary"), label="topic_summary")).strip().split("\n")[0].strip()

    async def diagnose_cycle(self, digest: dict, recall_history: list = None, latent: dict = None) -> str:
        """LLM diagnostic: feed cycle trace data + recall history for analysis."""
        if latent is None: latent = {"surprise": 0.3, "valence": 0.0, "velocity": 0.3}
        lines = []
        for key, val in digest.items():
            if isinstance(val, dict):
                inner = ", ".join(f"{k}={v}" for k, v in val.items())
                lines.append(f"{key}: {inner}")
            elif isinstance(val, list):
                lines.append(f"{key}: [{', '.join(str(v) for v in val[:8])}]")
            else:
                lines.append(f"{key}: {val}")
        
        # Add recall dynamics summary
        if recall_history:
            lines.append("\nRECALL DYNAMICS (most-recalled memories):")
            for rh in recall_history[:5]:
                drift_dir = "+" if rh.get("valence_drift", 0) > 0 else ""
                lines.append(
                    f"  [{rh.get('recall_count',0)}x] V_orig={rh.get('original_valence',0):.2f} "
                    f"V_now={rh.get('valence_at',0):.2f} drift={drift_dir}{rh.get('valence_drift',0):.2f} "
                    f"str={rh.get('encoding_strength',0):.2f}: {(rh.get('memory_text',''))[:60]}"
                )
        
        data_block = "\n".join(lines)
        sys_prompt = (
            "You are a diagnostic module analyzing one processing cycle of a Bayesian predictive coding system. "
            "Pay attention to recall dynamics: rumination loops (same negative memory recalled repeatedly), "
            "valence drift (memories shifting from original encoding), covariance inflation from contradictory recall, "
            "and energy depletion from excessive recall. Be concise, clinical, and actionable."
        )
        user_prompt = (
            f"CYCLE TELEMETRY:\n{data_block}\n\n"
            f"In 2-4 sentences: What happened this cycle? Are any recall patterns concerning? What should the system attend to next?"
        )
        try:
            # Static manifold — diagnostic needs consistent, unconstrained generation
            # regardless of system state (high temp + wide nucleus for thorough analysis)
            static_manifold = {
                "temperature": 1.0, "top_p": 0.95, "min_p": 0.01,
                "top_k": 40, "max_tokens": 600, "repetition_penalty": 0
            }
            return (await self._call(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                manifold=static_manifold,
                label="diagnostic"
            )).strip()
        except Exception:
            return ""

    # ── PREDICTION TASKS ──

    async def predict_trajectory(self, conversation: list, latent: dict, user_model: dict = None, state_conditioning: bool = False) -> str:
        """Pre-Action Prediction: generate hypothetical reply, then predict user's next message."""
        s, v, vel = latent.get("surprise", 0.3), latent.get("valence", 0.0), latent.get("velocity", 0.3)

        # ── Step 1: Generate hypothetical assistant response ──
        hyp_manifold = _compute_manifold(latent, "hypothetical")
        
        # Build context — serialize whatever the resolver provided
        context_lines = []
        for m in conversation[-4:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            context_lines.append(f"{role}: {m.get('content', '')[:300]}")
        
        hyp_sys = _state_conditioning(latent) + "\n" if state_conditioning else ""
        hyp_sys += "You are an AI assistant. Respond to the user's latest message."
        
        messages = [
            {"role": "system", "content": hyp_sys},
            {"role": "user", "content": "\n\n".join(context_lines)}
        ]

        try:
            hypothetical_reply = (await self._call(messages, manifold=hyp_manifold, label="predict_hypothetical_reply")).strip()
        except Exception:
            hypothetical_reply = ""

        if not hypothetical_reply:
            return ""

        # ── Step 2: Predict user's next message ──
        pred_manifold = _compute_manifold(latent, "predict", feedback=self._feedback)
        
        # Full conversation including hypothetical reply
        full_lines = []
        for m in conversation[-3:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            full_lines.append(f"{role}: {m.get('content', '')[:300]}")
        full_lines.append(f"Assistant: {hypothetical_reply[:300]}")
        
        pred_sys = ""
        if state_conditioning:
            pred_sys = _state_conditioning(latent) + "\n"
        pred_sys += (
            "You are predicting what the USER will say next. "
            "Read the conversation below and write the user's likely next message. "
            "Output ONLY what the user would say — not what the assistant would say."
        )
        
        pred_messages = [
            {"role": "system", "content": pred_sys},
            {"role": "user", "content": "\n\n".join(full_lines)}
        ]

        try:
            result = (await self._call(pred_messages, manifold=pred_manifold, label="predict_pre_action")).replace('"', '').strip()
            return result
        except Exception:
            return ""

    async def predicted_input_module(self, conversation: list, latent: dict, actual_output: str, user_model: dict = None, state_conditioning: bool = False) -> str:
        """Post-Action Prediction: predict user's next message given what was just said."""
        manifold = _compute_manifold(latent, "predict", feedback=self._feedback)

        context_lines = []
        for m in conversation[-3:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            context_lines.append(f"{role}: {m.get('content', '')[:300]}")
        context_lines.append(f"Assistant: {actual_output[:300]}")
        
        pred_sys = ""
        if state_conditioning:
            pred_sys = _state_conditioning(latent) + "\n"
        pred_sys += (
            "You are predicting what the USER will say next. "
            "Read the conversation below and write the user's likely next message. "
            "Output ONLY what the user would say — not what the assistant would say."
        )
        
        messages = [
            {"role": "system", "content": pred_sys},
            {"role": "user", "content": "\n\n".join(context_lines)}
        ]

        try:
            result = (await self._call(messages, manifold=manifold, label="predict_post_action")).replace('"', '').strip()
            return result if result else actual_output[:40]
        except Exception: return actual_output[:40]

    # ── WORKING MEMORY COMPRESSION ──

    async def compress_working_memory(self, context_messages: list, latent: dict, 
                                       word_budget: int = 80, state_conditioning: bool = False) -> str:
        """Compress context messages into lossy working memory text.
        
        Accepts generic [{role, content}] from pipeline resolver — could be
        raw conversation, encoded memories, or any configured combination.
        Returns a single text block. Caller assembles final context.
        """
        if not context_messages:
            return ""
        
        # Build transcript from whatever the resolver provided
        lines = []
        for m in context_messages[-12:]:
            if isinstance(m, dict):
                role = "User" if m.get("role") == "user" else "Assistant"
                content = m.get("content", "")[:400]
            elif isinstance(m, str):
                role = "User"
                content = m[:400]
            else:
                continue
            if content.strip():
                lines.append(f"{role}: {content}")
        
        if not lines:
            return ""
        
        transcript = "\n".join(lines)
        manifold = _compute_manifold(latent, "compression", feedback=self._feedback)
        
        sys_parts = []
        if state_conditioning:
            sys_parts.append(_state_conditioning(latent))
        sys_parts.append(
            "You are a note-taker. Read the conversation transcript below and write a brief summary of what was discussed. "
            "Include what the user asked and what the assistant replied. "
            "Do NOT continue the conversation. Do NOT add new content. Just summarize what happened."
        )
        
        try:
            compressed = await self._call(
                [{"role": "system", "content": "\n".join(sys_parts)},
                 {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}],
                manifold=manifold, label="compress_working_memory"
            )
            return compressed.strip() if compressed else ""
        except Exception:
            pass
        
        # Fallback: last 4 items as-is
        fallback = []
        for m in context_messages[-4:]:
            if isinstance(m, dict):
                role = "User" if m.get("role") == "user" else "Assistant"
                fallback.append(f"{role}: {m.get('content', '')[:150]}")
        return "\n".join(fallback)

    async def hyde_validate(self, memory_text: str, latent: dict = None) -> str:
        """HyDE validation: generate what the LLM independently knows about this topic.
        
        The caller computes embedding similarity between the LLM's independent
        knowledge and the actual memory. High similarity = memory aligns with
        world knowledge → BOOST. Low similarity = novel → neutral.
        
        Key: we DON'T feed the memory content. We extract the TOPIC and ask
        the LLM what it knows independently. Otherwise it just parrots/expands.
        
        Returns: hypothesis text (caller does embedding comparison)
        """
        if not memory_text or len(memory_text) < 10:
            return ""
        if latent is None: latent = {"surprise": 0.3, "valence": 0.0, "velocity": 0.3}
        
        # Extract a short topic from the memory (first ~30 chars, strip prefixes)
        topic = memory_text
        for prefix in ["The user said:", "The user stated:", "User said:", "User stated:", 
                       "User confirmed:", "User asked:", "The User said:", "The User stated:",
                       "Assistant said:", "The Assistant said:", "The assistant said:",
                       "The assistant stated:", "Record this:"]:
            if topic.startswith(prefix):
                topic = topic[len(prefix):].strip().strip('"').strip()
                break
        topic = topic[:80].split("\n")[0].strip()  # first line, max 80 chars
        
        if len(topic) < 5:
            return ""
        
        try:
            hypothesis = await self._call(
                [{"role": "system", "content": "Answer briefly from your own knowledge."},
                 {"role": "user", "content": f"What do you know about: {topic}"}],
                manifold=_compute_manifold(latent, "internal", feedback=self._feedback),
                label="hyde_validate"
            )
            return hypothesis.strip() if hypothesis else ""
        except Exception:
            return ""

    # ── PRIMARY OUTPUT ──

    async def output_module(self, conversation: list, latent: dict, dynamic_axes: dict, error_result: dict, enriched: Optional[dict] = None, topic_summary: str = "", feedback: dict = None, user_model: dict = None, state_conditioning: bool = False) -> str:
        """Primary output: LLM generates the assistant's response.
        
        conversation is whatever the pipeline resolver assembled — could be
        encoded memories only, WM + LT, raw conversation, or any combination.
        We serialize it faithfully without assuming structure.
        """
        is_silence = error_result.get("is_silence", False)
        manifold = _compute_manifold(latent, "output", feedback=feedback)
        
        sys_parts = []
        if state_conditioning:
            sys_parts.append(_state_conditioning(latent))
        sys_parts.append("You are an AI assistant.")
        if topic_summary: sys_parts.append(f"Topic: {topic_summary}")
        
        messages = [{"role": "system", "content": "\n".join(sys_parts)}]
        
        # Serialize all resolved context into a single user turn
        context_lines = []
        for msg in conversation:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:
                context_lines.append(f"{role}: {content}")
        
        if context_lines:
            messages.append({"role": "user", "content": "\n\n".join(context_lines)})
            
        return await self._call(messages, manifold=manifold, label="output")