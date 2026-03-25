"""
diagnostic.py — Cycle diagnostic: full trace dump + LLM analysis.

Collects ALL trace events from a processing cycle, assembles them into
complete text (nothing cherry-picked), and optionally sends to an LLM
for interpretation.

The report IS the trace — every broadcast event, in order, with full details.
"""

import json
import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CycleDiagnostic:

    def __init__(self):
        self.reset(0)

    def reset(self, cycle_number: int = 0):
        self._traces: list = []
        self._cycle: int = cycle_number
        self._raw_input: str = ""
        self._raw_output: str = ""
        self._input_memory: str = ""
        self._output_memory: str = ""
        self._prior_latent: dict = {}
        self._posterior_latent: dict = {}
        self._mem_context: list = []
        self._raw_conversation: list = []
        self._prev_prediction: str = ""

    # ── Data feeds (called from brain.py) ──

    def begin_cycle(self, cycle_number: int, raw_input: str,
                    prior_latent: dict, raw_conversation: list = None,
                    prev_prediction: str = ""):
        self.reset(cycle_number)
        self._raw_input = raw_input
        self._prior_latent = dict(prior_latent)
        self._raw_conversation = list(raw_conversation or [])
        self._prev_prediction = prev_prediction

    def ingest(self, trace: dict):
        """Receive any broadcast trace event."""
        if not trace or not isinstance(trace, dict):
            return
        self._traces.append(trace)

    def set_input_memory(self, text: str):
        self._input_memory = text

    def set_memory_context(self, context: list):
        self._mem_context = list(context) if context else []

    def set_output(self, raw_reply: str, output_memory: str,
                   posterior_latent: dict = None):
        self._raw_output = raw_reply or ""
        self._output_memory = output_memory or ""
        if posterior_latent:
            self._posterior_latent = dict(posterior_latent)

    # Kept for backward compat — now no-ops
    def set_perception(self, **kwargs): pass
    def set_prediction(self, text: str): self._prev_prediction = text

    # ────────────────────────────────────────────────────────
    # REPORT: Full trace dump
    # ────────────────────────────────────────────────────────

    def build_report(self) -> str:
        """Assemble ALL trace data into structured text. Nothing omitted."""
        R = []
        R.append(f"{'='*60}")
        R.append(f"CYCLE #{self._cycle} — FULL TRACE DUMP")
        R.append(f"{'='*60}\n")

        # ── RAW DATA ──
        R.append("── RAW INPUT/OUTPUT ──\n")
        R.append(f'USER INPUT: "{self._raw_input}"')
        R.append(f'STORED AS INPUT MEMORY: "{self._input_memory}"')
        if self._prev_prediction:
            R.append(f'PREVIOUS PREDICTION: "{self._prev_prediction}"')
        R.append(f'SYSTEM OUTPUT: "{self._raw_output}"')
        R.append(f'STORED AS OUTPUT MEMORY: "{self._output_memory}"')
        R.append("")

        # ── LATENT STATE ──
        if self._prior_latent and self._posterior_latent:
            R.append("── LATENT STATE ──\n")
            for dim in ["surprise", "valence", "velocity"]:
                p = self._prior_latent.get(dim, 0)
                q = self._posterior_latent.get(dim, 0)
                R.append(f"  {dim}: {p:.3f} → {q:.3f} (Δ={q-p:+.3f})")
            R.append("")

        # ── MEMORY CONTEXT vs ACTUAL CONVERSATION ──
        R.append("── MEMORY CONTEXT (what the LLM saw) ──\n")
        if self._mem_context:
            for i, m in enumerate(self._mem_context):
                role = m.get("role", "?").upper()
                R.append(f'  [{i+1}] [{role}] "{m.get("content", "")}"')
        else:
            R.append("  (empty — LLM saw no conversation context)")
        R.append("")

        if self._raw_conversation:
            R.append(f"── ACTUAL CONVERSATION (last {min(8, len(self._raw_conversation))}) ──\n")
            for i, m in enumerate(self._raw_conversation[-8:]):
                role = m.get("role", "?").upper()
                R.append(f'  [{i+1}] [{role}] "{m.get("content", "")}"')
            R.append("")

        # ── ALL TRACE EVENTS (in order) ──
        R.append("── TRACE EVENTS ──\n")
        for i, t in enumerate(self._traces):
            label = t.get("label", t.get("type", "?"))
            summary = t.get("summary", "")
            duration = t.get("duration_ms", "")
            
            header = f"[{i+1}] {label}"
            if duration:
                header += f" ({duration}ms)"
            R.append(header)
            if summary:
                R.append(f"  {summary}")
            
            # Dump all details
            details = t.get("details", {})
            if details:
                for k, v in details.items():
                    if isinstance(v, dict):
                        R.append(f"  {k}:")
                        for dk, dv in v.items():
                            R.append(f"    {dk}: {dv}")
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        R.append(f"  {k}: ({len(v)} items)")
                        for j, item in enumerate(v[:8]):
                            parts = [f"{ik}={iv}" for ik, iv in item.items() if ik != "_vec"]
                            R.append(f"    [{j}] {', '.join(parts[:8])}")
                    else:
                        R.append(f"  {k}: {v}")
            
            # LLM trace: show sampling params and response preview
            if t.get("type") == "llm_trace":
                sampling = t.get("sampling", {})
                if sampling:
                    parts = [f"{k}={v}" for k, v in sampling.items()]
                    R.append(f"  sampling: {', '.join(parts)}")
                resp = t.get("response", "")
                if resp:
                    R.append(f'  response: "{resp[:200]}"')
                budget = t.get("budget", {})
                if budget:
                    R.append(f"  budget: {budget}")
            
            R.append("")

        return "\n".join(R)

    async def analyze(self, llm_client=None, endpoint: str = None) -> str:
        """Send full trace dump to LLM for interpretation."""
        report = self.build_report()
        sys_prompt = (
            "You are a systems diagnostician analyzing a Bayesian predictive coding conversational AI. "
            "The trace dump shows every processing step from one cycle.\n\n"
            "Focus on:\n"
            "1. MEMORY FIDELITY: Is the system remembering what was actually said, or distorting/inflating it?\n"
            "2. CONTEXT QUALITY: Does the LLM see a coherent conversation, or is it seeing redundant/missing memories?\n"
            "3. PREDICTION ACCURACY: Were predictions correct? Is the system improving?\n"
            "4. BEHAVIORAL COHERENCE: Does the output match what was asked?\n"
            "5. ANOMALIES: What needs attention?\n\n"
            "Write 4-6 sentences. Reference specific values and quotes. Diagnose root causes."
        )

        # Try llm_client first
        if llm_client and hasattr(llm_client, '_call'):
            try:
                return await llm_client._call(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": report}],
                    manifold={"temperature": 1.0, "top_p": 0.95, "min_p": 0.01, "top_k": 40, "repetition_penalty": 0, "max_tokens": 600},
                    label="diagnostic"
                )
            except Exception as e:
                logger.warning(f"Diagnostic LLM call failed: {e}")

        # Fallback: direct HTTP
        if endpoint:
            try:
                import aiohttp
                payload = {
                    "model": "local",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": report}
                    ],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "min_p": 0.01,
                    "top_k": 40,
                    "repetition_penalty": 0,
                    "max_tokens": 600, 
                    "stream": False
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{endpoint}/v1/chat/completions",
                                          json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.ok:
                            data = await resp.json()
                            return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Diagnostic HTTP fallback failed: {e}")

        return ""