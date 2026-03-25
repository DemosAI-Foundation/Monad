"""
main.py — FastAPI server + WebSocket hub.

Startup lifecycle:
  1. Initialize SQLite (memory.py)
  2. Initialize ChromaDB + sentence-transformers (embeddings.py)
  3. Restore brain state from persistence
  4. Wire LLM trace callback to WebSocket broadcast
  5. Start idle loop (Langevin dynamics)

All user input arrives via WebSocket, passes through brain.process_input(),
and results stream back as JSON messages (chat, traces, state updates).
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from brain import BayesianBrain
from memory import Memory
from llm import LLMClient
from embeddings import EmbeddingStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

brain   = BayesianBrain()
memory  = Memory()
llm     = LLMClient(endpoint="http://127.0.0.1:8083")
vectors = EmbeddingStore()

class Hub:
    def __init__(self): self._clients: set[WebSocket] = set()
    async def connect(self, ws: WebSocket): await ws.accept(); self._clients.add(ws)
    def disconnect(self, ws: WebSocket): self._clients.discard(ws)
    async def broadcast(self, data: dict):
        if not self._clients: return
        payload = json.dumps(data)
        dead: set[WebSocket] = set()
        for ws in self._clients:
            try: await ws.send_text(payload)
            except Exception: dead.add(ws)
        self._clients -= dead

hub = Hub()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.init()
    await vectors.init()
    await brain.restore_from_memory(memory)   
    llm.trace_callback = hub.broadcast
    brain.start_idle_loop(llm, memory, hub.broadcast, vectors)
    logger.info("System ready.")
    yield
    if brain._idle_task:
        brain._idle_task.cancel()
        try: await brain._idle_task
        except asyncio.CancelledError: pass

app = FastAPI(title="Bayesian Brain Continuous v1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    await ws.send_text(json.dumps({"type": "init", "data": brain.state_snapshot(), "messages": brain.messages, "llm_endpoint": llm.endpoint, "connected": await llm.test_connection()}))
    try:
        while True:
            raw  = await ws.receive_text()
            data = json.loads(raw)
            if data["type"] == "send": asyncio.create_task(brain.process_input(data["text"], llm, memory, hub.broadcast, vectors))
            elif data["type"] == "set_endpoint":
                llm.endpoint = data["endpoint"].rstrip("/")
                connected    = await llm.test_connection()
                await hub.broadcast({"type": "endpoint_status", "endpoint": llm.endpoint, "connected": connected})
    except WebSocketDisconnect: hub.disconnect(ws)

@app.get("/api/memory/predictions")
async def get_predictions(): return JSONResponse(await memory.get_predictions(50))
@app.get("/api/memory/beliefs")
async def get_beliefs(): return JSONResponse(await memory.get_latent_log(100))
@app.get("/api/memory/errors")
async def get_errors(): return JSONResponse(await memory.get_error_stats())
@app.get("/api/memory/silence")
async def get_silence(): return JSONResponse(await memory.get_silence_log(30))
@app.get("/api/memory/vectors")
async def get_vectors(): return JSONResponse(await memory.get_enriched_log(30))
@app.get("/api/vector/stats")
async def get_vector_stats():
    try:
        return JSONResponse({"messages": vectors._messages_col.count() if vectors._messages_col else 0, "concepts": vectors._concepts_col.count() if vectors._concepts_col else 0, "model": vectors._embed_model_name, "adaptive_threshold": round(vectors._sim_stats.threshold(), 3), "sim_stats": {"n": vectors._sim_stats.n, "mean": round(vectors._sim_stats._mean, 3), "std": round(vectors._sim_stats.std, 3)}})
    except Exception as e: return JSONResponse({"error": str(e)})

@app.get("/api/debug/perplexity")
async def debug_perplexity():
    """Test perplexity endpoint directly — hit this in your browser to diagnose."""
    import aiohttp as _aiohttp
    test_prompt = "User: hello how are you"
    results = {"endpoint": llm.endpoint, "test_prompt": test_prompt, "strategies": {}}
    
    if not llm.endpoint:
        return JSONResponse({"error": "No LLM endpoint configured"})
    
    async with _aiohttp.ClientSession() as session:
        # Strategy 1: /v1/completions
        try:
            async with session.post(f"{llm.endpoint}/v1/completions", json={"prompt": test_prompt, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                status = resp.status
                data = await resp.json() if resp.ok else await resp.text()
                if resp.ok and isinstance(data, dict):
                    choices = data.get("choices", [])
                    lp_data = choices[0].get("logprobs", {}) if choices else {}
                    token_lps = lp_data.get("token_logprobs", [])
                    tokens = lp_data.get("tokens", [])
                    results["strategies"]["v1_completions"] = {
                        "status": status, "ok": True,
                        "n_tokens": len(tokens), "n_logprobs": len([lp for lp in token_lps if lp is not None]),
                        "sample_tokens": tokens[:5], "sample_logprobs": token_lps[:5],
                        "response_keys": list(data.keys()),
                        "logprobs_keys": list(lp_data.keys()) if lp_data else "none"
                    }
                else:
                    results["strategies"]["v1_completions"] = {"status": status, "ok": False, "body": str(data)[:300]}
        except Exception as e:
            results["strategies"]["v1_completions"] = {"error": str(e)}
        
        # Strategy 2: /completion (native)
        try:
            async with session.post(f"{llm.endpoint}/completion", json={"prompt": test_prompt, "n_probs": 1, "n_predict": 0, "echo": True, "temperature": 0}, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                status = resp.status
                data = await resp.json() if resp.ok else await resp.text()
                if resp.ok and isinstance(data, dict):
                    probs = data.get("completion_probabilities") or data.get("tokens") or []
                    results["strategies"]["completion_native"] = {
                        "status": status, "ok": True,
                        "response_keys": list(data.keys()),
                        "n_prob_entries": len(probs),
                        "sample_entry": probs[0] if probs else "empty",
                        "sample_entry_keys": list(probs[0].keys()) if probs and isinstance(probs[0], dict) else "n/a"
                    }
                else:
                    results["strategies"]["completion_native"] = {"status": status, "ok": False, "body": str(data)[:300]}
        except Exception as e:
            results["strategies"]["completion_native"] = {"error": str(e)}
    
    return JSONResponse(results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)