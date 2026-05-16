"""LLM OptiCoach web app routes and session-scoped orchestration."""

import json
import os
import re
import secrets
import subprocess
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from utils.diagnostics import full_diagnostic
from utils.llm_coach import (
    DIAGNOSTIC_COACH_SYSTEM,
    REFEREE_SYSTEM,
    CoachRequest,
    build_diagnostic_prompt,
    build_referee_prompt,
    call_coach,
    run_inference_benchmark,
    run_routing_assertion,
)
from utils.diagnostics import benchmark_snapshot
from utils.platform_models import detect_platform, get_model_recommendations, get_solo_recommendation

app = FastAPI(title="LLM OptiCoach")
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", secrets.token_hex(32)),
    same_site="lax",
)
app.state.sessions: Dict[str, Dict[str, Any]] = {}


def get_session_state(request: Request) -> Dict[str, Any]:
    session_id = request.session.get("sid")
    if not session_id:
        session_id = secrets.token_urlsafe(24)
        request.session["sid"] = session_id

    sessions = app.state.sessions
    if session_id not in sessions:
        sessions[session_id] = {
            "last_diagnostic": None,
            "user_api_key": None,
            "user_base_url": "https://api.openai.com/v1",
            "user_model": "gpt-4o-mini",
            "last_benchmark": None,
            "benchmark_baseline": None,
            "benchmark_delta": None,
            "benchmark_note": None,
            "routing_assertion": None,
            "routing_assertion_note": None,
        }
    return sessions[session_id]


def build_benchmark_delta(baseline: Dict[str, Any], latest: Dict[str, Any]) -> Dict[str, Any]:
    def delta_pct(new_val: Optional[float], old_val: Optional[float]) -> Optional[float]:
        if new_val is None or old_val in (None, 0):
            return None
        return round(((new_val - old_val) / old_val) * 100.0, 2)

    return {
        "ttft_s": {
            "baseline": baseline.get("ttft_s"),
            "latest": latest.get("ttft_s"),
            "delta_pct": delta_pct(latest.get("ttft_s"), baseline.get("ttft_s")),
        },
        "tokens_per_second": {
            "baseline": baseline.get("tokens_per_second"),
            "latest": latest.get("tokens_per_second"),
            "delta_pct": delta_pct(latest.get("tokens_per_second"), baseline.get("tokens_per_second")),
        },
        "elapsed_s": {
            "baseline": baseline.get("elapsed_s"),
            "latest": latest.get("elapsed_s"),
            "delta_pct": delta_pct(latest.get("elapsed_s"), baseline.get("elapsed_s")),
        },
    }


def render_scan_results(request: Request, state: Dict[str, Any]):
    if not state.get("last_diagnostic"):
        return render_error(request, "No diagnostic report available yet. Please run a scan.", status_code=400)
    return templates.TemplateResponse(
        request=request,
        name="scan_results.html",
        context={
            "request": request,
            "report": state["last_diagnostic"],
            "benchmark": state.get("last_benchmark"),
            "benchmark_baseline": state.get("benchmark_baseline"),
            "benchmark_delta": state.get("benchmark_delta"),
            "benchmark_note": state.get("benchmark_note"),
            "routing_assertion": state.get("routing_assertion"),
            "routing_assertion_note": state.get("routing_assertion_note"),
            "prefill_base_url": state.get("user_base_url", "https://api.openai.com/v1"),
            "prefill_model": state.get("user_model", "gpt-4o-mini"),
        },
    )


def render_error(request: Request, error: str, status_code: int = 400):
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"request": request, "error": error},
        status_code=status_code,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return render_error(request, str(exc.detail), status_code=exc.status_code)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    state = get_session_state(request)
    if not state.get("last_diagnostic"):
        report = full_diagnostic(model_path=None)
        state["last_diagnostic"] = report
    return render_scan_results(request, state)


@app.post("/scan", response_class=HTMLResponse)
async def run_scan(request: Request, model_path: Optional[str] = Form(None)):
    state = get_session_state(request)
    report = full_diagnostic(model_path=model_path)
    state["last_diagnostic"] = report
    return render_scan_results(request, state)


@app.post("/coach", response_class=HTMLResponse)
async def get_coach(
    request: Request,
    user_notes: str = Form(...),
    api_key: str = Form(...),
    base_url: str = Form("https://api.openai.com/v1"),
    model: str = Form("gpt-4o-mini"),
):
    state = get_session_state(request)
    if not state.get("last_diagnostic"):
        return render_error(request, "Please run a diagnostic scan first", status_code=400)

    state["user_api_key"] = api_key
    state["user_base_url"] = base_url
    state["user_model"] = model

    prompt = build_diagnostic_prompt(state["last_diagnostic"], user_notes)
    req = CoachRequest(
        system_prompt=DIAGNOSTIC_COACH_SYSTEM,
        user_prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    response = call_coach(req)
    if response.success:
        return templates.TemplateResponse(
            request=request,
            name="coach_response.html",
            context={"request": request, "advice": response.response_text},
        )
    return render_error(request, f"Coach failed: {response.error}", status_code=500)


@app.post("/benchmark", response_class=HTMLResponse)
async def run_benchmark(
    request: Request,
    benchmark_prompt: str = Form(
        "Measure coding assistant response quality and speed for Jetson local developer workflow."
    ),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    action: str = Form("run"),
):
    state = get_session_state(request)
    if not state.get("last_diagnostic"):
        return render_error(request, "Please run a diagnostic scan first", status_code=400)

    if action == "set_baseline":
        if not state.get("last_benchmark"):
            return render_error(request, "Run a benchmark first before setting baseline", status_code=400)
        state["benchmark_baseline"] = state["last_benchmark"]
        state["benchmark_delta"] = None
        state["benchmark_note"] = "Baseline updated from latest benchmark run."
        return render_scan_results(request, state)

    if action == "clear_baseline":
        state["benchmark_baseline"] = None
        state["benchmark_delta"] = None
        state["benchmark_note"] = "Benchmark baseline cleared."
        return render_scan_results(request, state)

    resolved_api_key = (api_key or "").strip() or state.get("user_api_key")
    resolved_base_url = (base_url or "").strip() or state.get("user_base_url", "https://api.openai.com/v1")
    resolved_model = (model or "").strip() or state.get("user_model", "gpt-4o-mini")

    if not resolved_api_key:
        state["benchmark_note"] = "Add your API key, then click Run Benchmark."
        return render_scan_results(request, state)

    state["user_api_key"] = resolved_api_key
    state["user_base_url"] = resolved_base_url
    state["user_model"] = resolved_model

    benchmark_req = CoachRequest(
        system_prompt=DIAGNOSTIC_COACH_SYSTEM,
        user_prompt=benchmark_prompt,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=0.0,
        max_tokens=512,
    )

    pre_snapshot = benchmark_snapshot()
    result = run_inference_benchmark(benchmark_req, benchmark_prompt=benchmark_prompt)
    post_snapshot = benchmark_snapshot()
    result["system_snapshot_before"] = pre_snapshot
    result["system_snapshot_after"] = post_snapshot
    state["last_benchmark"] = result

    baseline = state.get("benchmark_baseline")
    if baseline and baseline.get("success") and result.get("success"):
        state["benchmark_delta"] = build_benchmark_delta(baseline, result)
        state["benchmark_note"] = "Compared latest run against your saved baseline."
    elif not baseline and result.get("success"):
        state["benchmark_baseline"] = result
        state["benchmark_delta"] = None
        state["benchmark_note"] = "First successful benchmark saved as baseline."
    else:
        state["benchmark_delta"] = None
        state["benchmark_note"] = "Benchmark run failed. Review error and retry."

    return render_scan_results(request, state)


@app.post("/routing-assert", response_class=HTMLResponse)
async def routing_assert(
    request: Request,
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    state = get_session_state(request)
    report = state.get("last_diagnostic")
    if not report:
        return render_error(request, "Please run a diagnostic scan first", status_code=400)

    resolved_api_key = (api_key or "").strip() or state.get("user_api_key")
    resolved_base_url = (base_url or "").strip() or state.get("user_base_url", "https://api.openai.com/v1")
    resolved_model = (model or "").strip() or state.get("user_model", "gpt-4o-mini")

    if not resolved_api_key:
        state["routing_assertion_note"] = "Add your API key, then click Run Routing Assertion Test."
        return render_scan_results(request, state)

    state["user_api_key"] = resolved_api_key
    state["user_base_url"] = resolved_base_url
    state["user_model"] = resolved_model

    serving = report.get("serving", {})
    expected_primary = serving.get("primary_model_candidate")
    expected_fallbacks = serving.get("fallback_chain", []) or []

    req = CoachRequest(
        system_prompt=DIAGNOSTIC_COACH_SYSTEM,
        user_prompt="Routing assertion",
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=0.0,
        max_tokens=80,
    )

    assertion = run_routing_assertion(
        req,
        expected_primary=expected_primary,
        expected_fallbacks=expected_fallbacks,
    )
    state["routing_assertion"] = assertion

    if assertion.get("success"):
        if assertion.get("verdict") == "pass":
            state["routing_assertion_note"] = "Assertion passed: responding model aligns with inferred chain."
        else:
            state["routing_assertion_note"] = "Assertion warning: response model did not clearly match inferred chain."
    else:
        state["routing_assertion_note"] = "Assertion failed: unable to verify runtime routing from API response."

    return render_scan_results(request, state)


@app.post("/referee", response_class=HTMLResponse)
async def referee_change(
    request: Request,
    proposed_change: str = Form(...),
    extra_context: str = Form(""),
):
    state = get_session_state(request)
    if not state.get("last_diagnostic") or not state.get("user_api_key"):
        return render_error(request, "Need a recent diagnostic + API key first", status_code=400)

    prompt = build_referee_prompt(proposed_change, state["last_diagnostic"], extra_context)
    req = CoachRequest(
        system_prompt=REFEREE_SYSTEM,
        user_prompt=prompt,
        model=state["user_model"],
        api_key=state["user_api_key"],
        base_url=state["user_base_url"],
        temperature=0.2,
    )
    response = call_coach(req)
    if response.success:
        return templates.TemplateResponse(
            request=request,
            name="referee_response.html",
            context={
                "request": request,
                "verdict": response.response_text,
                "proposed_change": proposed_change,
            },
        )
    return render_error(request, response.error or "Referee failed", status_code=500)


@app.get("/optimizations", response_class=HTMLResponse)
async def optimizations_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="optimizations.html",
        context={"request": request},
    )


# === BEGINNER-FIRST ONBOARDING WIZARD ROUTES ===

# Embedding model name fragments to exclude from inference model lists
_EMBED_KEYWORDS = ("embed", "nomic", "rerank", "bge", "e5-")


def _parse_size_gb(size_str: str) -> float:
    """Parse '4.7 GB' or '274 MB' into a float in GB."""
    try:
        val, unit = size_str.split()
        val = float(val)
        if unit.upper() == "MB":
            return val / 1024.0
        return val  # assume GB
    except Exception:
        return 0.0


def get_available_models() -> list:
    """Return available Ollama inference models sorted smallest to largest.

    Returns a list of dicts: {name, size_gb} sorted ascending by size.
    Embedding/rerank models are excluded.
    Falls back to placeholder names if Ollama is not available.
    """
    FALLBACK = [
        {"name": "nemotron-3-nano:4b", "size_gb": 2.8},
        {"name": "qwen2.5-coder:7b", "size_gb": 4.7},
    ]
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return FALLBACK

        lines = result.stdout.strip().split("\n")[1:]  # Skip header line
        models = []
        for line in lines:
            parts = line.split()
            # Columns: NAME  ID  SIZE  UNIT  MODIFIED...
            if len(parts) < 4:
                continue
            name = parts[0]
            # Skip embedding/rerank models
            if any(kw in name.lower() for kw in _EMBED_KEYWORDS):
                continue
            size_str = parts[2] + " " + parts[3]
            size_gb = _parse_size_gb(size_str)
            models.append({"name": name, "size_gb": size_gb})

        models.sort(key=lambda m: m["size_gb"])
        return models if models else FALLBACK
    except Exception:
        return FALLBACK


@app.get("/onboard", response_class=HTMLResponse)
async def onboard_step1(request: Request):
    """Step 1: Hardware check."""
    state = get_session_state(request)
    report = full_diagnostic(model_path=None)
    state["last_diagnostic"] = report

    pc = report.get("power_clocks", {})
    nvp = pc.get("nvpmodel", {})
    nvp_text = (nvp.get("stdout", "") + " " + nvp.get("stderr", "")).upper()
    maxn_ok = nvp.get("success", False) and "MAXN" in nvp_text

    # Detect platform
    plat = detect_platform()
    state["platform_info"] = plat

    # Parse real system metrics from tegrastats
    mg = report.get("memory_gpu", {})
    teg_stdout = mg.get("tegrastats", {}).get("stdout", "")

    hw: Dict[str, Any] = {
        "maxn_ok": maxn_ok,
        "platform_name": plat.get("display_name", "Unknown"),
        "is_jetson": plat.get("is_jetson", False),
        "model_budget_gb": plat.get("model_budget_gb", 0),
    }

    ram_m = re.search(r"RAM (\d+)/(\d+)MB", teg_stdout)
    if ram_m:
        hw["ram_used_gb"] = round(int(ram_m.group(1)) / 1024, 1)
        hw["ram_total_gb"] = round(int(ram_m.group(2)) / 1024, 1)
        hw["ram_pct"] = round(int(ram_m.group(1)) / int(ram_m.group(2)) * 100)

    swap_m = re.search(r"SWAP (\d+)/(\d+)MB", teg_stdout)
    if swap_m:
        hw["swap_used_mb"] = int(swap_m.group(1))
        hw["swap_total_gb"] = round(int(swap_m.group(2)) / 1024, 1)

    gpu_m = re.search(r"GR3D_FREQ (\d+)%", teg_stdout)
    if gpu_m:
        hw["gpu_load_pct"] = int(gpu_m.group(1))

    cpu_m = re.search(r"CPU \[([^\]]+)\]", teg_stdout)
    if cpu_m:
        cores = cpu_m.group(1).split(",")
        loads = []
        for c in cores:
            cm = re.match(r"(\d+)%@(\d+)", c.strip())
            if cm:
                loads.append(int(cm.group(1)))
        if loads:
            hw["cpu_cores"] = len(loads)
            hw["cpu_avg_pct"] = round(sum(loads) / len(loads))
            hw["cpu_freq_mhz"] = int(re.match(r"\d+%@(\d+)", cores[0].strip()).group(1))

    temps = re.findall(r"(\w+)@([\d.]+)C", teg_stdout)
    if temps:
        hw["thermals"] = {name: float(val) for name, val in temps}
        hw["temp_max"] = max(hw["thermals"].values())
        hw["temp_max_name"] = max(hw["thermals"], key=hw["thermals"].get)

    # ── Fallback metrics for non-Jetson (DGX Spark, GB10 OEM, etc.) ──
    # These systems don't have tegrastats; use /proc/meminfo + nvidia-smi.
    if "ram_pct" not in hw:
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            if total_kb:
                used_kb = total_kb - avail_kb
                hw["ram_total_gb"] = round(total_kb / 1024 / 1024, 1)
                hw["ram_used_gb"] = round(used_kb / 1024 / 1024, 1)
                hw["ram_pct"] = round(used_kb / total_kb * 100)
            swap_total = meminfo.get("SwapTotal", 0)
            swap_free = meminfo.get("SwapFree", 0)
            if swap_total:
                hw["swap_used_mb"] = (swap_total - swap_free) // 1024
                hw["swap_total_gb"] = round(swap_total / 1024 / 1024, 1)
        except Exception:
            pass

    if "gpu_load_pct" not in hw:
        try:
            nvsmi = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if nvsmi.returncode == 0:
                hw["gpu_load_pct"] = int(nvsmi.stdout.strip().split("\n")[0])
        except Exception:
            pass

    if "cpu_cores" not in hw:
        try:
            import multiprocessing
            hw["cpu_cores"] = multiprocessing.cpu_count()
            loadavg = open("/proc/loadavg").read().split()
            hw["cpu_avg_pct"] = round(float(loadavg[0]) / hw["cpu_cores"] * 100)
        except Exception:
            pass

    return templates.TemplateResponse(
        request=request,
        name="onboard_wizard.html",
        context={
            "request": request,
            "step": 1,
            "hardware": hw,
        },
    )


# ── Agent framework metadata ────────────────────────────────────────
_FRAMEWORKS = {
    "hermes": {
        "name": "Hermes / OpenClaw",
        "icon": "fa-solid fa-wand-magic-sparkles",
        "tip": (
            "Hermes works best with <strong>hermes3:8b</strong> as the tool-calling agent. "
            "It's purpose-built for structured function calls and agent personality."
        ),
        "fast_label": "Tool-Calling Agent",
        "heavy_label": "Reasoning Model",
    },
    "openwebui": {
        "name": "Open WebUI",
        "icon": "fa-solid fa-globe",
        "tip": (
            "Open WebUI supports function calling with most models. "
            "Any Qwen or Hermes model works well as the primary."
        ),
        "fast_label": "Primary Model",
        "heavy_label": "Secondary Model",
    },
    "langchain": {
        "name": "LangChain / LangGraph / CrewAI",
        "icon": "fa-solid fa-link",
        "tip": (
            "LangChain agents need reliable tool calling. "
            "Models with native function-call support (Qwen, Hermes) work best."
        ),
        "fast_label": "Agent Model",
        "heavy_label": "Reasoning Model",
    },
    "other": {
        "name": "Other / Custom",
        "icon": "fa-solid fa-code",
        "tip": (
            "Pick the best fast model for quick tasks and the best "
            "large model for complex reasoning. Any Ollama model works."
        ),
        "fast_label": "Fast Model",
        "heavy_label": "Deep Model",
    },
}


@app.post("/onboard/mode", response_class=HTMLResponse)
async def onboard_mode(request: Request):
    """Step 2: Choose how you want to use local AI."""
    state = get_session_state(request)
    plat = state.get("platform_info") or detect_platform()
    return templates.TemplateResponse(
        request=request,
        name="onboard_wizard.html",
        context={
            "request": request,
            "step": "mode",
            "platform": plat,
        },
    )


@app.post("/onboard/models", response_class=HTMLResponse)
async def onboard_models(request: Request, mode: str = Form("duo"), framework: str = Form("")):
    """Step 3: Model selection — adapts to chosen mode."""
    state = get_session_state(request)
    state["wizard_mode"] = mode
    state["wizard_framework"] = framework

    plat = state.get("platform_info") or detect_platform()
    state["platform_info"] = plat

    model_list = get_available_models()
    model_names = [m["name"] for m in model_list]

    fw_info = _FRAMEWORKS.get(framework, _FRAMEWORKS["other"]) if framework else None

    if mode == "solo":
        solo_rec = get_solo_recommendation(plat, model_list)
        state["model_recs_solo"] = solo_rec
        return templates.TemplateResponse(
            request=request,
            name="onboard_wizard.html",
            context={
                "request": request,
                "step": "models",
                "mode": mode,
                "platform": plat,
                "available_models": model_names,
                "model_sizes": {m["name"]: m["size_gb"] for m in model_list},
                "solo_rec": solo_rec,
            },
        )
    else:
        # duo or agent mode — both use two-model selection
        recs = get_model_recommendations(plat, model_list)
        state["model_recs"] = recs
        return templates.TemplateResponse(
            request=request,
            name="onboard_wizard.html",
            context={
                "request": request,
                "step": "models",
                "mode": mode,
                "framework": fw_info,
                "platform": plat,
                "available_models": model_names,
                "model_sizes": {m["name"]: m["size_gb"] for m in model_list},
                "recs": recs,
                "default_fast_model": recs["orchestrator"]["selected"],
                "default_smart_model": recs["heavy"]["selected"],
                "suggest_fast": recs["orchestrator"]["suggest_download"],
                "suggest_smart": recs["heavy"]["suggest_download"],
            },
        )


@app.post("/onboard/benchmark", response_class=HTMLResponse)
async def onboard_benchmark(
    request: Request,
    fast_model: str = Form(""),
    smart_model: str = Form(""),
    solo_model: str = Form(""),
):
    """Step 4: Show measurement starting point."""
    state = get_session_state(request)
    mode = state.get("wizard_mode", "duo")

    if mode == "solo":
        state["wizard_fast_model"] = solo_model
        state["wizard_smart_model"] = solo_model
    else:
        state["wizard_fast_model"] = fast_model
        state["wizard_smart_model"] = smart_model
    
    return templates.TemplateResponse(
        request=request,
        name="onboard_wizard.html",
        context={
            "request": request,
            "step": "benchmark",
            "mode": mode,
            "fast_model": state["wizard_fast_model"],
            "smart_model": state["wizard_smart_model"],
        },
    )


def _measure_model_via_api(model_name: str, prompt: str = "Write a hello world function in Python.") -> Dict[str, Any]:
    """Measure a model using the Ollama HTTP API for accurate timing."""
    import urllib.request
    try:
        payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())

        eval_count = data.get("eval_count", 0)
        eval_dur_ns = data.get("eval_duration", 0)
        prompt_dur_ns = data.get("prompt_eval_duration", 0)

        tps = round(eval_count / (eval_dur_ns / 1e9), 1) if eval_dur_ns else 0
        ttft = round(prompt_dur_ns / 1e9, 2) if prompt_dur_ns else 0

        return {"success": True, "ttft": ttft, "tps": tps, "model": model_name}
    except Exception:
        return {"success": False, "ttft": None, "tps": None, "model": model_name}


def _safe_tps(result: Dict[str, Any]) -> str:
    """Return tps value or '—' for display."""
    v = result.get("tps")
    return str(v) if v is not None else "—"


def _safe_ttft(result: Dict[str, Any]) -> str:
    v = result.get("ttft")
    return str(v) if v is not None else "—"


@app.post("/measure-baseline", response_class=HTMLResponse)
async def measure_baseline(
    request: Request,
    fast_model: str = Form(...),
    smart_model: str = Form(...),
):
    """Run baseline measurements on both models via Ollama API."""
    state = get_session_state(request)

    fast_result = _measure_model_via_api(fast_model)
    smart_result = _measure_model_via_api(smart_model) if smart_model != fast_model else fast_result

    baseline = {
        "fast_model": fast_model,
        "smart_model": smart_model,
        "fast_ttft": _safe_ttft(fast_result),
        "fast_tps": _safe_tps(fast_result),
        "smart_ttft": _safe_ttft(smart_result),
        "smart_tps": _safe_tps(smart_result),
        "timestamp": str(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }

    state["wizard_baseline"] = baseline
    state["wizard_measurements"] = [{"label": "Baseline", **baseline}]

    return templates.TemplateResponse(
        request=request,
        name="onboard_wizard.html",
        context={
            "request": request,
            "step": "results",
            "baseline": baseline,
        },
    )


@app.get("/tune", response_class=HTMLResponse)
async def tune_dashboard(request: Request):
    """Tuning dashboard with optimization suggestions."""
    state = get_session_state(request)
    baseline = state.get("wizard_baseline")
    measurements = state.get("wizard_measurements", [])
    
    if not baseline:
        return render_error(request, "Please complete onboarding first", status_code=400)
    
    plat = state.get("platform_info") or detect_platform()
    
    return templates.TemplateResponse(
        request=request,
        name="tune_dashboard.html",
        context={
            "request": request,
            "baseline": baseline,
            "measurements": measurements[1:] if len(measurements) > 1 else [],
            "is_jetson": plat.get("is_jetson", False),
        },
    )


@app.post("/pull-model")
async def pull_model(request: Request, model: str = Form(...)):
    """Pull/download a model via Ollama API. Returns JSON progress."""
    import urllib.request

    # Validate model name: only allow alphanumeric, colons, dots, hyphens, underscores, slashes
    if not re.match(r'^[a-zA-Z0-9._:/-]+$', model):
        return {"status": "error", "message": "Invalid model name"}

    try:
        payload = json.dumps({"name": model, "stream": False}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return {"status": "ok", "message": f"Successfully pulled {model}", "detail": data.get("status", "")}
    except Exception as e:
        return {"status": "error", "message": f"Failed to pull {model}: {e}"}


@app.post("/measure-again", response_class=HTMLResponse)
async def measure_again(request: Request, what_changed: str = Form("")):
    """Run another measurement after an optimization."""
    state = get_session_state(request)
    baseline = state.get("wizard_baseline")
    
    if not baseline:
        return render_error(request, "No baseline found", status_code=400)
    
    # Re-measure using Ollama API
    fast_result = _measure_model_via_api(baseline["fast_model"], "Explain what quantum computing is in one sentence.")
    smart_result = (
        _measure_model_via_api(baseline["smart_model"], "Explain what quantum computing is in one sentence.")
        if baseline["smart_model"] != baseline["fast_model"]
        else fast_result
    )

    new_measurement = {
        "label": what_changed or f"Measurement #{len(state.get('wizard_measurements', []))}",
        "fast_model": baseline["fast_model"],
        "smart_model": baseline["smart_model"],
        "fast_ttft": _safe_ttft(fast_result),
        "fast_tps": _safe_tps(fast_result),
        "smart_ttft": _safe_ttft(smart_result),
        "smart_tps": _safe_tps(smart_result),
        "timestamp": str(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }
    
    measurements = state.get("wizard_measurements", [baseline])
    measurements.append(new_measurement)
    state["wizard_measurements"] = measurements
    
    return templates.TemplateResponse(
        request=request,
        name="tune_dashboard.html",
        context={
            "request": request,
            "baseline": baseline,
            "measurements": measurements[1:],
        },
    )


@app.get("/tune/guide/{guide_id}", response_class=HTMLResponse)
async def tune_guide(request: Request, guide_id: str):
    """Step-by-step guide for a specific optimization."""
    state = get_session_state(request)
    baseline = state.get("wizard_baseline", {})
    smart_model = baseline.get("smart_model", "your smart model")
    fast_model = baseline.get("fast_model", "your fast model")

    # Derive a q4 pull command for the current smart model
    # e.g. qwen2.5:32b-instruct-q5_K_M  →  qwen2.5:32b-instruct-q4_K_M
    smart_q4 = smart_model.replace("q5_K_M", "q4_K_M").replace("q5_0", "q4_0")
    if smart_q4 == smart_model:
        # No recognisable quantization tag — append a generic suggestion
        base = smart_model.split(":")[0]
        tag = smart_model.split(":")[1] if ":" in smart_model else ""
        smart_q4 = f"{base}:{tag}-q4_K_M" if tag else f"{base}:latest-q4_K_M"

    GUIDES: Dict[str, Any] = {
        "clocks": {
            "icon": "📌",
            "title": "Clock Locking",
            "subtitle": "Stop the Jetson from slowing its GPU down between requests.",
            "applies_to": "Both models",
            "already_done_note": (
                "If you already ran 'sudo nvpmodel -m 0 && sudo jetson_clocks' earlier in this session, "
                "your clocks are already locked. You'll need to re-run these after every reboot."
            ),
            "warning": None,
            "steps": [
                {
                    "title": "Set power mode to maximum (MAXN)",
                    "description": (
                        "This tells the Jetson to use all its cores at full power. "
                        "Without this, the GPU holds back to save energy."
                    ),
                    "commands": ["sudo nvpmodel -m 0"],
                    "note": "You'll be asked for your password. Type it and press Enter.",
                },
                {
                    "title": "Lock clocks to their fastest stable speed",
                    "description": (
                        "By default the GPU clock 'floats' — it ramps up when busy and drops when idle. "
                        "Locking it removes that startup delay before each response."
                    ),
                    "commands": ["sudo jetson_clocks"],
                    "note": "Wait 5–10 seconds after this command before running a benchmark.",
                },
                {
                    "title": "Verify clocks are locked",
                    "description": "This just shows you the current clock state — nothing changes.",
                    "commands": ["sudo jetson_clocks --show"],
                    "note": "Look for lines showing CPU/GPU clocks at their maximum values.",
                },
                {
                    "title": "Make it permanent on reboot (optional)",
                    "description": (
                        "These settings reset after a reboot. To re-apply automatically, "
                        "add both commands to /etc/rc.local or a systemd service."
                    ),
                    "commands": [],
                    "note": "Skip this for now — come back to it once you're happy with performance.",
                },
            ],
        },
        "quantize": {
            "icon": "💾",
            "title": "Switch Smart Model to a Lighter Quantization",
            "subtitle": f"Get more speed from {smart_model} with a smaller bit-width version.",
            "applies_to": f"Smart model only ({smart_model})",
            "already_done_note": None,
            "warning": (
                "This downloads a new copy of the model (~15–20 GB). "
                "Make sure you have space on disk before starting. "
                "Your existing model stays installed — you can always switch back."
            ),
            "steps": [
                {
                    "title": "Understand the tradeoff",
                    "description": (
                        "Your current model uses q5_K_M — 5-bit weights. "
                        "Switching to q4_K_M uses 4-bit weights: roughly 20% smaller, 20–40% faster, "
                        "with a very small reduction in answer quality. "
                        "For most coding tasks, q4_K_M is indistinguishable from q5_K_M."
                    ),
                    "commands": [],
                    "note": None,
                },
                {
                    "title": "Pull the q4 version",
                    "description": "This downloads the lighter version alongside your existing model.",
                    "commands": [f"ollama pull {smart_q4}"],
                    "note": "This will take several minutes depending on your internet speed.",
                },
                {
                    "title": "Test it with a quick prompt",
                    "description": "Run a quick question to make sure it loaded correctly.",
                    "commands": [f'ollama run {smart_q4} "Write a hello world function in Python"'],
                    "note": "If you see a response, it's working.",
                },
                {
                    "title": "Go back through setup to select the new model",
                    "description": (
                        "Return to the model selection step and pick the new q4 version "
                        "as your Smart Model, then measure again."
                    ),
                    "commands": [],
                    "note": "Or just click 'Measure now' below — it will re-run with whichever model is currently loaded.",
                },
            ],
        },
        "parallel": {
            "icon": "⚡",
            "title": "Allow Parallel Requests",
            "subtitle": "Let Ollama handle two requests at the same time instead of one.",
            "applies_to": "Both models (Ollama serving layer)",
            "already_done_note": None,
            "warning": (
                "Running two models in parallel uses more memory. "
                "On 64 GB unified memory you have plenty of headroom, "
                "but if you run into sluggishness, set this back to 1."
            ),
            "steps": [
                {
                    "title": "Understand when this helps",
                    "description": (
                        "Right now, if your fast model is busy answering a question "
                        "and you ask the smart model something, it has to wait. "
                        "Setting OLLAMA_NUM_PARALLEL=2 lets both run at the same time. "
                        "This matters most if you're building an app that talks to both models."
                    ),
                    "commands": [],
                    "note": None,
                },
                {
                    "title": "Set the environment variable",
                    "description": (
                        "Stop Ollama, set the variable, then restart it. "
                        "The variable tells Ollama how many requests it can handle simultaneously."
                    ),
                    "commands": [
                        "sudo systemctl stop ollama",
                        "sudo bash -c 'echo OLLAMA_NUM_PARALLEL=2 >> /etc/default/ollama'",
                        "sudo systemctl start ollama",
                    ],
                    "note": "If Ollama isn't running as a systemd service, add 'export OLLAMA_NUM_PARALLEL=2' to your ~/.bashrc instead.",
                },
                {
                    "title": "Verify it's running",
                    "description": "Check that Ollama started back up correctly.",
                    "commands": ["ollama list"],
                    "note": "If you see your models listed, Ollama is running fine.",
                },
                {
                    "title": "Load both models into memory",
                    "description": (
                        "Send a quick request to each model so they're both warm in memory "
                        "before you benchmark."
                    ),
                    "commands": [
                        f'ollama run {fast_model} "hi"',
                        f'ollama run {smart_model} "hi"',
                    ],
                    "note": "The first response will be slow as the model loads. Subsequent ones will be fast.",
                },
            ],
        },
        "context": {
            "icon": "🧠",
            "title": "Smart Context Management",
            "subtitle": "The single biggest speedup: feed the model less, get answers faster.",
            "applies_to": "Both models — especially the deep model",
            "already_done_note": None,
            "warning": None,
            "steps": [
                {
                    "title": "Understand why this matters more than hardware tuning",
                    "description": (
                        "Your models run at a fixed speed — that's set by memory bandwidth, "
                        "not clocks or settings. But every request has two phases: "
                        "(1) reading your prompt (prefill), and (2) generating the answer. "
                        "A 1K-token prompt takes ~0.1s to read. A 16K-token prompt takes ~2s. "
                        "The generation speed (tok/s) stays the same, but you wait longer before "
                        "the first token appears. By turn 20 of a conversation, your prompt can "
                        "be 10K+ tokens of history — and the model re-reads ALL of it every turn."
                    ),
                    "commands": [],
                    "note": "This is why chatbots feel fast at first and slow down over time.",
                },
                {
                    "title": "Set a practical context length limit",
                    "description": (
                        "Ollama defaults to a large context window. If you don't need 16K tokens "
                        "of history, setting a smaller limit forces older messages out and keeps "
                        "prefill time low. For most tasks, 4096–8192 tokens is plenty."
                    ),
                    "commands": [
                        "sudo mkdir -p /etc/systemd/system/ollama.service.d",
                        "sudo bash -c 'cat > /etc/systemd/system/ollama.service.d/context.conf << EOF\n"
                        "[Service]\n"
                        'Environment="OLLAMA_CONTEXT_LENGTH=8192"\n'
                        "EOF'",
                        "sudo systemctl daemon-reload && sudo systemctl restart ollama",
                    ],
                    "note": (
                        "Start with 8192. If you find the model 'forgets' recent context, "
                        "bump it up to 16384. Lower values = faster prefill."
                    ),
                },
                {
                    "title": "Load one model at a time (relay pattern)",
                    "description": (
                        "By default Ollama keeps every model you use loaded in memory. "
                        "Two models = double the RAM pressure. Setting MAX_LOADED_MODELS=1 "
                        "means Ollama unloads model A before loading model B. You lose a few "
                        "seconds on the swap, but each model runs with more memory headroom "
                        "and the system stays out of swap."
                    ),
                    "commands": [
                        "sudo mkdir -p /etc/systemd/system/ollama.service.d",
                        "sudo bash -c 'cat > /etc/systemd/system/ollama.service.d/memory.conf << EOF\n"
                        "[Service]\n"
                        'Environment="OLLAMA_MAX_LOADED_MODELS=1"\n'
                        'Environment="OLLAMA_KEEP_ALIVE=5m"\n'
                        "EOF'",
                        "sudo systemctl daemon-reload && sudo systemctl restart ollama",
                    ],
                    "note": (
                        "This is how multi-model workflows (orchestrator → coder → reviewer) "
                        "run fast on limited hardware. Each model gets the full memory budget."
                    ),
                },
                {
                    "title": "Use semantic memory instead of full history",
                    "description": (
                        "The most powerful optimization: instead of sending the entire conversation "
                        "history to the model every turn, store past messages in a vector database "
                        "(like LanceDB) and retrieve only the 3–5 most relevant memories per request. "
                        "This keeps your prompt at ~1K tokens no matter how long the session runs. "
                        "The model gets the RIGHT context, not ALL context — and responds faster."
                    ),
                    "commands": [],
                    "note": (
                        "This is the pattern used by AgentSoul and similar memory frameworks. "
                        "Open WebUI also has a built-in memory feature that does this. "
                        "If you selected 'Agent Framework' mode, your framework likely supports this."
                    ),
                },
                {
                    "title": "The numbers: what context management actually saves",
                    "description": (
                        "Here's a real example on this hardware:\n\n"
                        "• Turn 1 (short prompt): TTFT ~0.1s, feels instant\n"
                        "• Turn 20 (full history): TTFT ~2-3s, feels sluggish\n"
                        "• Turn 20 (with semantic memory, 5 recalls): TTFT ~0.2s, feels instant\n\n"
                        "The tok/s doesn't change, but the wall-clock time per interaction "
                        "drops 5-10x because the model isn't re-reading old conversations."
                    ),
                    "commands": [],
                    "note": (
                        "This matters more than any hardware tuning. Clock locking might give you "
                        "10-20% more tok/s. Context management gives you 5-10x faster interactions."
                    ),
                },
            ],
        },
    }

    guide = GUIDES.get(guide_id)
    if not guide:
        return render_error(request, f"Guide '{guide_id}' not found", status_code=404)

    return templates.TemplateResponse(
        request=request,
        name="tune_guide.html",
        context={"request": request, "guide": guide},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
