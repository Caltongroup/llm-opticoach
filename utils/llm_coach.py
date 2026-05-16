import json
import time
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel

class CoachRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    model: str = "gpt-4o-mini"
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.3
    max_tokens: int = 2000

class CoachResponse(BaseModel):
    success: bool
    response_text: str = ""
    error: Optional[str] = None


def estimate_tokens(text: str) -> int:
    # Reasonable rough estimate for quick benchmark scoring when usage isn't returned.
    return max(1, int(len(text) / 4))

def call_coach(req: CoachRequest) -> CoachResponse:
    headers = {"Authorization": f"Bearer {req.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.user_prompt}
        ],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens
    }
    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(f"{req.base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return CoachResponse(success=True, response_text=content)
    except Exception as e:
        return CoachResponse(success=False, error=str(e))


def run_inference_benchmark(req: CoachRequest, benchmark_prompt: Optional[str] = None) -> Dict[str, Any]:
    prompt = benchmark_prompt or (
        "You are benchmarking local coding assistance latency and throughput. "
        "Produce a concise optimization checklist for a Jetson inference stack with 6 bullets."
    )
    headers = {"Authorization": f"Bearer {req.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": req.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a benchmark payload. Return concise deterministic text only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": min(req.max_tokens, 512),
        "stream": True,
    }

    started = time.perf_counter()
    first_token_s: Optional[float] = None
    chunks = []
    usage: Dict[str, Any] = {}

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                f"{req.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_piece = delta.get("content", "")
                        if content_piece:
                            if first_token_s is None:
                                first_token_s = time.perf_counter() - started
                            chunks.append(content_piece)
                    if obj.get("usage"):
                        usage = obj.get("usage", {})

        elapsed_s = time.perf_counter() - started
        response_text = "".join(chunks)
        completion_tokens = usage.get("completion_tokens") or estimate_tokens(response_text)
        tokens_per_second = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0

        return {
            "success": True,
            "model": req.model,
            "base_url": req.base_url,
            "prompt": prompt,
            "ttft_s": round(first_token_s or elapsed_s, 3),
            "elapsed_s": round(elapsed_s, 3),
            "completion_tokens": int(completion_tokens),
            "tokens_per_second": round(tokens_per_second, 3),
            "response_preview": response_text[:240],
            "usage": usage,
            "timestamp": time.time(),
        }
    except Exception as exc:
        return {
            "success": False,
            "model": req.model,
            "base_url": req.base_url,
            "prompt": prompt,
            "error": str(exc),
            "ttft_s": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "completion_tokens": 0,
            "tokens_per_second": 0.0,
            "response_preview": "",
            "usage": {},
            "timestamp": time.time(),
        }


def run_routing_assertion(
    req: CoachRequest,
    expected_primary: Optional[str] = None,
    expected_fallbacks: Optional[list[str]] = None,
    assertion_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = assertion_prompt or "Reply with ROUTING_ASSERT_OK and one short line about active model path."
    expected_fallbacks = expected_fallbacks or []

    headers = {"Authorization": f"Bearer {req.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": "Return concise deterministic output for routing assertion."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{req.base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed_s = round(time.perf_counter() - started, 3)
        response_model = data.get("model")
        choice = (data.get("choices") or [{}])[0]
        response_text = ((choice.get("message") or {}).get("content") or "").strip()

        comparable = [m.lower() for m in [expected_primary, *expected_fallbacks] if m]
        response_model_l = (response_model or "").lower()
        model_in_expected_chain = bool(response_model_l and any(m in response_model_l for m in comparable))

        routing_headers = {
            key: value
            for key, value in resp.headers.items()
            if any(token in key.lower() for token in ["model", "router", "route", "provider", "openai"])
        }

        verdict = "pass" if model_in_expected_chain or not comparable else "warn"
        if not response_model:
            verdict = "warn"

        return {
            "success": True,
            "verdict": verdict,
            "requested_model": req.model,
            "reported_response_model": response_model,
            "expected_primary": expected_primary,
            "expected_fallbacks": expected_fallbacks,
            "model_in_expected_chain": model_in_expected_chain,
            "elapsed_s": elapsed_s,
            "response_preview": response_text[:240],
            "response_usage": data.get("usage", {}),
            "routing_headers": routing_headers,
            "timestamp": time.time(),
        }
    except Exception as exc:
        return {
            "success": False,
            "verdict": "error",
            "requested_model": req.model,
            "reported_response_model": None,
            "expected_primary": expected_primary,
            "expected_fallbacks": expected_fallbacks,
            "model_in_expected_chain": False,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "response_preview": "",
            "response_usage": {},
            "routing_headers": {},
            "error": str(exc),
            "timestamp": time.time(),
        }

DIAGNOSTIC_COACH_SYSTEM = """You are a patient, plain-English coach for local LLM users on Jetson and CUDA machines. Explain things simply. Give prioritized, actionable advice with exact commands and risk level."""

REFEREE_SYSTEM = """You are a conservative referee. Reply with SAFE / PROCEED WITH CAUTION / RISKY, explanation, improved command if needed, and what to monitor."""


def _compact_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": probe.get("success"),
        "returncode": probe.get("returncode"),
        "stdout": (probe.get("stdout") or "")[:500],
        "stderr": (probe.get("stderr") or "")[:300],
        "error": probe.get("error"),
    }


def summarize_diagnostic_report(report: Dict[str, Any]) -> Dict[str, Any]:
    platform = report.get("platform", {})
    power = report.get("power_clocks", {})
    serving = report.get("serving", {})
    memory = report.get("memory_gpu", {})

    nvp = power.get("nvpmodel", {})
    nvp_text = f"{nvp.get('stdout', '')} {nvp.get('stderr', '')}".upper()
    maxn_detected = "MAXN" in nvp_text

    return {
        "timestamp": report.get("timestamp"),
        "platform": {
            "is_jetson": platform.get("is_jetson"),
            "platform": platform.get("platform"),
            "jetpack": _compact_probe(platform.get("jetpack", {})) if isinstance(platform.get("jetpack"), dict) else platform.get("jetpack"),
        },
        "critical_signals": {
            "maxn_detected": maxn_detected,
            "serving_process_detected": bool((serving.get("processes") or "").strip()),
            "docker_probe": _compact_probe(serving.get("docker", {})) if isinstance(serving.get("docker"), dict) else serving.get("docker"),
        },
        "power_probes": {
            "nvpmodel": _compact_probe(nvp) if isinstance(nvp, dict) else nvp,
            "jetson_clocks": _compact_probe(power.get("jetson_clocks", {})) if isinstance(power.get("jetson_clocks"), dict) else power.get("jetson_clocks"),
        },
        "memory_probe": {
            "free": _compact_probe(memory.get("free", {})) if isinstance(memory.get("free"), dict) else memory.get("free"),
            "tegrastats": _compact_probe(memory.get("tegrastats", {})) if isinstance(memory.get("tegrastats"), dict) else memory.get("tegrastats"),
            "nvidia_smi": _compact_probe(memory.get("nvidia_smi", {})) if isinstance(memory.get("nvidia_smi"), dict) else memory.get("nvidia_smi"),
        },
        "auto_issues": report.get("auto_issues", []),
    }

def build_diagnostic_prompt(report: dict, notes: str) -> str:
    summary = summarize_diagnostic_report(report)
    summary_json = json.dumps(summary, indent=2)
    return (
        "You are analyzing a local LLM runtime setup for an NVIDIA developer kit user.\n"
        "Prioritize practical steps they can run immediately.\n\n"
        f"Structured diagnostic summary:\n{summary_json}\n\n"
        f"User notes:\n{notes}\n\n"
        "Return:\n"
        "1) Top 3 bottlenecks (ranked)\n"
        "2) Exact commands for each fix\n"
        "3) Risk level for each fix\n"
        "4) What metric should improve after each fix\n"
    )

def build_referee_prompt(change: str, report: dict, context: str) -> str:
    summary = summarize_diagnostic_report(report)
    summary_json = json.dumps(summary, indent=2)
    return (
        f"Proposed change:\n{change}\n\n"
        f"Current state summary:\n{summary_json}\n\n"
        f"Extra context:\n{context}\n\n"
        "Act as a strict safety referee. Output:\n"
        "- Verdict: SAFE / PROCEED WITH CAUTION / RISKY\n"
        "- Why\n"
        "- Improved command (if needed)\n"
        "- Rollback step\n"
        "- What to monitor after applying\n"
    )
