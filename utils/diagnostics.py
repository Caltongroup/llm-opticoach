import os
import platform
import re
import shutil
import subprocess
import time
import json
from urllib.error import URLError
from urllib.request import Request, urlopen
from datetime import datetime
from typing import Any, Dict, Optional


def run_cmd(cmd: str, timeout: int = 10) -> Dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "duration_s": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "success": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
            "error": f"Timeout after {timeout}s",
            "duration_s": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "command": cmd,
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "duration_s": round(time.time() - started, 3),
        }


def missing_tool_result(tool_name: str, detail: str = "") -> Dict[str, Any]:
    suffix = f" ({detail})" if detail else ""
    return {
        "command": tool_name,
        "success": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": f"Tool not available: {tool_name}{suffix}",
        "duration_s": 0.0,
    }


def safe_read_text(path: str, max_chars: int = 200000) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max_chars)
    except Exception:
        return None


def collect_routing_policy_hints(process_lines: list[str]) -> Dict[str, Any]:
    policy_sources = []
    policy_entries = []
    primary_candidates = []
    fallback_candidates = []

    def add_unique(items: list, value: str):
        if value and value not in items:
            items.append(value)

    known_files = [
        os.path.join(os.getcwd(), "config", "runtime_modes.json"),
        os.path.join(os.getcwd(), "config", "model_profiles.json"),
        os.path.join(os.getcwd(), "hermes", "config", "runtime_modes.json"),
        os.path.join(os.getcwd(), "hermes", "config", "model_profiles.json"),
        "/home/darrell/LLM/config/runtime_modes.json",
        "/home/darrell/LLM/config/model_profiles.json",
        "/home/darrell/LLM/hermes/config/runtime_modes.json",
        "/home/darrell/LLM/hermes/config/model_profiles.json",
    ]

    for path in known_files:
        if not os.path.exists(path):
            continue
        text = safe_read_text(path)
        if not text:
            continue
        add_unique(policy_sources, path)

        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

        if parsed is not None:
            lowered = json.dumps(parsed).lower()
            if any(token in lowered for token in ["fallback", "handoff", "router", "route_chain", "model_chain"]):
                add_unique(policy_entries, f"{path}: contains routing/fallback keys")

            def walk_json(value: Any):
                if isinstance(value, dict):
                    for key, val in value.items():
                        key_l = str(key).lower()
                        if "primary" in key_l and "model" in key_l and isinstance(val, str):
                            add_unique(primary_candidates, val)
                        if "model" in key_l and isinstance(val, str) and key_l in {"model", "model_name", "served_model"}:
                            add_unique(primary_candidates, val)
                        if any(word in key_l for word in ["fallback", "handoff", "backup"]) and isinstance(val, str):
                            add_unique(fallback_candidates, val)
                        walk_json(val)
                elif isinstance(value, list):
                    for item in value:
                        walk_json(item)

            walk_json(parsed)
        else:
            lower_text = text.lower()
            if any(token in lower_text for token in ["fallback", "handoff", "router", "model"]):
                add_unique(policy_entries, f"{path}: contains routing keywords (non-JSON parse)")

    env_like_pattern = re.compile(r"\b([A-Z_]*(?:MODEL|FALLBACK|ROUTER|HANDOFF)[A-Z_]*)=([^\s]+)")
    url_pattern = re.compile(r"https?://[^\s]+")

    route_endpoints = []
    for line in process_lines:
        lower = line.lower()
        for match in env_like_pattern.findall(line):
            name, value = match
            if "FALLBACK" in name or "HANDOFF" in name:
                add_unique(fallback_candidates, value)
                add_unique(policy_entries, f"process env hint: {name}={value}")
            elif "MODEL" in name:
                add_unique(primary_candidates, value)
                add_unique(policy_entries, f"process env hint: {name}={value}")

        for endpoint in url_pattern.findall(line):
            if any(token in lower for token in ["router", "route", "proxy", "openai", "v1"]):
                add_unique(route_endpoints, endpoint)

    confidence = "low"
    if policy_sources and (primary_candidates or fallback_candidates):
        confidence = "high"
    elif policy_sources or policy_entries:
        confidence = "medium"

    return {
        "policy_sources": policy_sources,
        "policy_entries": policy_entries,
        "primary_candidates": primary_candidates,
        "fallback_candidates": fallback_candidates,
        "route_endpoints": route_endpoints,
        "confidence": confidence,
    }


def query_ollama_endpoint(path: str, timeout: int = 4) -> Dict[str, Any]:
    url = f"http://127.0.0.1:11434{path}"
    started = time.time()
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        parsed = json.loads(raw)
        return {
            "command": f"GET {path}",
            "success": True,
            "returncode": 0,
            "stdout": raw[:2000],
            "stderr": "",
            "duration_s": round(time.time() - started, 3),
            "json": parsed,
        }
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "command": f"GET {path}",
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "duration_s": round(time.time() - started, 3),
        }


def is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or shutil.which("tegrastats") is not None


def get_platform_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "is_jetson": is_jetson(),
        "timestamp": datetime.now().isoformat(),
    }
    if info["is_jetson"]:
        if os.path.exists("/etc/nv_tegra_release"):
            info["jetpack"] = run_cmd("cat /etc/nv_tegra_release", timeout=5)
        else:
            info["jetpack"] = missing_tool_result("/etc/nv_tegra_release", "file missing")
    return info


def get_power_and_clocks() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if is_jetson():
        if shutil.which("nvpmodel"):
            data["nvpmodel"] = run_cmd("nvpmodel -q", timeout=8)
        else:
            data["nvpmodel"] = missing_tool_result("nvpmodel")

        if shutil.which("jetson_clocks"):
            data["jetson_clocks"] = run_cmd("jetson_clocks --show", timeout=8)
        else:
            data["jetson_clocks"] = missing_tool_result("jetson_clocks")
    else:
        if shutil.which("nvidia-smi"):
            data["nvidia_smi"] = run_cmd(
                "nvidia-smi --query-gpu=name,utilization.gpu,power.draw --format=csv",
                timeout=8,
            )
        else:
            data["nvidia_smi"] = missing_tool_result("nvidia-smi")
    return data


def get_memory_gpu() -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "free": run_cmd("free -h", timeout=5),
        "uptime": run_cmd("uptime", timeout=5),
    }
    if is_jetson():
        if shutil.which("tegrastats"):
            data["tegrastats"] = run_cmd("timeout 3s tegrastats --interval 500 | tail -3", timeout=6)
        else:
            data["tegrastats"] = missing_tool_result("tegrastats")
    else:
        if shutil.which("nvidia-smi"):
            data["nvidia_smi"] = run_cmd(
                "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv",
                timeout=8,
            )
        else:
            data["nvidia_smi"] = missing_tool_result("nvidia-smi")
    return data


def detect_serving() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    active_models = []
    orchestrators = []
    fallback_hints = []
    runtime_notes = []

    def add_unique(items: list, value: str):
        if value and value not in items:
            items.append(value)

    def extract_flag_value(parts: list, flag_name: str) -> Optional[str]:
        for idx, part in enumerate(parts):
            if part == flag_name and idx + 1 < len(parts):
                return parts[idx + 1]
            if part.startswith(flag_name + "="):
                return part.split("=", 1)[1]
        return None

    if shutil.which("pgrep"):
        processes = run_cmd("pgrep -af 'vllm|tensorrt|ollama|llama.cpp|trtllm'", timeout=5)
        raw_lines = [line.strip() for line in (processes.get("stdout", "") or "").splitlines() if line.strip()]
        filtered_lines = [
            line
            for line in raw_lines
            if "pgrep -af" not in line and "grep -E" not in line and "copilot-chat" not in line
        ]
        data["processes"] = "\n".join(filtered_lines)
        data["process_probe"] = processes

        for line in filtered_lines:
            lower = line.lower()
            parts = line.split()

            if "hermes" in lower:
                add_unique(orchestrators, "Hermes")
            if "openclaw" in lower:
                add_unique(orchestrators, "OpenClaw")

            model_flag = extract_flag_value(parts, "--model")
            served_flag = extract_flag_value(parts, "--served-model-name")
            ollama_run = None
            if "ollama run" in lower:
                try:
                    run_index = parts.index("run")
                    if run_index + 1 < len(parts):
                        ollama_run = parts[run_index + 1]
                except ValueError:
                    ollama_run = None

            for candidate in (model_flag, served_flag, ollama_run):
                if candidate:
                    add_unique(active_models, candidate)

            if any(key in lower for key in ["fallback", "handoff", "router", "route", "escalat"]):
                add_unique(fallback_hints, line)

        routing_info = collect_routing_policy_hints(filtered_lines)
        data["routing"] = routing_info
        for model_name in routing_info.get("primary_candidates", []):
            add_unique(active_models, model_name)
        for hint in routing_info.get("policy_entries", []):
            add_unique(fallback_hints, hint)
    else:
        data["processes"] = ""
        data["process_probe"] = missing_tool_result("pgrep")
        data["routing"] = {
            "policy_sources": [],
            "policy_entries": [],
            "primary_candidates": [],
            "fallback_candidates": [],
            "route_endpoints": [],
            "confidence": "low",
        }

    if shutil.which("docker"):
        docker_result = run_cmd("docker ps --format '{{.Names}}'", timeout=5)
        data["docker"] = docker_result
        for name in (docker_result.get("stdout", "") or "").splitlines():
            lower_name = name.lower()
            if "hermes" in lower_name:
                add_unique(orchestrators, "Hermes")
            if "openclaw" in lower_name:
                add_unique(orchestrators, "OpenClaw")
    else:
        data["docker"] = missing_tool_result("docker")

    if shutil.which("ollama"):
        ollama_ps = run_cmd("ollama ps", timeout=8)
        data["ollama_ps"] = ollama_ps
        if ollama_ps.get("success"):
            lines = (ollama_ps.get("stdout", "") or "").splitlines()
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    add_unique(active_models, parts[0])

        ollama_api_ps = query_ollama_endpoint("/api/ps")
        data["ollama_api_ps"] = ollama_api_ps
        if ollama_api_ps.get("success"):
            for item in ollama_api_ps.get("json", {}).get("models", []):
                model_name = item.get("name") or item.get("model")
                if model_name:
                    add_unique(active_models, model_name)

        ollama_api_tags = query_ollama_endpoint("/api/tags")
        data["ollama_api_tags"] = ollama_api_tags
        installed = []
        if ollama_api_tags.get("success"):
            for item in ollama_api_tags.get("json", {}).get("models", []):
                model_name = item.get("name")
                if model_name:
                    installed.append(model_name)
        data["installed_models"] = installed

        if "ollama" in (data.get("processes", "").lower()) and not active_models:
            runtime_notes.append(
                "Ollama daemon is running, but no model is currently loaded. This is normal until a request is served."
            )
    else:
        data["ollama_ps"] = missing_tool_result("ollama")
        data["ollama_api_ps"] = missing_tool_result("ollama/api")
        data["ollama_api_tags"] = missing_tool_result("ollama/api")
        data["installed_models"] = []

    data["active_models"] = active_models
    data["orchestrators_detected"] = orchestrators
    data["fallback_hints"] = fallback_hints

    routing = data.get("routing", {})
    fallback_chain = routing.get("fallback_candidates", [])
    if orchestrators and not fallback_chain and not fallback_hints:
        data["routing_risk"] = "high"
    elif fallback_chain or fallback_hints:
        data["routing_risk"] = "low"
    else:
        data["routing_risk"] = "low"

    data["primary_model_candidate"] = (routing.get("primary_candidates") or active_models or [None])[0]
    data["fallback_chain"] = fallback_chain
    data["runtime_notes"] = runtime_notes
    return data


def benchmark_snapshot() -> Dict[str, Any]:
    """Collect lightweight system signals around an inference benchmark run."""
    snapshot: Dict[str, Any] = {
        "captured_at": datetime.now().isoformat(),
        "memory": run_cmd("free -h", timeout=5),
    }
    if is_jetson() and shutil.which("tegrastats"):
        snapshot["thermal"] = run_cmd("timeout 4s tegrastats --interval 500 | tail -2", timeout=6)
    elif shutil.which("nvidia-smi"):
        snapshot["thermal"] = run_cmd(
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv",
            timeout=6,
        )
    else:
        snapshot["thermal"] = missing_tool_result("tegrastats/nvidia-smi")
    return snapshot


def full_diagnostic(model_path: Optional[str] = None) -> Dict[str, Any]:
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": get_platform_info(),
        "power_clocks": get_power_and_clocks(),
        "memory_gpu": get_memory_gpu(),
        "serving": detect_serving(),
        "model_path": model_path,
        "auto_issues": [],
    }

    issues = []
    runtime_notes = report["serving"].get("runtime_notes", [])
    if report["platform"]["is_jetson"]:
        nvpmodel_data = report["power_clocks"].get("nvpmodel", {})
        nvp_text = f"{nvpmodel_data.get('stdout', '')} {nvpmodel_data.get('stderr', '')}".upper()
        if not nvpmodel_data.get("success"):
            issues.append(
                {
                    "severity": "high",
                    "area": "Power Mode",
                    "issue": "Unable to read current power mode reliably",
                    "fix": "Run nvpmodel -q directly and verify permissions/environment.",
                }
            )
        elif "MAXN" not in nvp_text:
            issues.append(
                {
                    "severity": "high",
                    "area": "Power Mode",
                    "issue": "Not running in MAXN performance mode",
                    "fix": "sudo nvpmodel -m 0 && sudo jetson_clocks",
                }
            )

        jetson_clocks_data = report["power_clocks"].get("jetson_clocks", {})
        if not jetson_clocks_data.get("success"):
            jc_text = (
                f"{jetson_clocks_data.get('stdout', '')} "
                f"{jetson_clocks_data.get('stderr', '')} "
                f"{jetson_clocks_data.get('error', '')}"
            ).lower()
            if "root user" in jc_text or "permission denied" in jc_text or "sudo" in jc_text:
                runtime_notes.append(
                    "Clock verification needs root. Optional check: sudo jetson_clocks --show"
                )
            else:
                issues.append(
                    {
                        "severity": "medium",
                        "area": "Clocks",
                        "issue": "Unable to verify jetson_clocks status",
                        "fix": "Check jetson_clocks availability and run 'jetson_clocks --show'.",
                    }
                )

    if not report["serving"].get("processes", "").strip():
        issues.append(
            {
                "severity": "medium",
                "area": "Serving",
                "issue": "No active supported serving process detected",
                "fix": "Start your serving runtime (vLLM/TensorRT-LLM/Ollama/llama.cpp) before benchmarking.",
            }
        )

    if report["serving"].get("orchestrators_detected") and not report["serving"].get("fallback_hints"):
        issues.append(
            {
                "severity": "low",
                "area": "Routing",
                "issue": "Orchestrator detected but no explicit fallback policy hint found",
                "fix": "Document and verify your fallback model chain so handoffs to larger models are intentional.",
            }
        )

    if report["serving"].get("routing_risk") == "high":
        issues.append(
            {
                "severity": "medium",
                "area": "Routing",
                "issue": "Fallback chain is unclear while orchestrator is active",
                "fix": "Define explicit primary and fallback model chain in runtime configuration and verify with one test request.",
            }
        )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    report["auto_issues"] = sorted(issues, key=lambda issue: severity_rank.get(issue.get("severity", "medium"), 1))
    report["serving"]["runtime_notes"] = runtime_notes
    return report
