"""NVIDIA platform detection and memory-tiered model recommendations.

Works across the NVIDIA edge/desktop AI family:
  Jetson: Orin Nano, Orin NX, AGX Orin, Thor
  Desktop: DGX Spark (GB10), GB10 OEM systems

Auto-detects the board and total memory, then recommends models that
actually fit the hardware.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


# ── Platform detection ──────────────────────────────────────────────

def detect_platform() -> Dict[str, Any]:
    """Return a dict describing the Jetson (or non-Jetson) platform."""
    info: Dict[str, Any] = {
        "board": "Unknown",
        "family": "unknown",
        "ram_total_mb": _get_total_ram_mb(),
        "is_jetson": False,
    }

    # Try /proc/device-tree/model first (reliable on ARM64 NVIDIA boards)
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_str = f.read().strip().rstrip("\x00")
        info["board"] = model_str
        info["is_jetson"] = "jetson" in model_str.lower() or "nvidia" in model_str.lower()
    except FileNotFoundError:
        pass

    # Fallback: /etc/nv_tegra_release (JetPack)
    if not info["is_jetson"]:
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                info["is_jetson"] = True
                info["board"] = info.get("board") or "NVIDIA Jetson (tegra)"
        except FileNotFoundError:
            pass

    # Check for DGX OS (/etc/dgx-release)
    info["is_dgx"] = False
    try:
        with open("/etc/dgx-release", "r") as f:
            info["is_dgx"] = True
            if info["board"] == "Unknown":
                info["board"] = "NVIDIA DGX System"
    except FileNotFoundError:
        pass

    # Classify into family
    board_lower = info["board"].lower()
    ram_mb = info["ram_total_mb"]

    if "dgx spark" in board_lower or (info["is_dgx"] and "spark" in board_lower):
        info["family"] = "dgx_spark"
        info["is_jetson"] = False  # DGX Spark is not a Jetson
        info["display_name"] = f"NVIDIA DGX Spark ({ram_mb // 1024} GB)"
    elif "gb10" in board_lower:
        # OEM GB10 systems (Acer Veriton GN100, ASUS Ascent GX10, Dell Pro Max, etc.)
        info["family"] = "gb10_oem"
        info["is_jetson"] = False
        info["display_name"] = f"NVIDIA GB10 System ({ram_mb // 1024} GB)"
    elif info["is_dgx"]:
        info["family"] = "dgx_other"
        info["is_jetson"] = False
        info["display_name"] = f"NVIDIA DGX ({ram_mb // 1024} GB)"
    elif "agx orin" in board_lower:
        info["family"] = "agx_orin"
        info["display_name"] = f"Jetson AGX Orin ({ram_mb // 1024} GB)"
    elif "orin nx" in board_lower:
        info["family"] = "orin_nx"
        info["display_name"] = f"Jetson Orin NX ({ram_mb // 1024} GB)"
    elif "orin nano" in board_lower:
        info["family"] = "orin_nano"
        info["display_name"] = f"Jetson Orin Nano ({ram_mb // 1024} GB)"
    elif "thor" in board_lower:
        info["family"] = "thor"
        info["display_name"] = f"Jetson Thor ({ram_mb // 1024} GB)"
    elif info["is_jetson"]:
        info["family"] = "jetson_other"
        info["display_name"] = f"Jetson ({ram_mb // 1024} GB)"
    else:
        info["family"] = "non_jetson"
        info["display_name"] = f"Linux system ({ram_mb // 1024} GB RAM)"

    # Compute usable memory for models (leave ~4 GB for OS + KV cache headroom)
    OS_RESERVE_MB = 4096
    info["model_budget_gb"] = round(max(0, ram_mb - OS_RESERVE_MB) / 1024, 1)

    return info


def _get_total_ram_mb() -> int:
    """Return total system RAM in MB from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024  # kB → MB
    except Exception:
        pass
    return 0


# ── Memory-tiered model recommendations ─────────────────────────────
#
# Each tier defines the best orchestrator and heavy-lifter for a given
# memory budget.  The tiers are checked top-down; first one whose
# min_budget_gb <= available budget wins.

_MODEL_TIERS: List[Dict[str, Any]] = [
    # ── 100+ GB budget (DGX Spark 128 GB, GB10 OEM systems) ─────────
    {
        "min_budget_gb": 100,
        "orchestrator": {
            "ideal": "hermes3:8b",
            "ideal_set": {"hermes3:8b", "hermes3:latest"},
            "pull": "ollama pull hermes3:8b",
            "size_gb": 4.9,
            "why": (
                "Nous Research Hermes 3 — purpose-built for tool calling, "
                "agent personality, and structured function calls. "
                "Lightweight enough to leave 120 GB for your heavy model."
            ),
            "priority": [
                "hermes3:8b", "hermes3:latest",
                "qwen3:8b", "qwen3:8b-nothink", "qwen3.5:9b",
                "qwen2.5:7b-instruct", "qwen2.5-coder:7b",
                "nemotron-3-nano:4b",
            ],
        },
        "heavy": {
            "ideal": "qwen3:72b",
            "ideal_set": {
                "qwen3:72b",
                "qwen2.5:72b-instruct-q4_K_M",
                "qwen2.5:72b-instruct-q5_K_M",
                "deepseek-r1:70b",
                "llama3.1:70b-instruct-q4_K_M",
            },
            "pull": "ollama pull qwen3:72b",
            "size_gb": 42.0,
            "why": (
                "72B parameter model with frontier-level reasoning. "
                "Your 128 GB of unified memory can run this comfortably "
                "alongside the orchestrator with room to spare for KV cache."
            ),
            "priority": [
                "qwen3:72b",
                "qwen2.5:72b-instruct-q5_K_M",
                "qwen2.5:72b-instruct-q4_K_M",
                "deepseek-r1:70b",
                "llama3.1:70b-instruct-q4_K_M",
                "qwen2.5-coder:32b-instruct-q5_K_M",
                "qwen2.5-coder:32b-instruct-q4_K_M",
                "qwen2.5:32b-instruct-q5_K_M",
                "qwen2.5:32b-instruct-q4_K_M",
                "qwen3.6:27b", "qwen3.5:27b", "qwen3.5:35b",
                "qwen3:30b",
            ],
        },
    },

    # ── 48–99 GB budget (AGX Orin 64 GB, Thor) ──────────────────────
    {
        "min_budget_gb": 48,
        "orchestrator": {
            "ideal": "hermes3:8b",
            "ideal_set": {"hermes3:8b", "hermes3:latest"},
            "pull": "ollama pull hermes3:8b",
            "size_gb": 4.9,
            "why": (
                "Nous Research Hermes 3 — purpose-built for tool calling, "
                "agent personality, and structured function calls. "
                "The standard for agentic orchestration."
            ),
            "priority": [
                "hermes3:8b", "hermes3:latest",
                "qwen3:8b", "qwen3:8b-nothink", "qwen3.5:9b",
                "qwen2.5:7b-instruct", "qwen2.5-coder:7b",
                "nemotron-3-nano:4b",
            ],
        },
        "heavy": {
            "ideal": "qwen2.5-coder:32b-instruct-q4_K_M",
            "ideal_set": {
                "qwen2.5-coder:32b-instruct-q4_K_M",
                "qwen2.5-coder:32b-instruct-q5_K_M",
                "qwen2.5:32b-instruct-q4_K_M",
                "qwen2.5:32b-instruct-q5_K_M",
            },
            "pull": "ollama pull qwen2.5-coder:32b-instruct-q4_K_M",
            "size_gb": 20.0,
            "why": (
                "32B code-trained model with deep reasoning. "
                "q4/q5 quantization keeps it at ~20 GB so both models "
                "fit comfortably in memory."
            ),
            "priority": [
                "qwen2.5-coder:32b-instruct-q4_K_M",
                "qwen2.5-coder:32b-instruct-q5_K_M",
                "qwen2.5:32b-instruct-q4_K_M",
                "qwen2.5:32b-instruct-q5_K_M",
                "qwen3.6:27b", "qwen3.5:27b", "qwen3.5:35b",
                "qwen3:30b", "qwen2.5:14b-instruct-q6_K",
                "gpt-oss:20b",
            ],
        },
    },

    # ── 24–47 GB budget (AGX Orin 32 GB) ────────────────────────────
    {
        "min_budget_gb": 24,
        "orchestrator": {
            "ideal": "hermes3:8b",
            "pull": "ollama pull hermes3:8b",
            "size_gb": 4.9,
            "why": (
                "Hermes 3 at 8B is fast and built for tool calling. "
                "Leaves most of your memory for the heavy model."
            ),
            "priority": [
                "hermes3:8b", "hermes3:latest",
                "qwen3:8b", "qwen3:8b-nothink", "qwen3.5:9b",
                "qwen2.5:7b-instruct", "qwen2.5-coder:7b",
                "nemotron-3-nano:4b",
            ],
        },
        "heavy": {
            "ideal": "qwen2.5-coder:14b-instruct-q4_K_M",
            "pull": "ollama pull qwen2.5-coder:14b-instruct-q4_K_M",
            "size_gb": 9.0,
            "why": (
                "14B code-trained model — strong reasoning while fitting "
                "in 32 GB alongside the orchestrator. "
                "Good balance of speed and quality."
            ),
            "priority": [
                "qwen2.5-coder:14b-instruct-q4_K_M",
                "qwen2.5:14b-instruct-q6_K",
                "qwen2.5:14b-instruct-q4_K_M",
                "qwen3.6:27b", "qwen3.5:27b",
                "gpt-oss:20b",
                "qwen3:8b", "qwen3.5:9b",
            ],
        },
    },

    # ── 12–23 GB budget (Orin NX 16 GB) ─────────────────────────────
    {
        "min_budget_gb": 12,
        "orchestrator": {
            "ideal": "nemotron-3-nano:4b",
            "pull": "ollama pull nemotron-3-nano:4b",
            "size_gb": 2.8,
            "why": (
                "At 2.8 GB, Nemotron Nano leaves maximum headroom for "
                "your heavy model on a 16 GB device."
            ),
            "priority": [
                "nemotron-3-nano:4b",
                "hermes3:8b", "hermes3:latest",
                "qwen3:8b", "qwen3:8b-nothink",
                "qwen2.5-coder:7b",
            ],
        },
        "heavy": {
            "ideal": "qwen2.5-coder:7b",
            "pull": "ollama pull qwen2.5-coder:7b",
            "size_gb": 4.7,
            "why": (
                "7B code-trained model — the largest that fits "
                "comfortably alongside an orchestrator in 16 GB."
            ),
            "priority": [
                "qwen2.5-coder:7b",
                "qwen3:8b", "qwen3.5:9b",
                "qwen2.5:7b-instruct",
            ],
        },
    },

    # ── 4–11 GB budget (Orin Nano 8 GB, Spark) ──────────────────────
    {
        "min_budget_gb": 4,
        "orchestrator": {
            "ideal": "nemotron-3-nano:4b",
            "pull": "ollama pull nemotron-3-nano:4b",
            "size_gb": 2.8,
            "why": (
                "Tiny and fast. On an 8 GB device this is your "
                "best option for the orchestrator role."
            ),
            "priority": [
                "nemotron-3-nano:4b",
                "qwen2.5-coder:1.5b",
                "qwen2.5:1.5b",
            ],
        },
        "heavy": {
            "ideal": "qwen2.5-coder:3b",
            "pull": "ollama pull qwen2.5-coder:3b",
            "size_gb": 2.0,
            "why": (
                "3B code model — the largest feasible on 8 GB. "
                "You may need to run one model at a time on this device."
            ),
            "priority": [
                "qwen2.5-coder:3b",
                "nemotron-3-nano:4b",
                "qwen2.5:3b",
            ],
        },
    },

    # ── Under 4 GB (Orin Nano 4 GB) ─────────────────────────────────
    {
        "min_budget_gb": 0,
        "orchestrator": {
            "ideal": "qwen2.5:1.5b",
            "pull": "ollama pull qwen2.5:1.5b",
            "size_gb": 1.0,
            "why": (
                "On 4 GB total you can only run one small model at a time. "
                "1.5B is the practical limit."
            ),
            "priority": [
                "qwen2.5:1.5b", "qwen2.5-coder:1.5b",
            ],
        },
        "heavy": {
            "ideal": "qwen2.5-coder:1.5b",
            "pull": "ollama pull qwen2.5-coder:1.5b",
            "size_gb": 1.0,
            "why": (
                "Same size — on 4 GB you'll swap between orchestrator and "
                "coder rather than running both simultaneously."
            ),
            "priority": [
                "qwen2.5-coder:1.5b", "qwen2.5:1.5b",
            ],
        },
    },
]


def get_model_recommendations(
    platform_info: Dict[str, Any],
    installed_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return role-matched model recommendations for the detected platform.

    Returns dict with keys:
        tier          – the matched tier dict
        budget_gb     – usable memory for models
        orchestrator  – {selected, is_ideal, suggest_download}
        heavy         – {selected, is_ideal, suggest_download}
        both_fit      – bool, whether both fit in memory
        combined_gb   – estimated combined size
        warning       – optional warning string
    """
    budget = platform_info.get("model_budget_gb", 0)
    installed_names = {m["name"] for m in installed_models}
    installed_sizes = {m["name"]: m["size_gb"] for m in installed_models}

    # Find the right tier
    tier = _MODEL_TIERS[-1]  # fallback to smallest
    for t in _MODEL_TIERS:
        if budget >= t["min_budget_gb"]:
            tier = t
            break

    result: Dict[str, Any] = {"tier": tier, "budget_gb": budget}

    for role in ("orchestrator", "heavy"):
        role_spec = tier[role]
        # Try to find best installed model from priority list
        selected = None
        for name in role_spec["priority"]:
            if name in installed_names:
                selected = name
                break

        # Check against the ideal set (any top-tier match counts)
        ideal_set = role_spec.get("ideal_set", {role_spec["ideal"]})
        is_ideal = selected in ideal_set
        suggest = None
        if not is_ideal:
            suggest = {
                "name": role_spec["ideal"],
                "size_gb": role_spec["size_gb"],
                "pull_cmd": role_spec["pull"],
                "why": role_spec["why"],
            }

        result[role] = {
            "selected": selected,
            "selected_size_gb": installed_sizes.get(selected, role_spec["size_gb"]) if selected else None,
            "is_ideal": is_ideal,
            "suggest_download": suggest,
        }

    # If neither role got a match, use ideal names as placeholders
    if not result["orchestrator"]["selected"]:
        result["orchestrator"]["selected"] = tier["orchestrator"]["ideal"]
        result["orchestrator"]["selected_size_gb"] = tier["orchestrator"]["size_gb"]
    if not result["heavy"]["selected"]:
        result["heavy"]["selected"] = tier["heavy"]["ideal"]
        result["heavy"]["selected_size_gb"] = tier["heavy"]["size_gb"]

    # Memory fit check
    orch_size = result["orchestrator"]["selected_size_gb"] or 0
    heavy_size = result["heavy"]["selected_size_gb"] or 0
    combined = round(orch_size + heavy_size, 1)
    result["combined_gb"] = combined
    result["both_fit"] = combined <= (budget * 0.85)  # 15% headroom for KV cache

    warning = None
    if not result["both_fit"] and combined > 0:
        if budget < 6:
            warning = (
                f"Your device has ~{budget} GB available for models. "
                "You'll likely need to run one model at a time — "
                "the orchestrator loads, handles the request, unloads, "
                "then the heavy model loads for big tasks."
            )
        else:
            warning = (
                f"Both models together use ~{combined} GB of your "
                f"~{budget} GB model budget. This is tight — watch for "
                "swap usage and slow responses during model switching."
            )
    result["warning"] = warning

    return result


# ── Solo-model recommendations ──────────────────────────────────────

# Priority-ordered list of the single best model for each tier.
# The heavy-lifter model is usually the right solo pick (bigger = smarter).
# For tiny tiers, the orchestrator IS the only option.

_SOLO_TIERS: List[Dict[str, Any]] = [
    {
        "min_budget_gb": 100,
        "ideal": "qwen3:72b",
        "ideal_set": {
            "qwen3:72b",
            "qwen2.5:72b-instruct-q4_K_M",
            "qwen2.5:72b-instruct-q5_K_M",
            "deepseek-r1:70b",
        },
        "pull": "ollama pull qwen3:72b",
        "size_gb": 42.0,
        "why": (
            "72B parameters — frontier-level reasoning and coding. "
            "Your 128 GB of memory handles this easily."
        ),
        "priority": [
            "qwen3:72b",
            "qwen2.5:72b-instruct-q5_K_M",
            "qwen2.5:72b-instruct-q4_K_M",
            "deepseek-r1:70b",
            "llama3.1:70b-instruct-q4_K_M",
            "qwen2.5-coder:32b-instruct-q5_K_M",
            "qwen2.5-coder:32b-instruct-q4_K_M",
            "qwen2.5:32b-instruct-q5_K_M",
            "qwen2.5:32b-instruct-q4_K_M",
            "qwen3:8b", "hermes3:8b",
        ],
    },
    {
        "min_budget_gb": 48,
        "ideal": "qwen2.5-coder:32b-instruct-q4_K_M",
        "ideal_set": {
            "qwen2.5-coder:32b-instruct-q4_K_M",
            "qwen2.5-coder:32b-instruct-q5_K_M",
            "qwen2.5:32b-instruct-q4_K_M",
            "qwen2.5:32b-instruct-q5_K_M",
        },
        "pull": "ollama pull qwen2.5-coder:32b-instruct-q4_K_M",
        "size_gb": 20.0,
        "why": (
            "32B code-trained model — strong reasoning and coding, "
            "fits comfortably in your memory."
        ),
        "priority": [
            "qwen2.5-coder:32b-instruct-q5_K_M",
            "qwen2.5-coder:32b-instruct-q4_K_M",
            "qwen2.5:32b-instruct-q5_K_M",
            "qwen2.5:32b-instruct-q4_K_M",
            "qwen3.6:27b", "qwen3.5:27b", "qwen3.5:35b",
            "qwen3:30b",
            "qwen3:8b", "hermes3:8b",
        ],
    },
    {
        "min_budget_gb": 24,
        "ideal": "qwen2.5-coder:14b-instruct-q4_K_M",
        "ideal_set": {
            "qwen2.5-coder:14b-instruct-q4_K_M",
            "qwen2.5:14b-instruct-q6_K",
            "qwen2.5:14b-instruct-q4_K_M",
        },
        "pull": "ollama pull qwen2.5-coder:14b-instruct-q4_K_M",
        "size_gb": 9.0,
        "why": "14B code-trained model — good balance of speed and quality for 32 GB.",
        "priority": [
            "qwen2.5-coder:14b-instruct-q4_K_M",
            "qwen2.5:14b-instruct-q6_K",
            "qwen2.5:14b-instruct-q4_K_M",
            "qwen3:8b", "hermes3:8b",
            "qwen2.5-coder:7b",
        ],
    },
    {
        "min_budget_gb": 12,
        "ideal": "qwen2.5-coder:7b",
        "pull": "ollama pull qwen2.5-coder:7b",
        "size_gb": 4.7,
        "why": "7B code model — the largest that runs smoothly on 16 GB.",
        "priority": [
            "qwen2.5-coder:7b", "qwen3:8b", "hermes3:8b",
            "qwen2.5:7b-instruct",
            "nemotron-3-nano:4b",
        ],
    },
    {
        "min_budget_gb": 4,
        "ideal": "qwen2.5-coder:3b",
        "pull": "ollama pull qwen2.5-coder:3b",
        "size_gb": 2.0,
        "why": "3B code model — the largest feasible on 8 GB.",
        "priority": [
            "qwen2.5-coder:3b", "nemotron-3-nano:4b", "qwen2.5:3b",
        ],
    },
    {
        "min_budget_gb": 0,
        "ideal": "qwen2.5-coder:1.5b",
        "pull": "ollama pull qwen2.5-coder:1.5b",
        "size_gb": 1.0,
        "why": "1.5B is the practical limit on 4 GB total memory.",
        "priority": [
            "qwen2.5-coder:1.5b", "qwen2.5:1.5b",
        ],
    },
]


def get_solo_recommendation(
    platform_info: Dict[str, Any],
    installed_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the single best model recommendation for solo mode."""
    budget = platform_info.get("model_budget_gb", 0)
    installed_names = {m["name"] for m in installed_models}
    installed_sizes = {m["name"]: m["size_gb"] for m in installed_models}

    tier = _SOLO_TIERS[-1]
    for t in _SOLO_TIERS:
        if budget >= t["min_budget_gb"]:
            tier = t
            break

    selected = None
    for name in tier["priority"]:
        if name in installed_names:
            selected = name
            break

    ideal_set = tier.get("ideal_set", {tier["ideal"]})
    is_ideal = selected in ideal_set
    suggest = None
    if not is_ideal:
        suggest = {
            "name": tier["ideal"],
            "size_gb": tier["size_gb"],
            "pull_cmd": tier["pull"],
            "why": tier["why"],
        }

    selected_size = installed_sizes.get(selected, tier["size_gb"]) if selected else tier["size_gb"]
    if not selected:
        selected = tier["ideal"]

    return {
        "budget_gb": budget,
        "selected": selected,
        "selected_size_gb": selected_size,
        "is_ideal": is_ideal,
        "suggest_download": suggest,
        "fits": selected_size <= (budget * 0.85),
    }
