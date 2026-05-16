# LLM OptiCoach

**A beginner-first setup wizard for running local AI on NVIDIA hardware.**

OptiCoach auto-detects your device, recommends models that fit your memory, and walks you through tuning — no jargon, no guesswork.

![Step 1: Hardware check](https://img.shields.io/badge/Step_1-Hardware_Check-emerald) ![Step 2: Model selection](https://img.shields.io/badge/Step_2-Model_Selection-blue) ![Step 3: Benchmark](https://img.shields.io/badge/Step_3-Benchmark-yellow)

## Supported Hardware

| Device | Memory | Status |
|--------|--------|--------|
| **NVIDIA DGX Spark** | 128 GB | ✅ Supported |
| **Jetson AGX Orin 64 GB** | 64 GB | ✅ Supported |
| **Jetson AGX Orin 32 GB** | 32 GB | ✅ Supported |
| **Jetson Orin NX 16 GB** | 16 GB | ✅ Supported |
| **Jetson Orin Nano 8 GB** | 8 GB | ✅ Supported |
| **Jetson Orin Nano 4 GB** | 4 GB | ✅ Supported |
| GB10 OEM systems (Acer, ASUS, Dell, etc.) | 128 GB | ✅ Supported |
| Jetson Thor | TBD | 🔜 Ready when it ships |

## Install (one command)

```bash
curl -sSL https://raw.githubusercontent.com/Caltongroup/llm-opticoach/main/install.sh | bash
```

This will:
1. Check that Python 3.10+ is installed
2. Offer to install [Ollama](https://ollama.com) if missing
3. Clone the repo to `~/llm_opti_coach`
4. Create a virtual environment and install dependencies
5. Set up a systemd service that starts on boot
6. Print the URL to open in your browser

### Manual install

```bash
git clone https://github.com/Caltongroup/llm-opticoach.git ~/llm_opti_coach
cd ~/llm_opti_coach
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Then open **http://your-device-ip:8080** in a browser.

## What it does

### Step 1: Hardware Check
- Auto-detects your NVIDIA board (Jetson, DGX Spark, GB10 OEM)
- Shows real-time RAM, CPU, GPU, and thermal metrics
- Checks power mode (MAXN) on Jetson devices

### Step 2: Model Selection
- Recommends an **orchestrator** (fast agent for tool calling) and a **heavy lifter** (large model for deep coding)
- Picks the best models from what you already have installed
- Suggests better downloads if you're missing ideal models
- Shows memory fit estimate so you know both models will run without thrashing

### Step 3: Benchmark
- Measures baseline performance (tokens/sec, time to first token)
- Tuning dashboard with step-by-step optimization guides
- Measure-compare workflow: change something, measure again, see the difference

## Model Recommendations by Tier

| Memory Budget | Orchestrator | Heavy Lifter |
|---------------|-------------|-------------|
| 100+ GB (DGX Spark) | hermes3:8b (5 GB) | qwen3:72b (42 GB) |
| 48–99 GB (AGX Orin 64) | hermes3:8b (5 GB) | qwen2.5-coder:32b q4/q5 (20 GB) |
| 24–47 GB (AGX Orin 32) | hermes3:8b (5 GB) | qwen2.5-coder:14b (9 GB) |
| 12–23 GB (Orin NX 16) | nemotron-3-nano:4b (3 GB) | qwen2.5-coder:7b (5 GB) |
| 4–11 GB (Orin Nano 8) | nemotron-3-nano:4b (3 GB) | qwen2.5-coder:3b (2 GB) |
| <4 GB (Orin Nano 4) | qwen2.5:1.5b (1 GB) | qwen2.5-coder:1.5b (1 GB) |

## Useful commands

```bash
# Check status
systemctl --user status opticoach

# View logs
journalctl --user -u opticoach -f

# Restart after changes
systemctl --user restart opticoach

# Update to latest
cd ~/llm_opti_coach && git pull && systemctl --user restart opticoach
```

## Tech Stack

- **Backend**: Python / FastAPI / Uvicorn
- **Frontend**: Jinja2 templates + Tailwind CSS
- **Models**: Ollama (local model serving)
- **Metrics**: tegrastats (Jetson) / nvidia-smi (DGX) / /proc (Linux)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com)
- NVIDIA hardware with unified memory (Jetson or DGX Spark)
- A browser (access from any device on your network)

## License

MIT
