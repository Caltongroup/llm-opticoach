# LLM OptiCoach - Beginner-First Redesign

## What I Just Built

A complete redesign of the app from a **novice user perspective**. Instead of technical diagnostics first, new users now get a simple 3-step guided wizard.

---

## The Three-Step Onboarding Path

### **Step 1: Hardware Check** (`/onboard`)
![novice sees]
- **What the user sees**: Simple green checkmarks for Memory, GPU, and Power Mode
- **What they need to do**: Check if MAXN mode is enabled (app tells them the exact command)
- **Why it matters**: Can't run AI models fast without maximum power mode
- **Plain English**: "Power Mode: Maximum performance mode isn't enabled yet. Run this command to fix it."

### **Step 2: Pick Your AI Models** (`/onboard/models`)
![novice chooses]
- **What the user sees**: Two cards with friendly icons
  - 🐰 Fast Model (Orchestrator) - "Quick decisions, routing, and light tasks"
  - 🧠 Smart Model (Heavy Lifting) - "Complex coding, detailed analysis, creative work"
- **What they need to know**: WHY they need two models (they work as a team, not the same model doing everything)
- **Defaults are pre-selected**: User can accept them or swap from available models
- **No technical jargon**: No mention of "quantization", "context windows", "token limits"

### **Step 3: Baseline Measurement** (`/measure-baseline`)
![novice understands]
- **What the user sees**: Simple cards showing:
  - "Fast Model Response: 1.2s" and "45 tok/s"
  - "Smart Model Response: 2.1s" and "38 tok/s"
- **What happens next**: User is taken to the Tuning Dashboard
- **What they learn**: "These are your starting numbers. Later, you'll try optimizations and measure again to see what actually helps."

---

## The Tuning Dashboard (`/tune`)

After baseline, user sees:

### **Your Baseline** 
Quick card showing both models' starting speeds. This is their anchor point.

### **Easy Optimizations to Try**
Three simple changes (no terminal commands for main ones):
1. **Clock Locking** (15% faster expected)
   - What it does: Lock GPU frequency to maximum stable
   - Who needs to know: "This often makes things faster"
   
2. **Use Quantized Model** (2x faster, similar quality)
   - What it does: Switch to a lighter version of your model
   - Who needs to know: "Smaller model = less memory used, faster response"
   
3. **Enable Batch Processing** (30% faster on multiple requests)
   - What it does: Group multiple AI requests together
   - Who needs to know: "Great if you're running multiple requests at once"

### **Run Another Measurement**
- Simple form: "What did you change? (optional)"
- Then: "Measure Now" button
- Result: Shows old vs new numbers in simple before/after cards

### **Your Measurement History**
- Timeline of all measurements they've run
- Compare each one to see what actually helped

---

## Key Design Principles (Expert Mode)

### **For the Novice User:**
1. **No jargon in the main flow** - All technical terms hidden behind "Technical Details" accordion
2. **Auto-selection where possible** - Default to the best available models
3. **Plain language outcomes** - "15% faster, same memory, still cool to touch" not "TTFT reduced from 1.2s to 1.02s"
4. **One decision at a time** - Not "here's 50 metrics analyze this"
5. **Action-oriented next steps** - "Run this optimization" not "you might want to consider..."

### **For the Expert Builder (You):**
1. **Session-scoped state** - Each user gets isolated measurements (no data leakage)
2. **Ollama integration** - Auto-detect available models, run local measurements (no API key required)
3. **Delta tracking** - Before/after comparisons automatically calculated
4. **Progressive disclosure** - Advanced details available but not forcing complexity on beginners

---

## The Code Structure

### **New Routes Added:**
- `GET /onboard` - Step 1 (Hardware check)
- `POST /onboard/models` - Step 2 (Model selection) 
- `POST /onboard/benchmark` - Step 3 (About to measure)
- `POST /measure-baseline` - Run baseline measurements locally
- `GET /tune` - Tuning dashboard
- `POST /measure-again` - Run another measurement after an optimization

### **New Templates:**
- `templates/onboard_wizard.html` - The 3-step wizard
- `templates/tune_dashboard.html` - Measurement history and tuning suggestions

### **Updated Files:**
- `main.py` - Added new routes + `get_available_models()` function + local measurement logic
- `templates/index.html` - Redesigned home page emphasizing the guided setup path

---

## What Happens When a Beginner Lands on the App

1. **Home page** shows:
   - Prominent "New to Jetson? Start here → Start 3-Step Setup" button
   - Below that: "Already familiar? Advanced workflow" (collapsed details showing diagnostic scan option)

2. **They click "Start 3-Step Setup"** → `/onboard`
   - See: "Let's check your hardware"
   - Hardware is checked automatically
   - If MAXN mode is off: Shows exact command with explanation
   - If MAXN mode is on: Shows green checkmark, "Ready to pick models"

3. **They click "Hardware ready"** → `/onboard/models`
   - See: Two model cards with friendly language
   - Can accept defaults or pick from dropdown
   - Click "Models ready - measure performance"

4. **System measures** → `/measure-baseline`
   - Takes ~30-60 seconds
   - Shows progress (kept simple, no scary details)
   - Results: Simple speed numbers

5. **They see Tuning Dashboard** → `/tune`
   - "Here are your baseline numbers"
   - "Here's what you can try"
   - "Try one, measure again, see what helps"

6. **From there**: They can run optimizations from other guides, measure again, compare

---

## How This Solves the Original Problems

| Problem | Solution |
|---------|----------|
| "I don't know what this app does" | Clear 3-step path: check hardware → pick models → measure |
| "Too much jargon" | All technical terms removed from main flow, plain English explanations |
| "Why do I need an API key?" | No API key required! Local measurements using Ollama directly |
| "Benchmark is confusing" | Simplified to: run inference, show response time and speed, that's it |
| "I don't know which model to pick" | Auto-selected defaults with clear roles (fast vs smart) |
| "Can't tell if I'm improving" | Before/after comparisons in simple format |
| "I feel lost after first measurement" | Tuning dashboard gives 3 concrete optimizations to try |
| "App assumes I'm a developer" | Entire UX redesigned for non-coders |

---

## Testing Instructions for You (Expert Persona)

1. **Visit home page**: `http://127.0.0.1:8080/`
   - Should show the new green "Start 3-Step Setup" button prominently

2. **Start onboarding**: Click the button
   - Step 1 appears asking about MAXN mode

3. **See model selection**:
   - Curl: `curl -X POST http://127.0.0.1:8080/onboard/models -d ""`
   - Should show two model dropdown cards

4. **Run baseline measurement**:
   - Curl: `curl -X POST http://127.0.0.1:8080/measure-baseline -d "fast_model=qwen2.5-coder:7b&smart_model=nemotron-3-nano:4b"`
   - Should show simple result cards

5. **See tuning dashboard**:
   - Visit: `http://127.0.0.1:8080/tune`
   - Should show baseline, optimization suggestions, measurement history

---

## What's Next (If You Want to Extend)

### **Immediate**: Make it even simpler
- Add "guided hand-holding" modal: "Here's what tokens/sec means..."
- Store measurement history in a file so it persists across sessions
- Show visual chart of performance over time

### **Next**: Real measurements
- Replace Ollama measurements with actual local vLLM/TensorRT calls
- Capture real TTFT data
- Show actual memory/thermal changes

### **Later**: Model marketplace
- "Popular models for coding: [carousel of 5 options]"
- "Popular models for chat: [carousel of 5 options]"
- Download counts and user ratings

### **Advanced**: Auto-tuning
- "Let me try 10 configurations and show you the Pareto frontier"
- "Best for speed", "Best for memory", "Best balanced"

---

## The Philosophy Behind This Redesign

**Old approach**: "Here's all the data, you figure it out"
**New approach**: "Here's exactly what to do next"

**Old audience**: Developers debugging systems
**New audience**: Vibe coders who want local AI but don't know hardware

**Old success metric**: "Did I find the bottleneck?"
**New success metric**: "Did I pick models and tune them without getting confused?"

The redesign assumes:
- User has no coding background ✓
- User has experience with cloud LLMs (ChatGPT, Claude) ✓
- User just got their first dev kit ✓
- User wants to "vibe code" with local AI ✓
- User is NOT a hardware expert ✓
- User wants simple, guided, predictable steps ✓

