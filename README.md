# Awesome Edge AI for Multimodal Agents [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="image.png" alt="Awesome Edge AI for Multimodal Agents"/>
</p>

<p align="center">
  <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Awesome"></a>
  <img src="https://img.shields.io/badge/focus-on--device%20%7C%20edge%20agents-0ea5e9" alt="Focus">
  <img src="https://img.shields.io/badge/modalities-text%20%7C%20vision%20%7C%20speech%20%7C%20action-8b5cf6" alt="Modalities">
  <img src="https://img.shields.io/badge/updated-2026-22c55e" alt="Updated 2026">
</p>

> A curated list of **papers, models, runtimes, hardware, benchmarks, and applications** for **multimodal agents** that run on **phones, NPUs, Jetson, wearables, and other edge devices**.
>
> Scope: **on-device / near-device** inference — not cloud-only agent frameworks.

**Why this list exists.** Cloud agents are powerful, but edge agents are what you can actually ship: private by default, sub-second, offline-capable, and cheap at scale. This repo tracks the stack that makes that possible — from 1-bit LLMs and mobile VLMs to GUI agents, speech pipelines, and NPU runtimes.

| Layer | What to look for |
| :---- | :--------------- |
| **Models** | Sub-3B SLMs, mobile VLMs, distilled diffusion, on-device ASR/TTS |
| **Systems** | KV cache, paging, speculative decode, NPU/GPU/ANE backends |
| **Agents** | GUI control, tool use, RAG, robotics / VLA, always-on assistants |
| **Eval** | Latency, energy, memory, success rate on real device UIs |

---

## 📑 Contents

* [Introduction](#-introduction)
* [Papers](#-papers)
  * [Surveys & Overviews](#surveys--overviews)
  * [LLM Inference on Edge](#llm-inference-on-edge)
  * [Multimodal & Generative Models](#multimodal--generative-models)
  * [Speech & Audio](#speech--audio)
  * [World Models & Embodied AI](#world-models--embodied-ai)
  * [Agent Systems on Edge](#agent-systems-on-edge)
* [On-Device Models](#-on-device-models)
* [Frameworks & Inference Engines](#-frameworks--inference-engines)
* [Hardware Platforms](#-hardware-platforms)
* [Optimization Techniques](#-optimization-techniques)
* [Benchmarks & Datasets](#-benchmarks--datasets)
* [Applications & Use Cases](#-applications--use-cases)
* [Community & Resources](#-community--resources)
* [Contributing](#-contributing)

---

## 🔹 Introduction

The next generation of **AI agents** is multimodal — they read screens, hear speech, generate images, and act in apps or the physical world. Running them on **mobile and edge hardware** unlocks:

- **Privacy** — raw audio, photos, and UI state never leave the device
- **Latency** — no WAN round-trip; agents can be always-on
- **Cost** — inference is paid in milliwatts, not tokens
- **Resilience** — works on airplanes, factories, and air-gapped networks

Typical edge budget (rule of thumb):

| Device class | Memory | Power | What fits today |
| :----------- | :----- | :---- | :-------------- |
| MCU / wearable | 1–64 MB | &lt;1 W | Keyword spotting, tiny ASR, 2–3B-unrealistic; use &lt;100M or cloud fallback |
| Phone / NPU SoC | 6–16 GB | 3–8 W | 1–8B SLM/VLM INT4, real-time ASR, few-step T2I, GUI agents |
| SBC / Jetson | 8–64 GB | 7–60 W | 7–32B local agents, VLA, multi-agent home/factory hubs |

This list is a map of that stack. Prefer items with **code, on-device numbers, or a real runtime** over cloud-only demos.

---

## 📄 Papers

### 🔖 Surveys & Overviews

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| A Comprehensive Survey on On-Device AI Models | ACM CSUR | 2025 | [Paper](https://dl.acm.org/doi/10.1145/3724420) | Models, systems, and deployment for on-device AI. |
| Mobile Edge Intelligence for Large Language Models | IEEE COMST | 2025 | [Paper](https://arxiv.org/abs/2407.18921) | LLMs at the mobile edge: latency, offload, serving. |
| Efficient Multimodal Large Language Models: A Survey | arXiv | 2024 | [Paper](https://arxiv.org/abs/2405.10739) | MLLM compression, vision tokens, and serving. |
| Efficient Diffusion Models: A Survey | arXiv | 2025 | [Paper](https://arxiv.org/abs/2502.06805) | Algorithm + systems view of fast diffusion. |
| A Survey of Resource-efficient LLM and Multimodal Foundation Models | arXiv | 2024 | [Paper](https://arxiv.org/abs/2401.08092) | Memory, compute, and energy across the LLM/MLLM stack. |
| Personal LLM Agents: Insights and Survey | arXiv | 2024 | [Paper](https://arxiv.org/abs/2401.05459) | Capability, efficiency, and security of personal agents. |

### 🧠 LLM Inference on Edge

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| LLM as a System Service on Mobile Devices | MobiCom | 2024 | [Paper](https://arxiv.org/abs/2403.11805) | OS-level KV cache, swapping, and multi-app sharing. |
| Large Language Models on Mobile Devices: Measurements & Optimizations | MobiSys | 2024 | [Paper](https://dl.acm.org/doi/10.1145/3662006.3662059) | Phone-side latency, energy, and thermal study. |
| PowerInfer-2: Fast Large Language Model Inference on a Smartphone | arXiv | 2024 | [Paper](https://arxiv.org/abs/2412.06607) \| [Code](https://github.com/SJTU-IPADS/PowerInfer) | Neuron-cluster sparsity; 47B model on a phone. |
| BitNet b1.58 / BitNet.cpp | arXiv | 2024 | [Paper](https://arxiv.org/abs/2402.17764) \| [Code](https://github.com/microsoft/BitNet) | 1.58-bit ternary LLMs; CPU-first edge inference. |
| Qwen2.5-Omni Technical Report | arXiv | 2025 | [Paper](https://arxiv.org/abs/2503.20215) | Streaming omni model (text/image/audio/video → text+speech). |
| MobileLLM | CVPR | 2024 | [Paper](https://arxiv.org/abs/2402.14905) \| [Code](https://github.com/facebookresearch/MobileLLM) | Sub-billion LLMs designed for phones, not just distilled. |
| EdgeMoE | MobiCom | 2024 | [Paper](https://arxiv.org/abs/2308.14352) | Expert-wise paging so MoE models fit in phone DRAM. |
| Transformer-Lite | arXiv | 2024 | [Paper](https://arxiv.org/abs/2403.20041) | High-efficiency LLM decode on phone GPUs (Qualcomm / MTK). |
| LLMCad | arXiv | 2023 | [Paper](https://arxiv.org/abs/2309.04255) | On-device speculative collaboration; up to 9.3× faster generation. |

### 🖼️ Multimodal & Generative Models

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| MobileCLIP / MobileCLIP2 | CVPR | 2024–25 | [Paper](https://arxiv.org/abs/2311.17049) \| [Code](https://github.com/apple/ml-mobileclip) | Fast image–text models; iPhone-class latency. |
| MiniCPM-V / MiniCPM-o | Nat. Commun. | 2025 | [Paper](https://www.nature.com/articles/s41467-025-61040-5) \| [Code](https://github.com/OpenBMB/MiniCPM-V) | Strong on-device MLLM; GPT-4V-level at phone scale. |
| InternVL / InternVL2-Mobile | CVPR | 2024 | [Paper](https://arxiv.org/abs/2312.14261) \| [Code](https://github.com/OpenGVLab/InternVL) | Open VLM family with mobile-size checkpoints. |
| LLaVA-Mini (1 vision token) | arXiv | 2025 | [Paper](https://arxiv.org/abs/2501.03895) | Extreme vision-token compression for LMMs. |
| MobileVLM / MobileVLM V2 | arXiv | 2024 | [Paper](https://arxiv.org/abs/2312.16886) \| [Code](https://github.com/Meituan-AutoML/MobileVLM) | VLM tuned for mobile throughput. |
| FastVLM | CVPR | 2025 | [Paper](https://arxiv.org/abs/2412.13303) \| [Code](https://github.com/apple/ml-fastvlm) | Apple hybrid vision encoder; Time-to-First-Token on iPhone. |
| EdgeSAM | ICCV | 2023 | [Paper](https://arxiv.org/abs/2312.06660) \| [Proj](https://mmlab-ntu.github.io/project/edgesam/) | Distilled SAM at 30+ FPS on iPhone 14. |
| SnapFusion / MobileDiffusion | ICML / arXiv | 2023–24 | [Paper](https://arxiv.org/abs/2306.00980) | &lt;2s on-device text-to-image. |
| SDXL-Turbo / LCM | ICLR | 2024 | [Paper](https://arxiv.org/abs/2311.17042) | 1–4 step diffusion; practical mobile T2I backbone. |

### 🔊 Speech & Audio

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| Whisper / distil-whisper | ICML | 2023–24 | [Paper](https://arxiv.org/abs/2212.04356) \| [Code](https://github.com/openai/whisper) | Robust ASR; distilled variants fit phones. |
| WhisperKit / whisper.cpp | GitHub | 2024– | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) \| [WhisperKit](https://github.com/argmaxinc/WhisperKit) | On-device ASR on ANE / CPU / GPU. |
| Moonshine | arXiv | 2024 | [Paper](https://arxiv.org/abs/2410.15608) \| [Code](https://github.com/usefulsensors/moonshine) | Tiny encoder-decoder ASR for edge, not a Whisper clone. |
| Moshi | arXiv | 2024 | [Paper](https://arxiv.org/abs/2410.00037) \| [Code](https://github.com/kyutai-labs/moshi) | Full-duplex spoken LM; ~200 ms practical latency. |
| Qwen2-Audio / Qwen2.5-Omni | arXiv | 2024–25 | [Paper](https://arxiv.org/abs/2407.10759) | Open audio-language models used in edge agents. |
| Piper / Kokoro TTS | GitHub | 2023– | [Piper](https://github.com/rhasspy/piper) \| [Kokoro](https://github.com/hexgrad/kokoro) | Fast local neural TTS for phones and SBCs. |

### 🌎 World Models & Embodied AI

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| AndroidWorld | NeurIPS | 2024 | [Paper](https://arxiv.org/abs/2405.14573) \| [Code](https://github.com/google-research/android_world) | 116 Android tasks; the default mobile-agent eval. |
| OSWorld | ICLR | 2025 | [Paper](https://arxiv.org/abs/2404.07972) \| [Site](https://os-world.github.io/) | Real desktop OS tasks for computer-use agents. |
| OpenVLA | CoRL | 2024 | [Paper](https://arxiv.org/abs/2406.09246) \| [Code](https://github.com/openvla/openvla) | Open vision-language-action policy. |
| π0 / π0.5 (Physical Intelligence) | arXiv | 2024–25 | [Paper](https://www.physicalintelligence.company/blog/pi0) | Generalist robot VLA; distillation path to edge. |
| TinyVLA | arXiv | 2024 | [Paper](https://arxiv.org/abs/2409.12514) | Compact VLA for onboard robot compute. |
| Mobile ALOHA | arXiv | 2024 | [Paper](https://arxiv.org/abs/2401.02117) \| [Site](https://mobile-aloha.github.io/) | Low-cost mobile manipulation + imitation. |

### 🤖 Agent Systems on Edge

| Title | Venue | Year | Materials | Description |
|:------|:-----:|:----:|:---------:|:------------|
| MobiAgent | arXiv | 2025 | [Paper](https://arxiv.org/abs/2509.00531) | Customizable mobile agents + acceleration + bench. |
| EcoAgent | arXiv | 2025 | [Paper](https://arxiv.org/abs/2505.05440) | Cloud planner + on-device execution/observation. |
| Mobile-Agent-v3 / GUI-Owl | arXiv | 2025 | [Paper](https://arxiv.org/abs/2508.15144) | Strong open GUI agents on AndroidWorld / OSWorld. |
| UI-TARS | arXiv | 2025 | [Paper](https://arxiv.org/abs/2501.12326) \| [Code](https://github.com/bytedance/UI-TARS) | End-to-end GUI agent; native screenshot-to-action. |
| AppAgent / AppAgent v2 | arXiv | 2024 | [Paper](https://arxiv.org/abs/2312.13771) \| [Code](https://github.com/TencentQQGYLab/AppAgent) | Learn to operate smartphone apps from screenshots. |
| FlashTTS | arXiv | 2025 | [Paper](https://arxiv.org/abs/2509.00195) | Test-time scaling for agentic LLMs on edge; +2.2× goodput. |
| AutoDroid / AutoDroid-V2 | MobiCom | 2024–25 | [Paper](https://arxiv.org/abs/2308.15272) | On-device + LLM hybrid Android task automation. |
| SeeClick / ShowUI | CVPR / arXiv | 2024–25 | [Paper](https://arxiv.org/abs/2401.10935) | Grounded GUI understanding for click agents. |


---

## 🧩 On-Device Models

Small models that are actually used as agent backbones on phones, NPUs, and SBCs.

| Model | Size | Modality | Why it matters on edge | Links |
|:------|:----:|:---------|:-----------------------|:------|
| **Qwen2.5 / Qwen3** (Instruct) | 0.5B–14B | Text | Default open SLM; strong tool use at 1.5–7B | [Qwen](https://github.com/QwenLM/Qwen2.5) |
| **Llama 3.2** | 1B / 3B | Text | Meta’s phone-first SLMs; ExecuTorch path | [Llama](https://www.llama.com/) |
| **Gemma 2 / Gemma 3** | 2B–27B | Text / VLM | Apache-friendly; Gemma 3 has vision at small sizes | [Gemma](https://ai.google.dev/gemma) |
| **Phi-4-mini / Phi-3.5-mini** | ~3.8B | Text | Dense quality at phone RAM | [Phi](https://huggingface.co/microsoft) |
| **SmolLM2 / SmolVLM** | 135M–1.7B | Text / VLM | Hugging Face tiny models; browser + phone | [SmolLM](https://github.com/huggingface/smollm) |
| **MobileLLM** | 125M–1B | Text | Architecture search for phones, not just distill | [Code](https://github.com/facebookresearch/MobileLLM) |
| **MiniCPM-V / MiniCPM-o** | ~8B (int4 ~5 GB) | V / A / T | Best-in-class on-device MLLM | [Code](https://github.com/OpenBMB/MiniCPM-V) |
| **FastVLM** | 0.5B–1.5B | VLM | Apple; low TTFT on iPhone | [Code](https://github.com/apple/ml-fastvlm) |
| **InternVL2-1B/2B/4B** | 1–4B | VLM | Open mobile VLM checkpoints | [Code](https://github.com/OpenGVLab/InternVL) |
| **BitNet b1.58 2B** | 2B (1.58-bit) | Text | CPU-native ternary inference | [BitNet](https://github.com/microsoft/BitNet) |
| **Moonshine Tiny/Base** | 27M / 61M | ASR | Edge speech without Whisper-scale decode | [Code](https://github.com/usefulsensors/moonshine) |
| **Piper / Kokoro** | &lt;100M | TTS | Local voice for agents | [Piper](https://github.com/rhasspy/piper) |

Quantization cheat sheet: **Q4_K_M / AWQ-INT4** is the default phone/Jetson tradeoff. Use **Q8** only if quality is the bottleneck; **1.58-bit / 2-bit** when RAM is the bottleneck.

---

## ⚙️ Frameworks & Inference Engines

### LLM / VLM runtimes

| Engine | Best on | Notes |
|:-------|:--------|:------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU, Metal, Vulkan, some NPUs | De-facto portable GGUF stack; multimodal via `llava`/`minicpm` |
| [MLC-LLM](https://llm.mlc.ai/) | iOS, Android, WebGPU, Metal | TVM compile-once, run everywhere |
| [ExecuTorch](https://pytorch.org/executorch/) | iOS, Android, MCU | PyTorch-native on-device; Llama 3.2 official path |
| [vLLM](https://github.com/vllm-project/vllm) / [LMDeploy](https://github.com/InternLM/lmdeploy) | Jetson, edge GPU | PagedAttention serving for 7–32B local agents |
| [SGLang](https://github.com/sgl-project/sglang) | Edge GPU | Fast structured decode / tool-calling servers |
| [Ollama](https://ollama.com/) | Laptop, SBC | Easiest local agent backend |
| [MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference) | Android, Web | Google’s on-device LLM API |
| [MNN](https://github.com/alibaba/MNN) | Android, iOS | Alibaba lightweight engine + LLM |
| [NCNN](https://github.com/Tencent/ncnn) | Mobile CPU/GPU | Battle-tested CV/ASR engine |
| [ONNX Runtime](https://onnxruntime.ai/) / [ORT-GenAI](https://github.com/microsoft/onnxruntime-genai) | Cross-platform | DirectML, CoreML, QNN, TensorRT EPs |
| [TensorRT / TensorRT-LLM](https://developer.nvidia.com/tensorrt) | NVIDIA dGPU, Jetson | Highest Jetson throughput |
| [Core ML](https://developer.apple.com/machine-learning/core-ml/) + [mlx](https://github.com/ml-explore/mlx) | Apple ANE / GPU | Native Apple Silicon path |
| [LiteRT](https://ai.google.dev/edge/litert) (TFLite) | Android NPU, MCU | Google on-device runtime |
| [QNN](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Snapdragon NPU | Hexagon offload for phones |
| [OpenVINO](https://docs.openvino.ai/) | Intel NPU/GPU | Laptops, industrial PCs |
| [llama.rn](https://github.com/mybigday/llama.rn) | React Native | On-device LLM in mobile apps |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) / [WhisperKit](https://github.com/argmaxinc/WhisperKit) | CPU / ANE | Local ASR |

### Agent / app layers

- [OpenClaw](https://github.com/openclaw/openclaw) — Self-hosted multi-agent assistant on Jetson-class boxes
- [ForestHub edge-agents](https://github.com/ForestHubAI/edge-agents) — Offline agents on Pi / Jetson / industrial gateways (GPIO, UART, MQTT)
- [Airgap](https://github.com/xmpuspus/airgap) — React Native on-device RAG agents (Gemma + llama.rn)
- [LangGraph](https://github.com/langchain-ai/langgraph) / [LlamaIndex](https://www.llamaindex.ai/) — Orchestration that can target a local runtime
- [Dify](https://github.com/langgenius/dify) — Visual agent builder, often paired with Ollama on the LAN
- [Ivy Tendril](https://github.com/Ivy-Interactive/Ivy-Tendril) — Agentic software factory with parallel Git worktrees (dev-time, not on-device)

---

## 🖥️ Hardware Platforms

| Platform | Typical TOPS / RAM | Sweet spot |
|:---------|:-------------------|:-----------|
| **Apple A17–M4** (ANE + Metal) | ~35 TOPS ANE / 8–36 GB | Best phone/laptop UX; Core ML + mlx + llama.cpp Metal |
| **Snapdragon 8 Gen 3 / 8 Elite** | Hexagon NPU | Android agents; QNN + MNN + MediaPipe |
| **MediaTek Dimensity 9300/9400** | APU | Android NPUs via NeuroPilot / LiteRT |
| **Google Tensor / Pixel** | TPU + GPU | AICore, Gemini Nano, on-device ASR |
| **NVIDIA Jetson Orin Nano / NX / AGX** | 40–275 TOPS / 8–64 GB | Home/factory agent hubs, VLA, local vLLM |
| **Qualcomm Dragonwing / RB3** | Industrial NPU | Robotics, cameras, cars |
| **Raspberry Pi 5 + Hailo / Coral** | 13–26 TOPS add-on | Cheap always-on vision + SLM offload |
| **Intel Core Ultra (Meteor/Lunar Lake NPU)** | 10–48 TOPS NPU | PC Copilot+ / OpenVINO agents |
| **ESP32-S3 / STM32 + NPU MCUs** | mW class | KWS, tiny ASR; cloud or phone for the LLM |
| **Galaxy Watch / Wear OS** | MB-class | Keyword + tiny binary; LLM in the phone/cloud (see ClawWatch) |

Rule of thumb: **put perception (ASR, CLIP, GUI encoder) on the NPU; keep decode of 1–8B SLMs on GPU/ANE/CPU; reserve cloud for rare hard planning.**

---

## 🛠️ Optimization Techniques

|                Category                | Methods / Papers                                               | Description                                                                                                                                                                                    |                                                                                                                                   Paper                                                                                                                                  |                                                                                                                        Code                                                                                                                        |
| :------------------------------------: | :------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|            **Quantization**            | GPTQ, AWQ, SmoothQuant, OmniQuant, QuaRot, QLoRA, DoRA         | W4/W8A8, group-wise or NF4 quantization; activation-aware scaling; outlier rotation; low-bit PEFT; LoRA decomposition for fine-tuning.                                                         | [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [QuaRot](https://arxiv.org/abs/2404.00456) / [QLoRA](https://arxiv.org/abs/2305.14314) / [DoRA](https://arxiv.org/abs/2402.09353) | [GPTQ](https://github.com/IST-DASLab/gptq) / [AWQ](https://github.com/mit-han-lab/llm-awq) / [SmoothQuant](https://github.com/mit-han-lab/smoothquant) / [QuaRot](https://github.com/Vahe1994/QuaRot) / [QLoRA](https://github.com/artidoro/qlora) |
|        **KV-cache Quantization**       | KVQuant, ZipCache, QAQ                                         | 2–3 bit KV compression with <0.1 perplexity drop; enables million-token context windows and memory savings.                                                                                    |                                                                   [KVQuant](https://arxiv.org/abs/2401.18079) / [ZipCache](https://arxiv.org/abs/2404.02878) / [QAQ](https://arxiv.org/abs/2401.06104)                                                                   |                                                                        [KVQuant](https://github.com/thu-nics/KVQuant) / [ZipCache](https://github.com/FMInference/ZipCache)                                                                        |
|         **Pruning & Sparsity**         | SparseGPT, Wanda, Wanda++, Movement pruning, N\:M sparsity     | Unstructured/structured sparsity up to 60% with minimal accuracy loss; block- and activation-aware pruning for LLMs.                                                                           |                                                             [SparseGPT](https://arxiv.org/abs/2301.00774) / [Wanda](https://arxiv.org/abs/2306.11695) / [Movement Pruning](https://arxiv.org/abs/2005.07683)                                                             |                                                                          [SparseGPT](https://github.com/IST-DASLab/sparsegpt) / [Wanda](https://github.com/locuslab/wanda)                                                                         |
|         **Efficient Attention**        | FlashAttention-3, PagedAttention (vLLM), MQA/GQA               | Mixed-precision & warp-specialized kernels; KV cache paging; fewer KV heads for faster decode.                                                                                                 |                                                                                 [FlashAttention-3](https://arxiv.org/abs/2407.08608) / [PagedAttention](https://arxiv.org/abs/2309.06180)                                                                                |                                                                    [FlashAttention](https://github.com/Dao-AILab/flash-attention) / [vLLM](https://github.com/vllm-project/vllm)                                                                   |
| **Speculative & Multi-token Decoding** | Medusa, EAGLE, EAGLE-3                                         | Multi-head speculative decoding; feature- and token-level prediction; 2–3.6× speedup.                                                                                                          |                                                                                          [Medusa](https://arxiv.org/abs/2309.02706) / [EAGLE](https://arxiv.org/abs/2401.10968)                                                                                          |                                                                          [Medusa](https://github.com/FasterDecoding/Medusa) / [EAGLE](https://github.com/SafeAILab/EAGLE)                                                                          |
|       **Multimodal Compression**       | ToMe, DynamicViT, LLaVA-Mini                                   | Token merging/pruning for ViTs; dynamic vision token selection; extreme compression (1 vision token vs 576).                                                                                   |                                                                [ToMe](https://arxiv.org/abs/2210.09461) / [DynamicViT](https://arxiv.org/abs/2106.02034) / [LLaVA-Mini](https://arxiv.org/abs/2408.03326)                                                                |                                                                     [ToMe](https://github.com/facebookresearch/ToMe) / [LLaVA-Mini](https://github.com/haotian-liu/LLaVA-Mini)                                                                     |
|         **Efficient Diffusion**        | Consistency Models, LCM, LCM-LoRA, ADD, SDXL-Turbo, SnapFusion | Few-step or 1-step generation; distillation & adversarial training; mobile-ready pipelines for <2s inference.                                                                                  |       [Consistency Models](https://arxiv.org/abs/2303.01469) / [LCM](https://arxiv.org/abs/2310.04378) / [ADD](https://arxiv.org/abs/2311.16290) / [SDXL-Turbo](https://stability.ai/news/stability-ai-sdxl-turbo) / [SnapFusion](https://arxiv.org/abs/2403.12036)      |                              [LCM](https://github.com/luosiallen/latent-consistency-model) / [SDXL-Turbo](https://github.com/Stability-AI/generative-models) / [SnapFusion](https://github.com/SnapFusion/SnapFusion)                              |
|    **System-level TTS Optimization**   | FlashTTS                                                   | Fast test-time scaling for agentic LLMs on edge; speculative beam extension, dynamic prefix scheduling, memory-aware model allocation. 2.2× higher goodput, 38–68% latency reduction vs. vLLM. |                                                                                                               [FlashTTS](https://arxiv.org/abs/2509.00195)                                                                                                               |                                                                                                                          –                                                                                                                         |
|         **On-device split / offload**        | EdgeMoE, Transformer-Lite, EcoAgent, LLMCad                 | Keep perception and decode on-device; page experts; use a tiny draft model; send only hard planning to the cloud.                                                                               |                                                             [EdgeMoE](https://arxiv.org/abs/2308.14352) / [Transformer-Lite](https://arxiv.org/abs/2403.20041) / [LLMCad](https://arxiv.org/abs/2309.04255)                                                             |                                                                                           [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)                                                                                          |
|              **1-bit / ternary**             | BitNet b1.58, BitNet.cpp                                    | Ternary weights for CPU-first phones and SBCs when INT4 still does not fit RAM.                                                                                                                |                                                                                                               [BitNet](https://arxiv.org/abs/2402.17764)                                                                                                                |                                                                                             [BitNet.cpp](https://github.com/microsoft/BitNet)                                                                                             |



---

## 📊 Benchmarks & Datasets

Measure **task success**, not only tokens/s. For agents, log latency, energy, RAM, and thermal throttling on the real device.

| Benchmark | Category | What it actually tests | Link |
|:----------|:---------|:-----------------------|:-----|
| **AndroidWorld** | Mobile GUI agent | 116 tasks / 20 Android apps; the default phone-agent eval | [GitHub](https://github.com/google-research/android_world) |
| **OSWorld** | Desktop computer-use | Real Ubuntu/Windows apps; multi-step GUI | [Site](https://os-world.github.io/) |
| **AITW / AndroidControl** | Mobile GUI | Large-scale Android action traces | [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild) |
| **ScreenSpot / ScreenSpot-Pro** | GUI grounding | Click the right widget from a screenshot | [ScreenSpot](https://github.com/njucckevin/SeeClick) |
| **MLPerf Tiny** | MCU / TinyML | KWS, VWW, image clf, anomaly; latency + energy | [MLCommons](https://mlcommons.org/benchmarks/inference-tiny/) |
| **MLPerf Client / MLPerf Inference** | Client / edge LLM | Laptop and edge-server LLM serving | [MLCommons](https://mlcommons.org/benchmarks/) |
| **Geekbench AI** | Device scoring | CPU / GPU / NPU AI score across phones and PCs | [Geekbench AI](https://www.geekbench.com/ai/) |
| **AI Benchmark** | Mobile SoC | Classic mobile NPU/GPU CNN/Transformer suite | [ai-benchmark.com](https://ai-benchmark.com/) |
| **OpenCompass / MMBench / MMMU** | VLM quality | Use *together with* on-device TTFT and RAM | [OpenCompass](https://github.com/open-compass/opencompass) |
| **AIoTBench** | Legacy mobile | Older TFLite / Caffe2 / PyTorch Mobile numbers | [arXiv](https://arxiv.org/abs/2005.05085) |

Suggested on-device report template: `model + quant + runtime + SoC → TTFT, tok/s, RAM peak, battery mAh / 1k tokens, thermal after 5 min`.

---

## 📱 Applications & Use Cases

| Category | Examples | Description | Links |
|:---------|:---------|:------------|:------|
| **On-device chat** | MobileLLM, Gemma 3, Llama 3.2, Phi-4-mini | Sub-3B / INT4 assistants in the phone process | [MobileLLM](https://github.com/facebookresearch/MobileLLM) · [Gemma](https://ai.google.dev/gemma) |
| **GUI / computer-use agents** | UI-TARS, AppAgent, Mobile-Agent, AutoDroid | Screenshot → tap/type on Android or desktop | [UI-TARS](https://github.com/bytedance/UI-TARS) · [AppAgent](https://github.com/TencentQQGYLab/AppAgent) · [Mobile-Agent](https://github.com/X-PLUG/MobileAgent) |
| **Speech agents** | Whisper.cpp, Moonshine, Moshi, Piper | Always-on ASR + local TTS; duplex where RAM allows | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) · [Moshi](https://github.com/kyutai-labs/moshi) · [Piper](https://github.com/rhasspy/piper) |
| **Wearable voice** | ClawWatch (NullClaw + Vosk) | Native watch agent: 2.8 MB Zig binary + 68 MB offline STT; ~1 MB RAM | [ClawWatch](https://github.com/ThinkOffApp/ClawWatch) |
| **Home / always-on hubs** | OpenClaw, ClawBox | Jetson Orin Nano box: multi-agent, browser, Telegram/WhatsApp | [OpenClaw](https://github.com/openclaw/openclaw) · [ClawBox](https://home-ai-assistant.com) |
| **Offline support / RAG** | Airgap | React Native + llama.rn; on-device Gemma + MiniSearch RAG | [Airgap](https://github.com/xmpuspus/airgap) |
| **Industrial / IoT** | ForestHub edge-agents | Pi / Jetson / gateways; GPIO, UART, MQTT as first-class nodes | [edge-agents](https://github.com/ForestHubAI/edge-agents) |
| **Robotics / VLA** | OpenVLA, TinyVLA, Mobile ALOHA, π0 | Onboard policies; distill giants down to Jetson-class | [OpenVLA](https://github.com/openvla/openvla) · [TinyVLA](https://tiny-vla.github.io/) · [Mobile ALOHA](https://mobile-aloha.github.io/) |
| **On-device T2I / avatars** | SnapFusion, MobileDiffusion, LCM, NanoAvatar | Few-step image gen; Android talking-head demo | [LCM](https://github.com/luosiallen/latent-consistency-model) · [NanoAvatar](https://github.com/wpydcr/NanoAvatar) |
| **Privacy / air-gap** | llama.cpp + local RAG | Factories, hospitals, airplanes — no token leaves the LAN | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **Agent identity (experimental)** | TWZRD Agent Intel | MCP trust scoring before agent-to-agent payments | [intel.twzrd.xyz](https://intel.twzrd.xyz) |

### Starter stacks (copy-paste)

| Goal | Stack |
|:-----|:------|
| **iPhone chat + vision** | FastVLM or MiniCPM-V → Core ML / MLC-LLM / mlx |
| **Android GUI agent** | UI-TARS or Mobile-Agent-v3 + MediaPipe / MNN INT4 |
| **Jetson home agent** | Qwen2.5-7B-Instruct AWQ + vLLM or llama.cpp CUDA + Piper |
| **Pi always-on sensor agent** | Hailo/Coral for vision + 1–3B GGUF on CPU + MQTT tools |
| **Watch / MCU** | Keyword spotting locally; LLM on the paired phone |

---

## 🌍 Community & Resources

**Lists & docs**
- [Awesome Edge AI](https://github.com/akshayubhat/awesome-edge-ai) — broader edge-AI list
- [Awesome On-Device LLM](https://github.com/NiuTrans/On-Device-LLMs) — on-device LLM papers
- [MLC AI](https://mlc.ai/) · [ONNX](https://onnx.ai/) · [ExecuTorch](https://pytorch.org/executorch/)
- [Google AI Edge](https://ai.google.dev/edge) — LiteRT, MediaPipe, on-device Gemini Nano
- [Apple Machine Learning Research](https://machinelearning.apple.com/) — FastVLM, MobileCLIP, mlx

**Hardware & products**
- [ClawBox](https://home-ai-assistant.com) — pre-configured Jetson assistant
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) · [Qualcomm AI Hub](https://aihub.qualcomm.com/)

**Where to watch**
- Venues: MobiCom, MobiSys, ASPLOS, MLSys, NeurIPS, CVPR, CoRL
- Hugging Face collections tagged `on-device`, `gguf`, `executorch`

---

## 🤝 Contributing

PRs welcome. Follow the [Awesome List Guidelines](https://github.com/sindresorhus/awesome/blob/main/contributing.md).

**In scope:** on-device / near-device multimodal models, runtimes, hardware, agents, and evals with a link (paper, code, or product).

**Out of scope:** cloud-only agent frameworks with no edge path; closed demos with no numbers.

Please include: one-line description, year, hardware class (phone / NPU / Jetson / MCU), and a working URL.

---

⭐️ Inspired by the vision of **efficient multimodal agents everywhere** — from watches and phones to factory gateways and robots.