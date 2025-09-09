# Awesome Edge AI for Multimodal Agents [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="image.png" alt="Awesome Edge AI"/>
</p>

> A curated list of **papers, frameworks, benchmarks, and applications** for efficient **multimodal agents** (LLMs, text-to-image, speech, world models, etc.) on **mobile and edge devices**.  
> Focused on **inference engines, optimization, and deployment** for real-world use.

---

## 📑 Contents
* [Introduction](#-introduction)
* [Papers](#-papers)
  * [Surveys & Overviews](#surveys--overviews)
  * [LLM Inference on Edge](#llm-inference-on-edge)
  * [Multimodal & Generative Models](#multimodal--generative-models)
  * [World Models & Embodied AI](#world-models--embodied-ai)
  * [Agent Systems on Edge](#agent-systems-on-edge)
* [Frameworks & Inference Engines](#-frameworks--inference-engines)
* [Optimization Techniques](#-optimization-techniques)
* [Benchmarks & Datasets](#-benchmarks--datasets)
* [Applications & Use Cases](#-applications--use-cases)
* [Community & Resources](#-community--resources)
* [Concluding Remarks](#concluding-remarks) 

---

## 🔹 Introduction
The next generation of **AI agents** is multimodal — capable of understanding and generating **text, images, speech, video, and embodied interactions**.  
Running these models on **mobile and edge devices** unlocks:
- **Privacy**: data stays on-device  
- **Low latency**: real-time interaction without cloud roundtrips  
- **Accessibility**: AI everywhere, even offline  
- **Efficiency**: tailored for constrained environments  

This repo tracks the latest progress in making multimodal AI **efficient, deployable, and agent-ready on edge hardware**.

---

## 📄 Papers

### 🔖 Surveys & Overviews

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| A Comprehensive Survey on On-Device AI Models | ACM Comput. Surveys | 2024 | [Paper](https://dl.acm.org/doi/10.1145/3724420) | Broad on-device overview (models, systems). |
| Mobile Edge Intelligence for Large Language Models | arXiv | 2024 | [Paper](https://arxiv.org/abs/2407.18921) | Survey of LLMs at mobile edge (latency, offload). |
| Efficient Diffusion Models: A Survey | arXiv | 2025 | [Paper](https://arxiv.org/abs/2502.06805) | Efficient diffusion (algo & systems) for edge. |
| Efficient Diffusion Models (IEEE TPAMI) | TPAMI | 2025 | [Paper](https://www.computer.org/csdl/journal/tp/2025/09/11002717/26GmRnP6FFe) | Practice-focused survey incl. deployment. |

### 🧠 LLM Inference on Edge

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| LLM as a System Service on Mobile Devices (LLMS) | arXiv | 2024 | [Paper](https://arxiv.org/abs/2403.11805) | KV-cache mgmt., compression & swapping on phones. |
| Bringing Open LLMs to Consumer Devices (MLC-LLM) | Blog | 2023 | [Post](https://blog.mlc.ai/2023/05/22/bringing-open-large-language-models-to-consumer-devices) | Universal deployment: phones, browsers, Apple/AMD/NVIDIA. |
| Llama.cpp (GGML) | GitHub | 2023– | [Repo](https://github.com/ggml-org/llama.cpp) | C/C++ local inference across CPUs/NPUs/GPUs. |
| Large Language Models on Mobile Devices: Measurements & Optimizations | MobiSys | 2024 | [Paper](https://dl.acm.org/doi/10.1145/3662006.3662059) | Empirical study of on-device LLM cost/latency. |

### 🖼️ Multimodal & Generative Models

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| MobileCLIP | CVPR | 2024 | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.pdf) \| [Code](https://github.com/apple/ml-mobileclip) | Image-text models optimized for iPhone latency. |
| LLaVA-Mini (1 vision token) | arXiv | 2025 | [Paper](https://arxiv.org/abs/2501.03895) | Compresses vision tokens → 1 token for LMMs. |
| MobileVLM | arXiv | 2023–24 | [Paper](https://arxiv.org/abs/2312.16886) \| [Code](https://github.com/Meituan-AutoML/MobileVLM) | VLM tuned for mobile throughput. |
| EdgeSAM | arXiv | 2023 | [Paper](https://arxiv.org/abs/2312.06660) \| [Proj](https://mmlab-ntu.github.io/project/edgesam/) | Distilled SAM at 30+ FPS on iPhone 14. |
| MiniCPM-V (efficient MLLM) | Nat. Commun. | 2025 | [Paper](https://www.nature.com/articles/s41467-025-61040-5) | On-device MLLM progress since 2024 releases. |

### 🌎 World Models & Embodied AI

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| AndroidWorld: Dynamic Benchmarking for Mobile Agents | arXiv | 2024 | [Paper](https://arxiv.org/abs/2405.14573) \| [Site](https://google-research.github.io/android_world/) | 116 tasks across 20 Android apps; agent eval. |

### 🤖 Agent Systems on Edge

| Title | Venue | Year | Materials | Description |
|:-----:|:-----:|:----:|:---------:|:-----------|
| MobiAgent: Systematic Framework for Customizable Mobile Agents | arXiv | 2025 | [Paper](https://arxiv.org/abs/2509.00531) | Mobile agent models + acceleration + benchmark suite. |
| EcoAgent: Edge–Cloud Collaborative Mobile Automation | arXiv | 2025 | [Paper](https://arxiv.org/abs/2505.05440) | Planner in cloud + execution/observation on-edge. |
| LLM as a System Service (OS-level integration) | arXiv | 2024 | [Paper](https://arxiv.org/abs/2403.11805) | System support for stateful on-device LLMs. |
| Mobile-Agent-v3 / GUI-Owl (GUI automation) | arXiv | 2025 | [Paper](https://arxiv.org/abs/2508.15144) | SOTA open models on AndroidWorld/OSWorld. |

---

## ⚙️ Frameworks & Inference Engines
- [ONNX Runtime](https://onnxruntime.ai/) — Cross-platform accelerator; hardware backends  
- [TensorRT](https://developer.nvidia.com/tensorrt) — Compiler + runtime for low-latency inference  
- [Core ML](https://developer.apple.com/machine-learning/core-ml/) — Apple on-device ML  
- [LiteRT (TensorFlow Lite)](https://www.tensorflow.org/lite) — Google’s on-device runtime  
- [MNN](https://github.com/alibaba/MNN) — Alibaba’s lightweight, efficient engine  
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Portable C/C++ LLM/VLM inference  
- [MLC-LLM](https://mlc.ai/mlc-llm/) — TVM-based universal deployment  

---

## 🛠️ Optimization Techniques

| Category | Methods / Papers | Description | Paper | Code |
| :------: | :--------------- | :----------- | :----: | :---: |
| **Quantization** | GPTQ, AWQ, SmoothQuant, OmniQuant, QuaRot, QLoRA, DoRA | W4/W8A8, group-wise or NF4 quantization; activation-aware scaling; outlier rotation; low-bit PEFT; LoRA decomposition for fine-tuning. | [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [QuaRot](https://arxiv.org/abs/2404.00456) / [QLoRA](https://arxiv.org/abs/2305.14314) / [DoRA](https://arxiv.org/abs/2402.09353) | [GPTQ](https://github.com/IST-DASLab/gptq) / [AWQ](https://github.com/mit-han-lab/llm-awq) / [SmoothQuant](https://github.com/mit-han-lab/smoothquant) / [QuaRot](https://github.com/Vahe1994/QuaRot) / [QLoRA](https://github.com/artidoro/qlora) |
| **KV-cache Quantization** | KVQuant, ZipCache, QAQ | 2–3 bit KV compression with <0.1 perplexity drop; enables million-token context windows and memory savings. | [KVQuant](https://arxiv.org/abs/2401.18079) / [ZipCache](https://arxiv.org/abs/2404.02878) / [QAQ](https://arxiv.org/abs/2401.06104) | [KVQuant](https://github.com/thu-nics/KVQuant) / [ZipCache](https://github.com/FMInference/ZipCache) |
| **Pruning & Sparsity** | SparseGPT, Wanda, Wanda++, Movement pruning, N:M sparsity | Unstructured/structured sparsity up to 60% with minimal accuracy loss; block- and activation-aware pruning for LLMs. | [SparseGPT](https://arxiv.org/abs/2301.00774) / [Wanda](https://arxiv.org/abs/2306.11695) / [Movement Pruning](https://arxiv.org/abs/2005.07683) | [SparseGPT](https://github.com/IST-DASLab/sparsegpt) / [Wanda](https://github.com/locuslab/wanda) |
| **Efficient Attention** | FlashAttention-3, PagedAttention (vLLM), MQA/GQA | Mixed-precision & warp-specialized kernels; KV cache paging; fewer KV heads for faster decode. | [FlashAttention-3](https://arxiv.org/abs/2407.08608) / [PagedAttention](https://arxiv.org/abs/2309.06180) | [FlashAttention](https://github.com/Dao-AILab/flash-attention) / [vLLM](https://github.com/vllm-project/vllm) |
| **Speculative & Multi-token Decoding** | Medusa, EAGLE, EAGLE-3 | Multi-head speculative decoding; feature- and token-level prediction; 2–3.6× speedup. | [Medusa](https://arxiv.org/abs/2309.02706) / [EAGLE](https://arxiv.org/abs/2401.10968) | [Medusa](https://github.com/FasterDecoding/Medusa) / [EAGLE](https://github.com/SafeAILab/EAGLE) |
| **Multimodal Compression** | ToMe, DynamicViT, LLaVA-Mini | Token merging/pruning for ViTs; dynamic vision token selection; extreme compression (1 vision token vs 576). | [ToMe](https://arxiv.org/abs/2210.09461) / [DynamicViT](https://arxiv.org/abs/2106.02034) / [LLaVA-Mini](https://arxiv.org/abs/2408.03326) | [ToMe](https://github.com/facebookresearch/ToMe) / [LLaVA-Mini](https://github.com/haotian-liu/LLaVA-Mini) |
| **Efficient Diffusion** | Consistency Models, LCM, LCM-LoRA, ADD, SDXL-Turbo, SnapFusion | Few-step or 1-step generation; distillation & adversarial training; mobile-ready pipelines for <2s inference. | [Consistency Models](https://arxiv.org/abs/2303.01469) / [LCM](https://arxiv.org/abs/2310.04378) / [ADD](https://arxiv.org/abs/2311.16290) / [SDXL-Turbo](https://stability.ai/news/stable-diffusion-xl-turbo) / [SnapFusion](https://arxiv.org/abs/2403.12036) | [LCM](https://github.com/luosiallen/latent-consistency-model) / [SDXL-Turbo](https://github.com/Stability-AI/generative-models) / [SnapFusion](https://github.com/SnapFusion/SnapFusion) |


---

## 📊 Benchmarks & Datasets

| Benchmark / Dataset         | Category                    | Description                                                                                     | Link |
|-----------------------------|-----------------------------|-------------------------------------------------------------------------------------------------|------|
| **MLPerf Tiny**             | Embedded / TinyML           | Industry-standard inference benchmark suite for ultra-low-power embedded devices (microcontrollers); covers tasks like keyword spotting, visual wake words, image classification, anomaly detection. Measures accuracy, latency, and energy. | [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/) |
| **AI Benchmark**            | Mobile AI                   | Mobile AI performance suite that scores AI workloads across devices, measuring CPU, GPU, and NPU performance. | [AI Benchmark](https://ai-benchmark.com/) |
| **AndroidWorld**            | UI Agent / Autonomous       | Dynamic benchmarking environment for autonomous agents controlling Android UIs. Contains 116 programmatically generated tasks across 20 apps; supports reproducible evaluation and robustness testing. | [AndroidWorld (GitHub)](https://github.com/google-research/android_world) |
| **Geekbench AI**            | Device AI Scoring           | AI-centric workload scoring benchmark that measures CPU, GPU, and NPU performance across a variety of AI tasks. | [Geekbench AI](https://www.geekbench.com/ai/) |
| **MLPerf Client**           | Client LLM / Desktop        | Client-side benchmarking toolkit for evaluating LLM and AI workloads on desktops, laptops, and similar devices. | [MLPerf – Client benchmarks](https://mlcommons.org/benchmarks/) |
| **AIoTBench**               | Mobile / Embedded (Legacy)  | Older mobile/embedded benchmark suite evaluating inference speed across mobile frameworks (TensorFlow Lite, Caffe2, PyTorch Mobile). Introduces metrics like VIPS and VOPS. | [AIoTBench (arXiv)](https://arxiv.org/abs/2005.05085) |


---

## 📱 Applications & Use Cases

| Category | Examples / Papers | Description | Paper | Code |
| :------: | :---------------- | :----------- | :---- | :--- |
| **On-device Chat Assistants** | MobileLLM, MobiLlama, EdgeMoE | Sub-billion or sparse LLMs optimized for phones; low memory/latency assistants. | [MobileLLM](https://arxiv.org/abs/2402.14905) / [MobiLlama](https://arxiv.org/abs/2402.16840) / [EdgeMoE](https://arxiv.org/abs/2308.14352) | [MobileLLM](https://github.com/facebookresearch/MobileLLM) / [MobiLlama](https://github.com/mbzuai-oryx/MobiLlama) |
| **Real-time Speech Translation & Vision** | Whisper, SeamlessM4T, MobileCLIP | On-device ASR + translation; efficient vision-language for realtime apps. | [Whisper](https://arxiv.org/abs/2212.04356) / [SeamlessM4T](https://arxiv.org/abs/2308.11596) / [MobileCLIP](https://arxiv.org/abs/2311.17049) | [Whisper](https://github.com/openai/whisper) |
| **AR/VR Embodied & GUI Agents** | Voyager, AppAgent, Mobile-Agent | Embodied agents (3D/VR) and GUI agents that operate smartphone apps. | [Voyager](https://arxiv.org/abs/2305.16291) / [AppAgent](https://arxiv.org/abs/2312.13771) / [Mobile-Agent](https://arxiv.org/abs/2401.16158) | [Voyager](https://github.com/MineDojo/Voyager) / [AppAgent](https://github.com/TencentQQGYLab/AppAgent) / [Mobile-Agent](https://github.com/X-PLUG/MobileAgent) |
| **Edge Creative Tools (Image/Video/Music)** | SnapFusion, MobileDiffusion, LCM/LCM-LoRA, SDXL-Turbo | Distillation/few-step diffusion for on-device image/video; single-step accelerators; practical mobile T2I. | [SnapFusion](https://arxiv.org/abs/2306.00980) / [MobileDiffusion](https://arxiv.org/abs/2311.16567) / [LCM](https://arxiv.org/abs/2310.04378) / [SDXL-Turbo](https://stability.ai/news/stability-ai-sdxl-turbo) | [SnapFusion](https://github.com/snap-research/SnapFusion) / [MobileDiffusion](https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/) / [LCM](https://github.com/luosiallen/latent-consistency-model) |
| **Robotics & IoT AI** | RT-2, Octo, OpenVLA, Mobile ALOHA | VLA policies and low-cost teleop datasets enabling general robot skills; efficient fine-tuning/serving. | [RT-2](https://arxiv.org/abs/2307.15818) / [Octo](https://arxiv.org/abs/2405.12213) / [OpenVLA](https://arxiv.org/abs/2406.09246) / [Mobile ALOHA](https://arxiv.org/abs/2401.02117) | [RT-2](https://robotics-transformer2.github.io/) / [OpenVLA](https://github.com/openvla/openvla) / [Mobile ALOHA](https://mobile-aloha.github.io/) |


---

## 🌍 Community & Resources
- [Awesome Edge AI](https://github.com/akshayubhat/awesome-edge-ai) — Related list 
- [MLC AI Community](https://mlc.ai/)  
- [ONNX Community](https://onnx.ai/)  

---


## 🤝 Contributing
Pull requests are welcome! Please follow the [Awesome List Guidelines](https://github.com/sindresorhus/awesome/blob/main/contributing.md).  

---
⭐️ Inspired by the vision of **efficient multimodal agents everywhere** — from phones to IoT to autonomous systems.