# 🌌 GenVision: GAN vs Diffusion-Based Image Generation and Comparison

> **Tagline:** *“One Prompt. Two Models. One Fair Comparison.”*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Diffusers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Live Demo & Repository Links

* 🚀 **LIVE DEMO (Hugging Face Spaces / Gradio Live)**: [https://huggingface.co/spaces/Akshaya267/GenVision](https://huggingface.co/spaces/Akshaya267/GenVision)
* 💻 **GITHUB REPOSITORY**: [https://github.com/Akshaya267/GenVision](https://github.com/Akshaya267/GenVision)

---

## 📌 Project Overview

**GenVision** is an end-to-end, empirical AI research web application designed to evaluate and compare **Generative Adversarial Networks (GANs)** against **Diffusion Models** under strict, fair, and reproducible scientific conditions.

Given a **single natural language prompt**, GenVision concurrently invokes both generative paradigms to synthesize output images at $512 \times 512$ resolution, performs automated computer vision & semantic metric evaluations, renders a 4-panel visual capability dashboard, dynamically computes benchmark winners, and provides diagnostic AI engineering commentary.

---

## ✨ Key Features

- **Fair Single-Prompt Testing**: Identical text prompts, seed controls, and evaluation conditions across both generative models.
- **Real Quantitative Vision Metrics**:
  - 🧠 **CLIP Semantic Alignment Score**: Measures text-to-image prompt adherence using OpenAI's `clip-vit-base-patch32`.
  - ⚡ **Generation Latency**: High-precision wall-clock timing ($s$) for single-step vs multi-step inference loops.
  - 🔍 **Image Sharpness**: OpenCV Laplacian variance ($\sigma^2$) quantifying fine spatial detail and high-frequency edge definition.
  - 🎨 **Color Diversity Index**: Standard deviation across RGB channels combined with multi-channel color histogram entropy.
  - 🕸️ **Edge Density Ratio**: Structural complexity measured via OpenCV Canny edge pixel detection.
- **Dynamic Winner Categorization**: Categorical badges for *Best Semantic Alignment*, *Fastest Generation*, *Highest Color Diversity*, *Highest Sharpness*, and *Overall Winner*.
- **4-Panel Analytics Dashboard**: Renders interactive Matplotlib visualizations:
  1. Absolute Metric Bar Chart
  2. Inference Latency Benchmark
  3. Min-Max Normalized Performance Scale ($0 - 100\%$)
  4. 5-Axis Capability Radar / Spider Chart
- **Automated AI Analysis Synthesis**: Generates technical diagnostic text explaining the structural trade-offs between GAN mapping networks and Diffusion denoising loops.
- **Export & Download Options**: Export individual GAN outputs, Diffusion outputs, benchmark charts, and comprehensive `.txt` benchmark reports.
- **Session Experiment Tracker**: Persistent history table tracking previous generations within the user's active session using `gr.State`.

---

## 🏗️ System Architecture & Model Pipelines

### 🔵 GAN Pipeline Architecture
```
Prompt → CLIP Text Encoder → Semantic Latent Mapper (W-Space) → DeepConv Generator → Image (512x512)
```
*Note on Text Conditioning*: Vanilla StyleGAN2 / DCGAN models are unconditional or class-conditioned. GenVision implements a **Text-Conditioned GAN Architecture**:
1. Prompt text is encoded via `openai/clip-vit-base-patch32` into a 512-dimensional semantic text embedding $\mathbf{z}_{\text{text}}$.
2. A multi-layer `SemanticLatentMapper` network projects $\mathbf{z}_{\text{text}}$ into the disentangled intermediate latent space $W \in \mathbb{R}^{512}$.
3. A PyTorch Progressive Deep Convolutional Generator (`TextConditionedGANGenerator`) transforms $W$ via transposed convolutions, batch normalization, and LeakyReLU activations into a $512 \times 512 \times 3$ RGB tensor.

---

### 🟣 Diffusion Pipeline Architecture
```
Prompt → CLIP Text Encoder → Latent Noise z₀ ~ N(0, I) → U-Net Denoising Loop → VAE Decoder → Image (512x512)
```
1. Uses `stabilityai/sd-turbo` via Hugging Face `diffusers.AutoPipelineForText2Image`.
2. Leverages Adversarial Diffusion Distillation (ADD) to reduce standard 50-step diffusion down to 1–4 fast inference steps.
3. Guided by text cross-attention layers, Gaussian noise is iteratively converted into crisp latent representations, which are decoded by the VAE into an RGB image.

---

## 📊 Metrics Calculation Methodology

| Metric | Scientific Formula / Function | Technical Interpretation |
| :--- | :--- | :--- |
| **CLIP Score** | $\text{CosineSim}(E_{\text{text}}(P), E_{\text{img}}(I)) \times 100$ | Measures semantic faithfulness to input prompt. |
| **Sharpness** | $\text{Var}(\nabla^2 \text{Image}_{\text{gray}})$ | Higher values indicate sharper focus & crisp details. |
| **Color Diversity** | $\bar{\sigma}_{\text{RGB}} \times \text{Entropy}_{\text{norm}}$ | Measures color spectrum richness and variation. |
| **Edge Density** | $\frac{\text{Count}(\text{Canny Edges})}{\text{Total Pixels}}$ | Evaluates structural detail and boundary complexity. |
| **Generation Time** | $\Delta t = t_{\text{end}} - t_{\text{start}}$ | Measures raw computational efficiency (seconds). |

---

## 📁 Project Structure

```
GenVision/
├── app.py                     # Main Gradio research interface & pipeline runner
├── requirements.txt           # Required Python packages & dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT Open Source License
├── .gitignore                 # Excludes cache, checkpoints, outputs & envs
├── models/
│   ├── __init__.py
│   ├── gan_generator.py       # Text-Conditioned GAN & Latent Mapper module
│   └── diffusion_generator.py # Stable Diffusion Turbo Hugging Face pipeline
├── metrics/
│   ├── __init__.py
│   ├── image_metrics.py       # Real CLIP, Laplacian, Color & Canny metrics
│   └── visualizer.py          # 4-Panel Matplotlib dashboard generator
├── utils/
│   ├── __init__.py
│   ├── image_utils.py         # Image saving, base64 conversion & report writer
│   └── model_utils.py         # CUDA/CPU detection & VRAM cache cleaner
├── screenshots/
│   └── .gitkeep               # Directory for UI screenshots
└── outputs/
    └── .gitkeep               # Directory for generated images & reports
```

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.10+
- Git
- NVIDIA GPU (Optional, recommended for fast CUDA execution)

### 1. Clone Repository
```bash
git clone https://github.com/Akshaya267/GenVision.git
cd GenVision
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python app.py
```
Open your browser at `http://localhost:7860`.

---

## ⚡ Google Colab Setup (Tesla T4)

To run on **Google Colab Tesla T4 GPU**:

1. Open Google Colab and set runtime to **GPU (Tesla T4)**.
2. Run the following cell:

```python
!git clone https://github.com/Akshaya267/GenVision.git
%cd GenVision
!pip install -r requirements.txt
!python app.py
```
Click the generated `*.gradio.live` link to open the live public interface.

---

## 🌐 Public Deployment (Hugging Face Spaces)

GenVision is optimized for deployment on **Hugging Face Spaces**:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Gradio** as the Space SDK.
3. Push the contents of the repository to your Hugging Face Space repository:
```bash
git remote add hf https://huggingface.co/spaces/Akshaya267/GenVision
git push hf main
```
4. Hugging Face Spaces will automatically build and launch the Gradio web interface.

---

## 🖼️ Screenshots Section

| Dashboard View | Model Comparison & Metrics |
| :---: | :---: |
| *(Add dashboard screenshot here)* | *(Add comparative metrics screenshot here)* |

---

## ⚠️ Limitations & Future Enhancements

### Current Limitations
- **CPU Latency**: Running Stable Diffusion on CPU fallback can take 10-25 seconds compared to < 1.0 second on T4 GPU.
- **GAN Text Generalization**: Text-conditioned GAN latent mappers trained on finite text-image embeddings are less versatile than multi-billion parameter diffusion U-Nets.

### Future Enhancements
- [ ] Integration of GigaGAN / StyleGAN-T text-conditioned GAN checkpoints.
- [ ] Support for SDXL Turbo, FLUX.1-schnell, and Midjourney-style prompt expansion.
- [ ] Human evaluation rating mode with pairwise preference logging.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author & Acknowledgments

- **Developer**: [Akshaya](https://github.com/Akshaya267)
- **Built With**: PyTorch, Hugging Face Diffusers, Transformers, Gradio, OpenCV, Matplotlib.
