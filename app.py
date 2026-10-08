import os
import sys
import time
import pandas as pd
from PIL import Image
import gradio as gr

# Add project root directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models.gan_generator import GANModelWrapper
from models.diffusion_generator import DiffusionModelWrapper
from metrics.image_metrics import evaluate_both_models
from metrics.visualizer import create_comparison_dashboard
from utils.image_utils import save_generated_image, generate_text_report
from utils.model_utils import get_device_and_dtype, clear_memory

# Initialize Model Wrappers
device, dtype = get_device_and_dtype()
gan_model = GANModelWrapper()
diffusion_model = DiffusionModelWrapper()

# Custom CSS for Dark Futuristic Research Dashboard
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg-dark: #0a0f1d;
    --card-bg: #11192e;
    --border-cyan: #00f0ff;
    --accent-purple: #7000ff;
    --text-glow: 0 0 10px rgba(0, 240, 255, 0.5);
}

body, .gradio-container {
    background-color: #0a0f1d !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

.main-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(180deg, #11192e 0%, #0a0f1d 100%);
    border-bottom: 2px solid #00f0ff;
    border-radius: 0 0 20px 20px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 25px rgba(0, 240, 255, 0.15);
}

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00f0ff, #7000ff, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}

.tagline {
    font-size: 1.2rem;
    color: #94a3b8;
    font-style: italic;
}

.glass-card {
    background: rgba(17, 25, 46, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

.generate-btn {
    background: linear-gradient(135deg, #00f0ff 0%, #7000ff 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}

.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.8) !important;
}

.winner-badge {
    background: linear-gradient(90deg, #10b981, #059669);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9rem;
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 8px;
}

.arch-box {
    background: #1e293b;
    border-left: 4px solid #00f0ff;
    padding: 12px 18px;
    border-radius: 8px;
    font-family: monospace;
    color: #e2e8f0;
    margin: 8px 0;
}
"""


def process_generation_and_comparison(prompt: str, seed: int, steps: int, history_state: list, progress=gr.Progress()):
    """
    Main orchestration function:
    1. Generates image via Text-Conditioned GAN
    2. Generates image via Stable Diffusion Turbo
    3. Calculates real metrics (CLIP, Sharpness, Color Diversity, Edge Density, Latency)
    4. Renders Matplotlib benchmark charts
    5. Determines categorical & overall winners dynamically
    6. Formats report & returns Gradio UI elements
    """
    if not prompt or len(prompt.strip()) == 0:
        raise gr.Error("Please enter a valid text prompt before generating!")

    progress(0.1, desc="⚡ Initializing AI Models on " + device.upper() + "...")
    status_msg = f"🟢 Hardware Accelerator: {device.upper()} | Model Loading..."

    # 1. Generate GAN Image
    progress(0.3, desc="🎨 Synthesizing image using Text-Conditioned GAN...")
    gan_img, gan_time = gan_model.generate(prompt, seed=int(seed), resolution=512)
    gan_img_path = save_generated_image(gan_img, prefix="gan")

    # 2. Generate Diffusion Image
    progress(0.6, desc="✨ Synthesizing image using Stable Diffusion Turbo...")
    diff_img, diff_time = diffusion_model.generate(prompt, num_inference_steps=int(steps), seed=int(seed), resolution=512)
    diff_img_path = save_generated_image(diff_img, prefix="diff")

    # 3. Calculate REAL Comparative Metrics & Dynamic Winners
    progress(0.85, desc="📊 Computing CLIP score, OpenCV sharpness, color diversity, and edge density...")
    metrics_dict, winner_summary = evaluate_both_models(
        prompt=prompt,
        gan_image=gan_img,
        diff_image=diff_img,
        gan_time=gan_time,
        diff_time=diff_time,
        gan_model_name=gan_model.model_name,
        diff_model_name=diffusion_model.model_name
    )

    # 4. Render 4-Panel Benchmark Visualizations
    chart_img = create_comparison_dashboard(metrics_dict)
    chart_path = save_generated_image(chart_img, prefix="chart")

    # 5. Format Metrics Table for UI Display
    df_metrics = pd.DataFrame([
        {
            "Metric": "Generation Latency (sec)",
            "GAN Model": f"{metrics_dict['GAN']['generation_time']:.3f} s",
            "Diffusion Model": f"{metrics_dict['Diffusion']['generation_time']:.3f} s"
        },
        {
            "Metric": "CLIP Semantic Score",
            "GAN Model": f"{metrics_dict['GAN']['clip_similarity']:.2f}%",
            "Diffusion Model": f"{metrics_dict['Diffusion']['clip_similarity']:.2f}%"
        },
        {
            "Metric": "Image Sharpness (Laplacian Var)",
            "GAN Model": f"{metrics_dict['GAN']['sharpness']:.2f}",
            "Diffusion Model": f"{metrics_dict['Diffusion']['sharpness']:.2f}"
        },
        {
            "Metric": "Color Diversity Index",
            "GAN Model": f"{metrics_dict['GAN']['color_diversity']:.2f}",
            "Diffusion Model": f"{metrics_dict['Diffusion']['color_diversity']:.2f}"
        },
        {
            "Metric": "Edge Density Ratio",
            "GAN Model": f"{metrics_dict['GAN']['edge_density']:.4f}",
            "Diffusion Model": f"{metrics_dict['Diffusion']['edge_density']:.4f}"
        },
        {
            "Metric": "Output Resolution",
            "GAN Model": metrics_dict['GAN']['resolution'],
            "Diffusion Model": metrics_dict['Diffusion']['resolution']
        }
    ])

    # 6. Format Winner Badges HTML
    winner_html = f"""
    <div style="background: rgba(17, 25, 46, 0.9); border: 1px solid #00f0ff; border-radius: 12px; padding: 15px; margin-top: 10px;">
        <h3 style="color: #00f0ff; margin-top: 0;">🏆 BENCHMARK WINNERS</h3>
        <div>
            <span class="winner-badge">🏆 Semantic Alignment: {winner_summary['clip_similarity']}</span>
            <span class="winner-badge">⚡ Generation Speed: {winner_summary['generation_time']}</span>
            <span class="winner-badge">🎨 Color Diversity: {winner_summary['color_diversity']}</span>
            <span class="winner-badge">🔍 Image Sharpness: {winner_summary['sharpness']}</span>
        </div>
        <h2 style="color: #facc15; margin-bottom: 0; margin-top: 12px;">{winner_summary['overall']} OVERALL WINNER</h2>
    </div>
    """

    # 7. Generate Downloadable Text Report
    report_file_path = generate_text_report(prompt, metrics_dict, winner_summary)

    # 8. Update Session History State
    new_history_entry = {
        "Timestamp": time.strftime("%H:%M:%S"),
        "Prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt,
        "GAN Time": f"{gan_time:.3f}s",
        "Diffusion Time": f"{diff_time:.3f}s",
        "Overall Winner": winner_summary['overall']
    }
    updated_history = [new_history_entry] + history_state
    history_df = pd.DataFrame(updated_history)

    progress(1.0, desc="✅ Complete!")

    return (
        gan_img,
        f"⏱️ Generation Time: {gan_time:.3f}s | 📐 Res: {gan_img.width}x{gan_img.height} | {gan_model.model_name}",
        diff_img,
        f"⏱️ Generation Time: {diff_time:.3f}s | 📐 Res: {diff_img.width}x{diff_img.height} | {diffusion_model.model_name}",
        df_metrics,
        winner_html,
        winner_summary['analysis_text'],
        chart_img,
        gan_img_path,
        diff_img_path,
        chart_path,
        report_file_path,
        updated_history,
        history_df
    )


# Build Gradio Blocks Application
def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="GenVision: GAN vs Diffusion Benchmark") as demo:

        # Session state for experiment history
        history_state = gr.State([])

        # Header Section
        gr.HTML("""
        <div class="main-header">
            <div class="main-title">GENVISION</div>
            <div style="font-size: 1.4rem; font-weight: 600; color: #00f0ff; margin-bottom: 4px;">
                GAN vs Diffusion-Based Image Generation & Quantitative Comparison
            </div>
            <div class="tagline">“One Prompt. Two Models. One Fair Comparison.”</div>
        </div>
        """)

        # Main Input Row
        with gr.Row():
            with gr.Column(scale=4):
                prompt_input = gr.Textbox(
                    label="Natural Language Prompt",
                    placeholder="Enter your prompt (e.g. A futuristic city at sunset with flying cars, neon lights, highly detailed)",
                    value="A futuristic city at sunset with flying cars",
                    lines=2
                )
            with gr.Column(scale=1):
                seed_input = gr.Number(label="Random Seed", value=42, precision=0)
                steps_input = gr.Slider(label="Diffusion Steps", minimum=1, maximum=10, value=2, step=1)

        generate_btn = gr.Button("⚡ Generate & Compare", elem_classes=["generate-btn"])

        gr.Markdown("---")

        # Side-by-Side Model Outputs
        gr.Markdown("### 🖼️ Side-by-Side Model Output Generation")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🔵 Model A: Text-Conditioned GAN")
                gan_output_img = gr.Image(label="GAN Generated Result", type="pil", interactive=False)
                gan_meta_info = gr.Markdown("Waiting for generation...")
                gan_download = gr.File(label="Download GAN Image")

            with gr.Column():
                gr.Markdown("#### 🟣 Model B: Stable Diffusion Turbo")
                diff_output_img = gr.Image(label="Diffusion Generated Result", type="pil", interactive=False)
                diff_meta_info = gr.Markdown("Waiting for generation...")
                diff_download = gr.File(label="Download Diffusion Image")

        gr.Markdown("---")

        # Quantitative Metrics & AI Analysis Section
        gr.Markdown("### 📊 Quantitative Performance Metrics")
        with gr.Row():
            with gr.Column(scale=3):
                metrics_table = gr.Dataframe(
                    headers=["Metric", "GAN Model", "Diffusion Model"],
                    datatype=["str", "str", "str"],
                    interactive=False
                )
            with gr.Column(scale=2):
                winner_html_output = gr.HTML("""
                <div style="padding: 15px; border: 1px dashed #00f0ff; border-radius: 12px; text-align: center; color: #94a3b8;">
                    Enter prompt and click 'Generate & Compare' to evaluate metrics.
                </div>
                """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🧠 AI Technical Analysis & Diagnostic Synthesis")
                ai_analysis_output = gr.Markdown("System awaiting generation execution...")

        gr.Markdown("---")

        # Visual Analytics & Dashboard Charts
        gr.Markdown("### 📈 Visual Analytics & Capability Radar")
        with gr.Row():
            chart_output_img = gr.Image(label="Performance & Radar Comparison Dashboard", type="pil", interactive=False)

        with gr.Row():
            chart_download = gr.File(label="Download Comparison Chart")
            report_download = gr.File(label="Download Full Benchmark Report (.txt)")

        gr.Markdown("---")

        # Visual Architecture Section
        gr.Markdown("### 🏗️ Visual Architecture Pipelines")
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="glass-card">
                    <h4 style="color: #00f0ff; margin-top: 0;">🔵 GAN Pipeline Architecture</h4>
                    <div class="arch-box">Prompt → CLIP Text Encoder → Semantic Latent Mapper (W-space) → DeepConv Generator → Image</div>
                    <p style="font-size: 0.88rem; color: #cbd5e1;">
                        Generative Adversarial Networks use a two-player game (Generator vs Discriminator). Text conditioning is achieved by projecting CLIP text embeddings into intermediate latent space <i>W</i> for single-step synthesis.
                    </p>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="glass-card">
                    <h4 style="color: #d946ef; margin-top: 0;">🟣 Diffusion Pipeline Architecture</h4>
                    <div class="arch-box">Prompt → Text Encoder → Initial Gaussian Noise → Latent Denoising U-Net → VAE Decoder → Image</div>
                    <p style="font-size: 0.88rem; color: #cbd5e1;">
                        Diffusion models reverse a thermodynamic noise degradation process step-by-step. Guided by text cross-attention layers inside a U-Net or Transformer, latent noise is iteratively converted into crisp RGB pixels.
                    </p>
                </div>
                """)

        gr.Markdown("---")

        # Experiment History Section
        gr.Markdown("### 📜 Session Experiment Tracker")
        history_table = gr.Dataframe(
            headers=["Timestamp", "Prompt", "GAN Time", "Diffusion Time", "Overall Winner"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False
        )

        # Wire Up Click Handlers
        generate_btn.click(
            fn=process_generation_and_comparison,
            inputs=[prompt_input, seed_input, steps_input, history_state],
            outputs=[
                gan_output_img,
                gan_meta_info,
                diff_output_img,
                diff_meta_info,
                metrics_table,
                winner_html_output,
                ai_analysis_output,
                chart_output_img,
                gan_download,
                diff_download,
                chart_download,
                report_download,
                history_state,
                history_table
            ]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # Support launch with public sharing or local server
    app.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
