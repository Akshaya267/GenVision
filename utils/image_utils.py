import os
import time
import json
import base64
from io import BytesIO
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_image(image: Image.Image, size=(512, 512)) -> Image.Image:
    """Resizes PIL image maintaining aspect ratio padding or direct resize."""
    return image.resize(size, Image.Resampling.LANCZOS)

def save_generated_image(image: Image.Image, prefix="gen") -> str:
    """Saves PIL Image to the outputs directory with a timestamped filename."""
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath, format="PNG")
    return filepath

def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image into a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_text_report(prompt: str, metrics: dict, winner_summary: dict) -> str:
    """Generates a formatted plain-text benchmark report."""
    report = f"""===================================================================
GENVISION: GAN VS DIFFUSION BENCHMARK REPORT
===================================================================
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Prompt: "{prompt}"

-------------------------------------------------------------------
1. GENERATION METRICS & COMPARISON TABLE
-------------------------------------------------------------------
Metric                     | GAN                      | Diffusion
-------------------------------------------------------------------
Model Architecture         | {metrics['GAN']['model_name']:<24} | {metrics['Diffusion']['model_name']:<24}
Generation Time (sec)      | {metrics['GAN']['generation_time']:<24.3f} | {metrics['Diffusion']['generation_time']:<24.3f}
CLIP Semantic Score        | {metrics['GAN']['clip_similarity']:<24.2f}% | {metrics['Diffusion']['clip_similarity']:<24.2f}%
Sharpness (Laplacian Var)  | {metrics['GAN']['sharpness']:<24.2f} | {metrics['Diffusion']['sharpness']:<24.2f}
Color Diversity Index      | {metrics['GAN']['color_diversity']:<24.2f} | {metrics['Diffusion']['color_diversity']:<24.2f}
Edge Density (Canny Ratio) | {metrics['GAN']['edge_density']:<24.4f} | {metrics['Diffusion']['edge_density']:<24.4f}
Output Resolution          | {metrics['GAN']['resolution']:<24} | {metrics['Diffusion']['resolution']:<24}

-------------------------------------------------------------------
2. CATEGORICAL BENCHMARK WINNERS
-------------------------------------------------------------------
🏆 Best Semantic Alignment : {winner_summary.get('clip_similarity', 'N/A')}
⚡ Fastest Generation Time   : {winner_summary.get('generation_time', 'N/A')}
🎨 Highest Color Diversity  : {winner_summary.get('color_diversity', 'N/A')}
🔍 Highest Image Sharpness  : {winner_summary.get('sharpness', 'N/A')}
👑 OVERALL WINNER           : {winner_summary.get('overall', 'N/A')}

-------------------------------------------------------------------
3. TECHNICAL AI ANALYSIS
-------------------------------------------------------------------
{winner_summary.get('analysis_text', 'No analysis generated.')}
===================================================================
"""
    report_path = os.path.join(OUTPUT_DIR, f"report_{int(time.time())}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path
