import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenVision.Visualizer")

def create_comparison_dashboard(metrics_dict: dict) -> Image.Image:
    """
    Generates a 4-panel professional dark-themed Matplotlib visual comparison dashboard.
    Panels:
    1. Performance Comparison (Bar Chart)
    2. Generation Time Benchmark (Bar Chart)
    3. Normalized Relative Metrics (0-100%)
    4. 5-Axis Spider Radar Chart (Semantic Alignment, Sharpness, Color Diversity, Structural Complexity, Speed Efficiency)
    """
    gan = metrics_dict["GAN"]
    diff = metrics_dict["Diffusion"]

    # Matplotlib Dark Navy Theme Setup
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10), facecolor='#0b1120')
    fig.suptitle("GENVISION: GAN vs Diffusion Quantitative Performance Benchmark", 
                 fontsize=16, fontweight='bold', color='#00f0ff', y=0.97)

    cyan_color = '#00f0ff'
    magenta_color = '#d946ef'

    # -------------------------------------------------------------
    # Subplot 1: Key Performance Metrics Bar Chart
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, facecolor='#11192e')
    categories = ['CLIP Score (%)', 'Sharpness', 'Color Diversity', 'Edge Density (x100)']
    
    gan_vals1 = [gan['clip_similarity'], gan['sharpness'], gan['color_diversity'], gan['edge_density'] * 100]
    diff_vals1 = [diff['clip_similarity'], diff['sharpness'], diff['color_diversity'], diff['edge_density'] * 100]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, gan_vals1, width, label='GAN', color=cyan_color, alpha=0.85, edgecolor='white', linewidth=0.5)
    rects2 = ax1.bar(x + width/2, diff_vals1, width, label='Diffusion', color=magenta_color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax1.set_title('Absolute Metric Values', color='white', fontsize=12, pad=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, color='#cbd5e1', fontsize=9)
    ax1.tick_params(axis='y', colors='#cbd5e1')
    ax1.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')

    # Add values on top of bars
    ax1.bar_label(rects1, fmt='%.1f', padding=3, color=cyan_color, fontsize=8)
    ax1.bar_label(rects2, fmt='%.1f', padding=3, color=magenta_color, fontsize=8)

    # -------------------------------------------------------------
    # Subplot 2: Generation Time Comparison (Lower is Better)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, facecolor='#11192e')
    models = ['GAN Generator', 'Diffusion Model']
    times = [gan['generation_time'], diff['generation_time']]
    bar_colors = [cyan_color, magenta_color]

    y_pos = np.arange(len(models))
    bars = ax2.barh(y_pos, times, height=0.45, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, color='#cbd5e1', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Generation Latency (seconds)', color='#cbd5e1', fontsize=10)
    ax2.set_title('Inference Latency (Lower is Better)', color='white', fontsize=12, pad=10, fontweight='bold')
    ax2.tick_params(axis='x', colors='#cbd5e1')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')

    for bar in bars:
        width_val = bar.get_width()
        ax2.text(width_val + max(times)*0.02, bar.get_y() + bar.get_height()/2, f'{width_val:.3f}s',
                 ha='left', va='center', color='white', fontweight='bold', fontsize=9)

    # -------------------------------------------------------------
    # Subplot 3: Min-Max Normalized Comparison (0 - 100%)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, facecolor='#11192e')
    norm_labels = ['Semantic', 'Speed', 'Sharpness', 'Color', 'Edge']
    
    # Calculate min-max normalization
    max_time = max(gan['generation_time'], diff['generation_time'], 1e-5)
    gan_speed_score = (1 - gan['generation_time'] / max_time) * 100
    diff_speed_score = (1 - diff['generation_time'] / max_time) * 100

    max_clip = max(gan['clip_similarity'], diff['clip_similarity'], 1e-5)
    max_sharp = max(gan['sharpness'], diff['sharpness'], 1e-5)
    max_color = max(gan['color_diversity'], diff['color_diversity'], 1e-5)
    max_edge = max(gan['edge_density'], diff['edge_density'], 1e-5)

    gan_norm = [
        (gan['clip_similarity'] / max_clip) * 100,
        gan_speed_score if gan_speed_score > 10 else 15,
        (gan['sharpness'] / max_sharp) * 100,
        (gan['color_diversity'] / max_color) * 100,
        (gan['edge_density'] / max_edge) * 100
    ]

    diff_norm = [
        (diff['clip_similarity'] / max_clip) * 100,
        diff_speed_score if diff_speed_score > 10 else 15,
        (diff['sharpness'] / max_sharp) * 100,
        (diff['color_diversity'] / max_color) * 100,
        (diff['edge_density'] / max_edge) * 100
    ]

    x3 = np.arange(len(norm_labels))
    ax3.bar(x3 - width/2, gan_norm, width, label='GAN', color=cyan_color, alpha=0.85)
    ax3.bar(x3 + width/2, diff_norm, width, label='Diffusion', color=magenta_color, alpha=0.85)
    ax3.set_title('Normalized Performance Scale (0 - 100%)', color='white', fontsize=12, pad=10, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(norm_labels, color='#cbd5e1', fontsize=9)
    ax3.tick_params(axis='y', colors='#cbd5e1')
    ax3.set_ylim(0, 115)
    ax3.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    ax3.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')

    # -------------------------------------------------------------
    # Subplot 4: 5-Axis Spider / Radar Chart
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, polar=True, facecolor='#11192e')
    radar_angles = np.linspace(0, 2 * np.pi, len(norm_labels), endpoint=False).tolist()
    radar_angles += radar_angles[:1]

    gan_radar = gan_norm + gan_norm[:1]
    diff_radar = diff_norm + diff_norm[:1]

    ax4.plot(radar_angles, gan_radar, color=cyan_color, linewidth=2, linestyle='solid', label='GAN')
    ax4.fill(radar_angles, gan_radar, color=cyan_color, alpha=0.25)

    ax4.plot(radar_angles, diff_radar, color=magenta_color, linewidth=2, linestyle='solid', label='Diffusion')
    ax4.fill(radar_angles, diff_radar, color=magenta_color, alpha=0.25)

    ax4.set_xticks(radar_angles[:-1])
    ax4.set_xticklabels(norm_labels, color='white', fontsize=9, fontweight='bold')
    ax4.set_title('5-Axis Architecture Capability Radar', color='white', fontsize=12, pad=15, fontweight='bold')
    ax4.tick_params(colors='#cbd5e1')
    ax4.grid(True, color='#94a3b8', linestyle='--', alpha=0.3)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), facecolor='#1e293b', edgecolor='none', labelcolor='white')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Convert plot to PIL Image
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba_buffer, "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return image
