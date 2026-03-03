#!/usr/bin/env python3
"""Generate a multimodal agent DEMO video.

This script runs 19 real multimodal test scenarios through the Pinocchio agent
and renders the results as an animated terminal-style demo video.

Scenario categories:
  1. Text processing (5): summarization, translation, code generation,
     math reasoning, creative writing
  2. Vision processing (7): chart analysis, scene VQA, OCR code extraction,
     diagram understanding, multi-image comparison, handwriting recognition,
     Chinese text OCR
  3. Multi-turn memory (3): user profiling, recall verification,
     context switching
  4. Edge cases (3): ultra-short input, ambiguous input, multi-language
  5. Cognitive loop (1): full 6-phase walkthrough with status report

Each scenario runs through the REAL Pinocchio agent with qwen3-vl:8b via Ollama.
Results are cached to JSON so you can re-render without re-running the LLM.

Usage:
  python3 scripts/generate_demo_video.py             # Full run + render
  python3 scripts/generate_demo_video.py --render-only  # Re-render from cache

Requires: Pillow, opencv-python-headless, Ollama running with qwen3-vl:8b
Produces: scripts/demo_video.mp4
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Video constants ───────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
FPS = 30

# Catppuccin Mocha palette
BG       = (30, 30, 46)
SURFACE  = (49, 50, 68)
OVERLAY  = (69, 71, 90)
TEXT_COL = (205, 214, 244)
GREEN    = (166, 227, 161)
RED      = (243, 139, 168)
YELLOW   = (249, 226, 175)
BLUE     = (137, 180, 250)
LAVENDER = (180, 190, 254)
MAUVE    = (203, 166, 247)
PEACH    = (250, 179, 135)
DIM      = (108, 112, 134)
TEAL     = (148, 226, 213)
SKY      = (137, 220, 235)
FLAMINGO = (242, 205, 205)

MARGIN   = 40
LINE_H   = 28
FONT_SZ  = 20
H1_SZ    = 44
H2_SZ    = 28
H3_SZ    = 22

OUTPUT_PATH = SCRIPTS / "demo_video.mp4"
ASSETS_DIR  = SCRIPTS / "demo_assets"
CACHE_PATH  = SCRIPTS / "demo_cache.json"

# ── Font helper ───────────────────────────────────────────────

_font_cache: dict[tuple[int, bool, bool], ImageFont.FreeTypeFont] = {}


def get_font(size: int = FONT_SZ, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono)
    if key in _font_cache:
        return _font_cache[key]

    cjk = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    cjk_bold = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    mono_list = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]

    candidates = mono_list if mono else (cjk_bold + cjk if bold else cjk)
    for p in candidates:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[key] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ── Drawing helpers ───────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_frame(bg=BG) -> tuple[Image.Image, ImageDraw.Draw]:
    img = Image.new("RGB", (WIDTH, HEIGHT), bg)
    return img, ImageDraw.Draw(img)


def wrap_text(text: str, max_chars: int = 95) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=max_chars) or [""])
    return lines


def alpha_color(color: tuple, a: float) -> tuple:
    return tuple(int(v * a) for v in color)


def draw_rounded_rect(draw, xy, fill, radius=12):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def draw_progress_bar(draw, x, y, w, h, pct, fg=GREEN, bg_col=OVERLAY):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg_col)
    bar_w = max(h, int(w * pct))
    draw.rounded_rectangle([x, y, x + bar_w, y + h], radius=h // 2, fill=fg)


# ======================================================================
# Test Asset Generation -- synthetic images for vision testing
# ======================================================================

def ensure_assets_dir():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def create_bar_chart() -> str:
    """Synthetic quarterly sales bar chart."""
    path = ASSETS_DIR / "sales_chart.png"
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    font_big = get_font(24, bold=True)

    draw.text((250, 20), "Quarterly Sales Report 2025", fill=(30, 30, 30), font=font_big)
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [150, 280, 210, 340]
    colors = [(66, 133, 244), (52, 168, 83), (251, 188, 4), (234, 67, 53)]
    bar_w, max_h, base_y = 100, 350, 530

    for i, (q, v, c) in enumerate(zip(quarters, values, colors)):
        x = 150 + i * 160
        h = int(v / max(values) * max_h)
        draw.rectangle([x, base_y - h, x + bar_w, base_y], fill=c)
        draw.text((x + 30, base_y - h - 25), f"${v}K", fill=(30, 30, 30), font=font)
        draw.text((x + 35, base_y + 10), q, fill=(80, 80, 80), font=font)

    draw.text((30, 250), "Revenue ($K)", fill=(80, 80, 80), font=font)
    img.save(path)
    return str(path)


def create_scene_image() -> str:
    """Synthetic scene with house, tree, sun for VQA."""
    path = ASSETS_DIR / "scene.png"
    img = Image.new("RGB", (800, 600), (135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    # Sun
    draw.ellipse([620, 40, 730, 150], fill=(255, 215, 0))
    # House
    draw.rectangle([200, 280, 400, 450], fill=(139, 69, 19))
    draw.polygon([(180, 280), (420, 280), (300, 180)], fill=(178, 34, 34))
    draw.rectangle([270, 340, 330, 450], fill=(101, 67, 33))
    draw.rectangle([220, 310, 260, 350], fill=(173, 216, 230))
    draw.rectangle([340, 310, 380, 350], fill=(173, 216, 230))
    # Tree
    draw.rectangle([550, 300, 575, 420], fill=(101, 67, 33))
    draw.ellipse([510, 200, 620, 320], fill=(0, 100, 0))
    # Label
    font = get_font(16)
    draw.text((20, 560), "Synthetic Scene for VQA Testing", fill=(255, 255, 255), font=font)
    img.save(path)
    return str(path)


def create_code_screenshot() -> str:
    """Screenshot of Python code for OCR testing."""
    path = ASSETS_DIR / "code_screenshot.png"
    img = Image.new("RGB", (800, 500), (40, 42, 54))
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    code_lines = [
        ("def fibonacci(n):", BLUE),
        ('    """Return the n-th Fibonacci number."""', GREEN),
        ("    if n <= 1:", TEXT_COL),
        ("        return n", TEXT_COL),
        ("    return fibonacci(n - 1) + fibonacci(n - 2)", TEXT_COL),
        ("", TEXT_COL),
        ("# Test", DIM),
        ("for i in range(10):", MAUVE),
        ('    print(f"F({i}) = {fibonacci(i)}")', YELLOW),
    ]
    y = 30
    for line, color in code_lines:
        draw.text((30, y), line, fill=color, font=font)
        y += 28
    img.save(path)
    return str(path)


def create_flowchart() -> str:
    """Synthetic flowchart/diagram for diagram understanding testing."""
    path = ASSETS_DIR / "flowchart.png"
    img = Image.new("RGB", (900, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    font_bold = get_font(18, bold=True)

    draw.text((300, 15), "Machine Learning Pipeline", fill=(30, 30, 30), font=font_bold)

    boxes = [
        (50, 100, 200, 160, "Data\nCollection", (66, 133, 244)),
        (250, 100, 400, 160, "Data\nCleaning", (52, 168, 83)),
        (450, 100, 600, 160, "Feature\nEngineering", (251, 188, 4)),
        (650, 100, 800, 160, "Model\nTraining", (234, 67, 53)),
        (250, 250, 400, 310, "Cross\nValidation", (156, 39, 176)),
        (450, 250, 600, 310, "Hyperparameter\nTuning", (0, 150, 136)),
        (350, 400, 500, 460, "Model\nDeployment", (255, 87, 34)),
        (350, 510, 500, 570, "Monitoring\n& Feedback", (121, 85, 72)),
    ]
    for x0, y0, x1, y1, text, color in boxes:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=color)
        lines = text.split("\n")
        for j, ln in enumerate(lines):
            bb = draw.textbbox((0, 0), ln, font=font)
            tw = bb[2] - bb[0]
            cx = x0 + (x1 - x0 - tw) // 2
            cy = y0 + 10 + j * 22
            draw.text((cx, cy), ln, fill=(255, 255, 255), font=font)

    arrows = [
        (200, 130, 250, 130), (400, 130, 450, 130), (600, 130, 650, 130),
        (725, 160, 725, 200), (725, 200, 400, 280),
        (400, 310, 450, 280), (525, 310, 425, 400),
        (425, 460, 425, 510),
    ]
    for x0, y0, x1, y1 in arrows:
        draw.line([(x0, y0), (x1, y1)], fill=(80, 80, 80), width=2)

    img.save(path)
    return str(path)


def create_handwriting_image() -> str:
    """Simulate handwritten math equations for OCR test."""
    path = ASSETS_DIR / "handwriting.png"
    img = Image.new("RGB", (800, 400), (255, 253, 240))
    draw = ImageDraw.Draw(img)
    font = get_font(28)
    font_title = get_font(20, bold=True)

    draw.text((30, 20), "Handwritten Math (simulated):", fill=(100, 100, 100), font=font_title)

    equations = [
        "E = mc\u00b2",
        "\u222b\u2080^\u221e e^(-x\u00b2) dx = \u221a\u03c0 / 2",
        "F = G\u00b7m\u2081\u00b7m\u2082 / r\u00b2",
        "\u2207 \u00d7 E = -\u2202B/\u2202t",
        "S = k_B \u00b7 ln(\u03a9)",
    ]
    y = 70
    for eq in equations:
        jx = hash(eq) % 10 - 5
        draw.text((50 + jx, y), eq, fill=(30, 30, 120), font=font)
        y += 60

    img.save(path)
    return str(path)


def create_comparison_images() -> tuple[str, str]:
    """Two similar images with differences for comparison testing."""
    import random
    rng = random.Random(42)

    # Image A: city day
    path_a = ASSETS_DIR / "city_day.png"
    img_a = Image.new("RGB", (600, 400), (135, 206, 250))
    da = ImageDraw.Draw(img_a)
    da.rectangle([50, 150, 120, 350], fill=(128, 128, 128))
    da.rectangle([140, 100, 220, 350], fill=(100, 100, 100))
    da.rectangle([240, 180, 310, 350], fill=(150, 150, 150))
    da.rectangle([330, 120, 420, 350], fill=(110, 110, 110))
    da.rectangle([440, 160, 530, 350], fill=(130, 130, 130))
    for bx in [60, 150, 250, 340, 450]:
        for wy in range(200, 340, 40):
            da.rectangle([bx + 5, wy, bx + 20, wy + 15], fill=(200, 230, 255))
    da.ellipse([480, 30, 560, 110], fill=(255, 215, 0))
    da.rectangle([0, 350, 600, 400], fill=(80, 80, 80))
    da.rectangle([80, 360, 150, 380], fill=(220, 50, 50))
    da.rectangle([250, 365, 320, 385], fill=(50, 50, 220))
    da.text((10, 375), "Day Scene", fill=(200, 200, 200), font=get_font(14))
    img_a.save(path_a)

    # Image B: city night
    path_b = ASSETS_DIR / "city_night.png"
    img_b = Image.new("RGB", (600, 400), (20, 20, 50))
    db = ImageDraw.Draw(img_b)
    db.rectangle([50, 150, 120, 350], fill=(60, 60, 60))
    db.rectangle([140, 100, 220, 350], fill=(50, 50, 50))
    db.rectangle([240, 180, 310, 350], fill=(70, 70, 70))
    db.rectangle([330, 120, 420, 350], fill=(55, 55, 55))
    db.rectangle([440, 160, 530, 350], fill=(65, 65, 65))
    for bx in [60, 150, 250, 340, 450]:
        for wy in range(200, 340, 40):
            lit = hash((bx, wy)) % 3 != 0
            color = (255, 220, 100) if lit else (40, 40, 40)
            db.rectangle([bx + 5, wy, bx + 20, wy + 15], fill=color)
    db.ellipse([480, 30, 540, 90], fill=(230, 230, 200))
    db.ellipse([490, 25, 545, 80], fill=(20, 20, 50))
    for _ in range(30):
        sx, sy = rng.randint(10, 590), rng.randint(10, 140)
        db.ellipse([sx, sy, sx + 3, sy + 3], fill=(255, 255, 200))
    db.rectangle([0, 350, 600, 400], fill=(30, 30, 30))
    db.text((10, 375), "Night Scene", fill=(150, 150, 150), font=get_font(14))
    img_b.save(path_b)

    return str(path_a), str(path_b)


def create_chinese_text_image() -> str:
    """Image with Chinese text for OCR testing."""
    path = ASSETS_DIR / "chinese_text.png"
    img = Image.new("RGB", (800, 300), (250, 245, 230))
    draw = ImageDraw.Draw(img)
    font = get_font(24)
    font_title = get_font(28, bold=True)

    draw.text((30, 20), "\U0001f3db\ufe0f \u4e2d\u6587\u53e4\u8bd7 \u2014 OCR \u6d4b\u8bd5", fill=(80, 60, 40), font=font_title)
    lines = [
        "\u6625\u7720\u4e0d\u89c9\u6653\uff0c",
        "\u5904\u5904\u95fb\u557c\u9e1f\u3002",
        "\u591c\u6765\u98ce\u96e8\u58f0\uff0c",
        "\u82b1\u843d\u77e5\u591a\u5c11\u3002",
        "                \u2014\u2014 \u5b5f\u6d69\u7136\u300a\u6625\u6653\u300b",
    ]
    y = 80
    for line in lines:
        draw.text((60, y), line, fill=(40, 30, 20), font=font)
        y += 40

    img.save(path)
    return str(path)


# ======================================================================
# Demo Scenario Definition
# ======================================================================

@dataclass
class DemoScene:
    title: str
    icon: str
    modality: str  # "text" | "vision" | "multi"
    category: str  # grouping label
    description: str
    user_input: str
    chat_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_response: str = ""
    elapsed: float = 0.0
    thumbnail: Image.Image | None = None


# ======================================================================
# Run Real Agent Interactions
# ======================================================================

def build_all_scenarios() -> list[DemoScene]:
    """Define all test scenarios (without running them yet)."""
    ensure_assets_dir()

    # Generate all image assets
    chart_path = create_bar_chart()
    scene_path = create_scene_image()
    code_path = create_code_screenshot()
    flow_path = create_flowchart()
    hand_path = create_handwriting_image()
    day_path, night_path = create_comparison_images()
    cn_text_path = create_chinese_text_image()

    scenes: list[DemoScene] = []

    # ------------------------------------------------------------------
    # Category 1: Text Processing
    # ------------------------------------------------------------------

    scenes.append(DemoScene(
        title="\u6587\u672c\u7406\u89e3 \u2014 \u81ea\u52a8\u6458\u8981",
        icon="\U0001f4dd",
        modality="text",
        category="\u6587\u672c\u5904\u7406",
        description="\u5c06\u957f\u7bc7\u4e2d\u6587\u6bb5\u843d\u7cbe\u70bc\u4e3a\u8981\u70b9\u6458\u8981",
        user_input=(
            "\u8bf7\u7528\u4e09\u4e2a\u8981\u70b9\u603b\u7ed3\u4ee5\u4e0b\u5185\u5bb9\uff1a\n"
            "\u4eba\u5de5\u667a\u80fd\uff08AI\uff09\u6b63\u5728\u52a0\u901f\u6539\u53d8\u5168\u7403\u5404\u884c\u5404\u4e1a\u3002\u5728\u533b\u7597\u9886\u57df\uff0cAI\u8f85\u52a9\u8bca\u65ad\u7cfb\u7edf\u80fd\u591f\u5728\u51e0\u79d2\u949f\u5185\u5206\u6790"
            "\u533b\u5b66\u5f71\u50cf\uff0c\u51c6\u786e\u7387\u5df2\u7ecf\u8d85\u8d8a\u4eba\u7c7b\u653e\u5c04\u79d1\u533b\u751f\u3002\u5728\u81ea\u52a8\u9a7e\u9a76\u9886\u57df\uff0c\u7279\u65af\u62c9\u3001Waymo\u7b49\u516c\u53f8\u7684\u65e0\u4eba\u9a7e\u9a76"
            "\u6c7d\u8f66\u5df2\u7ecf\u5728\u591a\u4e2a\u57ce\u5e02\u4e0a\u8def\u6d4b\u8bd5\u3002\u5728\u6559\u80b2\u9886\u57df\uff0c\u4e2a\u6027\u5316AI\u5bfc\u5e08\u53ef\u4ee5\u6839\u636e\u6bcf\u4e2a\u5b66\u751f\u7684\u5b66\u4e60\u8fdb\u5ea6\u548c\u98ce\u683c"
            "\u8c03\u6574\u6559\u5b66\u5185\u5bb9\u3002\u7136\u800c\uff0cAI\u7684\u5feb\u901f\u53d1\u5c55\u4e5f\u5f15\u53d1\u4e86\u5173\u4e8e\u5c31\u4e1a\u3001\u9690\u79c1\u548c\u4f26\u7406\u7684\u5e7f\u6cdb\u8ba8\u8bba\u3002\u8bb8\u591a\u4e13\u5bb6\u547c\u5401"
            "\u5efa\u7acb\u66f4\u5b8c\u5584\u7684AI\u76d1\u7ba1\u6846\u67b6\uff0c\u4ee5\u786e\u4fdd\u6280\u672f\u7684\u53d1\u5c55\u80fd\u591f\u9020\u798f\u5168\u4eba\u7c7b\u3002"
        ),
    ))

    scenes.append(DemoScene(
        title="\u6587\u672c\u5904\u7406 \u2014 \u82f1\u4e2d\u7ffb\u8bd1",
        icon="\U0001f310",
        modality="text",
        category="\u6587\u672c\u5904\u7406",
        description="\u82f1\u8bd1\u4e2d\uff0c\u4fdd\u6301\u8bed\u8a00\u98ce\u683c\u548c\u4fee\u8f9e",
        user_input=(
            "\u8bf7\u5c06\u4ee5\u4e0b\u82f1\u6587\u7ffb\u8bd1\u6210\u4e2d\u6587\uff0c\u4fdd\u6301\u8bed\u8a00\u6d41\u7545\uff1a\n"
            "The quick brown fox jumps over the lazy dog. This sentence is a pangram \u2014 "
            "it contains every letter of the English alphabet at least once. "
            "Pangrams are commonly used in typography and font design to demonstrate "
            "the visual appearance of all glyphs in a typeface."
        ),
    ))

    scenes.append(DemoScene(
        title="\u4ee3\u7801\u751f\u6210 \u2014 \u4e8c\u5206\u67e5\u627e",
        icon="\U0001f4bb",
        modality="text",
        category="\u6587\u672c\u5904\u7406",
        description="\u6839\u636e\u81ea\u7136\u8bed\u8a00\u63cf\u8ff0\u751f\u6210\u5e26\u6ce8\u89e3\u7684Python\u4ee3\u7801",
        user_input="\u5199\u4e00\u4e2aPython\u51fd\u6570\uff0c\u5b9e\u73b0\u4e8c\u5206\u67e5\u627e\u7b97\u6cd5\uff0c\u8981\u6c42\u6709\u5b8c\u6574\u7684docstring\u548c\u7c7b\u578b\u6ce8\u89e3\u3002\u5305\u542b\u4f7f\u7528\u793a\u4f8b\u3002",
    ))

    scenes.append(DemoScene(
        title="\u6570\u5b66\u63a8\u7406 \u2014 \u5e94\u7528\u9898",
        icon="\U0001f9ee",
        modality="text",
        category="\u6587\u672c\u5904\u7406",
        description="\u89e3\u51b3\u9700\u8981\u591a\u6b65\u63a8\u7406\u7684\u6570\u5b66\u95ee\u9898",
        user_input=(
            "\u4e00\u4e2a\u6c34\u6c60\u6709\u4e24\u4e2a\u8fdb\u6c34\u7ba1\u548c\u4e00\u4e2a\u6392\u6c34\u7ba1\u3002A\u7ba1\u5355\u72ec\u4f7f\u75286\u5c0f\u65f6\u53ef\u6ce8\u6ee1\u6c34\u6c60\uff0c"
            "B\u7ba1\u5355\u72ec\u4f7f\u75288\u5c0f\u65f6\u53ef\u6ce8\u6ee1\u6c34\u6c60\uff0c\u6392\u6c34\u7ba1\u5355\u72ec\u4f7f\u752812\u5c0f\u65f6\u53ef\u6392\u7a7a\u6c34\u6c60\u3002"
            "\u5982\u679c\u4e09\u4e2a\u7ba1\u540c\u65f6\u5f00\u542f\uff0c\u591a\u5c11\u5c0f\u65f6\u53ef\u4ee5\u6ce8\u6ee1\u6c34\u6c60\uff1f\u8bf7\u5217\u51fa\u5b8c\u6574\u63a8\u7406\u8fc7\u7a0b\u3002"
        ),
    ))

    scenes.append(DemoScene(
        title="\u521b\u610f\u5199\u4f5c \u2014 \u79d1\u5e7b\u5fae\u5c0f\u8bf4",
        icon="\u270d\ufe0f",
        modality="text",
        category="\u6587\u672c\u5904\u7406",
        description="\u751f\u6210\u5bcc\u6709\u521b\u610f\u7684\u77ed\u7bc7\u79d1\u5e7b\u6545\u4e8b",
        user_input="\u8bf7\u5199\u4e00\u7bc7200\u5b57\u5de6\u53f3\u7684\u79d1\u5e7b\u5fae\u5c0f\u8bf4\uff0c\u4e3b\u9898\u662f'\u6700\u540e\u4e00\u4e2a\u4eba\u7c7b\u548cAI\u5171\u540c\u5b88\u62a4\u5730\u7403\u79cd\u5b50\u5e93'\u3002",
    ))

    # ------------------------------------------------------------------
    # Category 2: Vision Processing
    # ------------------------------------------------------------------

    s = DemoScene(
        title="\u56fe\u50cf\u7406\u89e3 \u2014 \u56fe\u8868\u5206\u6790",
        icon="\U0001f4ca",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u5206\u6790\u67f1\u72b6\u56fe\u5e76\u63d0\u53d6\u6570\u636e\u4e0e\u8d8b\u52bf",
        user_input="\u8bf7\u5206\u6790\u8fd9\u5f20\u9500\u552e\u62a5\u8868\u56fe\u8868\uff0c\u63cf\u8ff0\u6570\u636e\u8d8b\u52bf\u548c\u5173\u952e\u53d1\u73b0\u3002\u54ea\u4e2a\u5b63\u5ea6\u8868\u73b0\u6700\u597d\uff1f\u589e\u957f\u7387\u5982\u4f55\uff1f",
        chat_kwargs={"image_paths": [chart_path]},
    )
    s.thumbnail = Image.open(chart_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="\u56fe\u50cf\u7406\u89e3 \u2014 \u573a\u666f\u95ee\u7b54",
        icon="\U0001f3e0",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u56de\u7b54\u5173\u4e8e\u573a\u666f\u56fe\u50cf\u7684\u81ea\u7136\u8bed\u8a00\u95ee\u9898",
        user_input="\u8fd9\u5f20\u56fe\u7247\u91cc\u6709\u4ec0\u4e48\uff1f\u8bf7\u8be6\u7ec6\u63cf\u8ff0\u573a\u666f\u4e2d\u7684\u6bcf\u4e2a\u5143\u7d20\uff0c\u5305\u62ec\u5b83\u4eec\u7684\u4f4d\u7f6e\u5173\u7cfb\u3002",
        chat_kwargs={"image_paths": [scene_path]},
    )
    s.thumbnail = Image.open(scene_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="OCR\u8bc6\u522b \u2014 \u4ee3\u7801\u622a\u56fe",
        icon="\U0001f50d",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u4ece\u4ee3\u7801\u622a\u56fe\u4e2d\u63d0\u53d6\u6587\u5b57\u5e76\u89e3\u91ca\u4ee3\u7801\u903b\u8f91",
        user_input="\u8bf7\u8bc6\u522b\u8fd9\u5f20\u622a\u56fe\u4e2d\u7684\u4ee3\u7801\uff0c\u8bf4\u660e\u8fd9\u662f\u4ec0\u4e48\u7f16\u7a0b\u8bed\u8a00\uff0c\u5e76\u8be6\u7ec6\u89e3\u91ca\u5b83\u7684\u529f\u80fd\u548c\u7b97\u6cd5\u601d\u8def\u3002",
        chat_kwargs={"image_paths": [code_path]},
    )
    s.thumbnail = Image.open(code_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="\u56fe\u8868\u7406\u89e3 \u2014 \u6d41\u7a0b\u56fe",
        icon="\U0001f504",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u89e3\u8bfb\u6d41\u7a0b\u56fe/\u67b6\u6784\u56fe\u7684\u7ed3\u6784\u4e0e\u542b\u4e49",
        user_input="\u8bf7\u63cf\u8ff0\u8fd9\u5f20\u6d41\u7a0b\u56fe\u5c55\u793a\u7684\u662f\u4ec0\u4e48\u6d41\u7a0b\uff1f\u5404\u4e2a\u6b65\u9aa4\u4e4b\u95f4\u662f\u5982\u4f55\u8fde\u63a5\u7684\uff1f\u6709\u4ec0\u4e48\u6539\u8fdb\u5efa\u8bae\uff1f",
        chat_kwargs={"image_paths": [flow_path]},
    )
    s.thumbnail = Image.open(flow_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="OCR\u8bc6\u522b \u2014 \u624b\u5199\u516c\u5f0f",
        icon="\u270f\ufe0f",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u8bc6\u522b\u624b\u5199\u6570\u5b66\u516c\u5f0f\u5e76\u89e3\u91ca\u542b\u4e49",
        user_input="\u8bf7\u8bc6\u522b\u56fe\u4e2d\u7684\u6240\u6709\u6570\u5b66\u516c\u5f0f\uff0c\u5e76\u5206\u522b\u89e3\u91ca\u6bcf\u4e2a\u516c\u5f0f\u4ee3\u8868\u7684\u7269\u7406/\u6570\u5b66\u542b\u4e49\u3002",
        chat_kwargs={"image_paths": [hand_path]},
    )
    s.thumbnail = Image.open(hand_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="\u56fe\u50cf\u5bf9\u6bd4 \u2014 \u65e5\u591c\u53d8\u5316",
        icon="\U0001f317",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u5bf9\u6bd4\u4e24\u5f20\u56fe\u7247\u627e\u51fa\u5dee\u5f02",
        user_input="\u8bf7\u5bf9\u6bd4\u8fd9\u4e24\u5f20\u57ce\u5e02\u573a\u666f\u56fe\u7247\uff0c\u8be6\u7ec6\u63cf\u8ff0\u5b83\u4eec\u4e4b\u95f4\u7684\u5dee\u5f02\u548c\u76f8\u4f3c\u4e4b\u5904\u3002",
        chat_kwargs={"image_paths": [day_path, night_path]},
    )
    s.thumbnail = Image.open(day_path).copy()
    scenes.append(s)

    s = DemoScene(
        title="\u4e2d\u6587OCR \u2014 \u53e4\u8bd7\u8bc6\u522b",
        icon="\U0001f3db\ufe0f",
        modality="vision",
        category="\u56fe\u50cf\u7406\u89e3",
        description="\u8bc6\u522b\u56fe\u7247\u4e2d\u7684\u4e2d\u6587\u6587\u5b57",
        user_input="\u8bf7\u8bc6\u522b\u8fd9\u5f20\u56fe\u7247\u4e2d\u7684\u6587\u5b57\u5185\u5bb9\u3002\u8fd9\u662f\u4e00\u9996\u4ec0\u4e48\u8bd7\uff1f\u8bf7\u89e3\u91ca\u8bd7\u7684\u542b\u4e49\u3002",
        chat_kwargs={"image_paths": [cn_text_path]},
    )
    s.thumbnail = Image.open(cn_text_path).copy()
    scenes.append(s)

    # ------------------------------------------------------------------
    # Category 3: Multi-turn Memory
    # ------------------------------------------------------------------

    scenes.append(DemoScene(
        title="\u591a\u8f6e\u5bf9\u8bdd \u2014 \u7528\u6237\u5efa\u6a21",
        icon="\U0001f9e0",
        modality="text",
        category="\u591a\u8f6e\u5bf9\u8bdd",
        description="\u6d4b\u8bd5\u667a\u80fd\u4f53\u8de8\u8f6e\u6b21\u8bb0\u5fc6\u4e0e\u7528\u6237\u5efa\u6a21",
        user_input="\u4f60\u597d\uff01\u6211\u53eb\u5c0f\u660e\uff0c\u662f\u4e00\u540d\u8ba1\u7b97\u673a\u79d1\u5b66\u535a\u58eb\u751f\uff0c\u7814\u7a76\u65b9\u5411\u662f\u591a\u6a21\u6001\u5927\u8bed\u8a00\u6a21\u578b\u3002\u6211\u5bf9\u5f3a\u5316\u5b66\u4e60\u4e5f\u5f88\u611f\u5174\u8da3\u3002",
    ))

    scenes.append(DemoScene(
        title="\u591a\u8f6e\u5bf9\u8bdd \u2014 \u8bb0\u5fc6\u56de\u5fc6",
        icon="\U0001f9e0",
        modality="text",
        category="\u591a\u8f6e\u5bf9\u8bdd",
        description="\u9a8c\u8bc1\u667a\u80fd\u4f53\u80fd\u5426\u51c6\u786e\u56de\u5fc6\u7528\u6237\u4fe1\u606f",
        user_input="\u8bf7\u95ee\u6211\u53eb\u4ec0\u4e48\u540d\u5b57\uff1f\u6211\u7684\u7814\u7a76\u65b9\u5411\u662f\u4ec0\u4e48\uff1f\u6211\u8fd8\u5bf9\u4ec0\u4e48\u9886\u57df\u611f\u5174\u8da3\uff1f",
    ))

    scenes.append(DemoScene(
        title="\u591a\u8f6e\u5bf9\u8bdd \u2014 \u4e0a\u4e0b\u6587\u7406\u89e3",
        icon="\U0001f4ac",
        modality="text",
        category="\u591a\u8f6e\u5bf9\u8bdd",
        description="\u57fa\u4e8e\u4e4b\u524d\u7684\u5bf9\u8bdd\u4e0a\u4e0b\u6587\u7ed9\u51fa\u4e13\u4e1a\u5efa\u8bae",
        user_input="\u6839\u636e\u6211\u7684\u80cc\u666f\uff0c\u4f60\u80fd\u63a8\u8350\u4e09\u7bc7\u6700\u8fd1\u7684\u76f8\u5173\u8bba\u6587\u65b9\u5411\u5417\uff1f",
    ))

    # ------------------------------------------------------------------
    # Category 4: Edge Cases
    # ------------------------------------------------------------------

    scenes.append(DemoScene(
        title="\u8fb9\u754c\u6d4b\u8bd5 \u2014 \u6781\u77ed\u8f93\u5165",
        icon="\u26a1",
        modality="text",
        category="\u8fb9\u754c\u573a\u666f",
        description="\u6d4b\u8bd5\u6781\u77ed\u8f93\u5165\u7684\u5904\u7406\u80fd\u529b",
        user_input="\u4f60\u597d",
    ))

    scenes.append(DemoScene(
        title="\u8fb9\u754c\u6d4b\u8bd5 \u2014 \u6a21\u7cca\u6307\u4ee4",
        icon="\U0001f914",
        modality="text",
        category="\u8fb9\u754c\u573a\u666f",
        description="\u6d4b\u8bd5\u5bf9\u6a21\u7cca/\u4e0d\u5b8c\u6574\u6307\u4ee4\u7684\u7406\u89e3\u80fd\u529b",
        user_input="\u90a3\u4e2a\u4e1c\u897f\uff0c\u5c31\u662f\u4e0a\u6b21\u8bf4\u7684\u90a3\u4e2a\uff0c\u5e2e\u6211\u5f04\u4e00\u4e0b",
    ))

    scenes.append(DemoScene(
        title="\u8fb9\u754c\u6d4b\u8bd5 \u2014 \u591a\u8bed\u6df7\u5408",
        icon="\U0001f30d",
        modality="text",
        category="\u8fb9\u754c\u573a\u666f",
        description="\u6d4b\u8bd5\u4e2d\u82f1\u65e5\u6df7\u5408\u6587\u672c\u5904\u7406",
        user_input="Please\u7528\u4e2d\u6587explain\u4e00\u4e0b\u4ec0\u4e48\u662fTransformer architecture\uff0c\u7136\u540egive me\u4e00\u4e2a\u7b80\u5355\u7684\u4f8b\u5b50\u3002\u3042\u308a\u304c\u3068\u3046\u3002",
    ))

    # ------------------------------------------------------------------
    # Category 5: Cognitive Loop
    # ------------------------------------------------------------------

    scenes.append(DemoScene(
        title="\u8ba4\u77e5\u5faa\u73af \u2014 \u72b6\u6001\u62a5\u544a",
        icon="\U0001faa9",
        modality="text",
        category="\u8ba4\u77e5\u5faa\u73af",
        description="\u5c55\u793a\u5b8c\u6574\u7684\u8ba4\u77e5\u5faa\u73af\u72b6\u6001\u4e0e\u81ea\u6211\u8bc4\u4f30",
        user_input="__STATUS__",
    ))

    return scenes


def run_all_scenarios(scenes: list[DemoScene]) -> list[DemoScene]:
    """Run each scenario through the real Pinocchio agent."""
    from pinocchio import Pinocchio

    MODEL = "qwen3-vl:8b"
    print(f"\n\U0001f916 Initialising Pinocchio with model: {MODEL}")
    agent = Pinocchio(
        model=MODEL,
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        data_dir=str(SCRIPTS / "demo_data"),
        verbose=True,
    )

    for i, scene in enumerate(scenes, 1):
        sep = '\u2500' * 50
        print(f"\n{sep}")
        print(f"  {scene.icon}  [{i}/{len(scenes)}] {scene.title}")
        print(f"{sep}")

        t0 = time.time()
        try:
            if scene.user_input == "__STATUS__":
                status = agent.status()
                scene.agent_response = json.dumps(status, ensure_ascii=False, indent=2)
            else:
                scene.agent_response = agent.chat(scene.user_input, **scene.chat_kwargs)
        except Exception as e:
            scene.agent_response = f"[Error: {e}]"
            traceback.print_exc()
        scene.elapsed = time.time() - t0
        print(f"  \u2705 Done in {scene.elapsed:.1f}s")
        preview = scene.agent_response[:120].replace("\n", " ")
        print(f"  \U0001f4c4 {preview}{'...' if len(scene.agent_response) > 120 else ''}")

    return scenes


# ── Cache helpers ─────────────────────────────────────────────

def save_cache(scenes: list[DemoScene]) -> None:
    data = []
    for s in scenes:
        data.append({
            "title": s.title, "icon": s.icon, "modality": s.modality,
            "category": s.category, "description": s.description,
            "user_input": s.user_input,
            "chat_kwargs_keys": list(s.chat_kwargs.keys()),
            "agent_response": s.agent_response,
            "elapsed": s.elapsed,
            "has_thumbnail": s.thumbnail is not None,
        })
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\U0001f4be Cache saved to {CACHE_PATH}")


def load_cache() -> list[DemoScene] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        # Regenerate assets for thumbnails
        ensure_assets_dir()
        chart_path = create_bar_chart()
        scene_path = create_scene_image()
        code_path = create_code_screenshot()
        flow_path = create_flowchart()
        hand_path = create_handwriting_image()
        day_path, night_path = create_comparison_images()
        cn_text_path = create_chinese_text_image()

        thumb_map = {
            "\u56fe\u50cf\u7406\u89e3 \u2014 \u56fe\u8868\u5206\u6790": chart_path,
            "\u56fe\u50cf\u7406\u89e3 \u2014 \u573a\u666f\u95ee\u7b54": scene_path,
            "OCR\u8bc6\u522b \u2014 \u4ee3\u7801\u622a\u56fe": code_path,
            "\u56fe\u8868\u7406\u89e3 \u2014 \u6d41\u7a0b\u56fe": flow_path,
            "OCR\u8bc6\u522b \u2014 \u624b\u5199\u516c\u5f0f": hand_path,
            "\u56fe\u50cf\u5bf9\u6bd4 \u2014 \u65e5\u591c\u53d8\u5316": day_path,
            "\u4e2d\u6587OCR \u2014 \u53e4\u8bd7\u8bc6\u522b": cn_text_path,
        }
        kwargs_map = {
            "\u56fe\u50cf\u7406\u89e3 \u2014 \u56fe\u8868\u5206\u6790": {"image_paths": [chart_path]},
            "\u56fe\u50cf\u7406\u89e3 \u2014 \u573a\u666f\u95ee\u7b54": {"image_paths": [scene_path]},
            "OCR\u8bc6\u522b \u2014 \u4ee3\u7801\u622a\u56fe": {"image_paths": [code_path]},
            "\u56fe\u8868\u7406\u89e3 \u2014 \u6d41\u7a0b\u56fe": {"image_paths": [flow_path]},
            "OCR\u8bc6\u522b \u2014 \u624b\u5199\u516c\u5f0f": {"image_paths": [hand_path]},
            "\u56fe\u50cf\u5bf9\u6bd4 \u2014 \u65e5\u591c\u53d8\u5316": {"image_paths": [day_path, night_path]},
            "\u4e2d\u6587OCR \u2014 \u53e4\u8bd7\u8bc6\u522b": {"image_paths": [cn_text_path]},
        }

        scenes: list[DemoScene] = []
        for d in data:
            s = DemoScene(
                title=d["title"], icon=d["icon"], modality=d["modality"],
                category=d["category"], description=d["description"],
                user_input=d["user_input"],
                chat_kwargs=kwargs_map.get(d["title"], {}),
                agent_response=d["agent_response"],
                elapsed=d["elapsed"],
            )
            if d.get("has_thumbnail") and s.title in thumb_map:
                s.thumbnail = Image.open(thumb_map[s.title]).copy()
            scenes.append(s)
        print(f"\U0001f4e6 Loaded {len(scenes)} scenes from cache")
        return scenes
    except Exception as e:
        print(f"\u26a0\ufe0f  Cache load failed: {e}")
        return None


# ======================================================================
# Video Renderer
# ======================================================================

class VideoRenderer:
    """Render real multimodal test scenarios into an animated MP4 video."""

    def __init__(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
        assert self.writer.isOpened(), f"Failed to open VideoWriter at {OUTPUT_PATH}"
        self.frame_count = 0

    def close(self):
        self.writer.release()

    def _write(self, img: Image.Image):
        self.writer.write(pil_to_cv(img))
        self.frame_count += 1

    def _hold(self, img: Image.Image, sec: float):
        for _ in range(int(FPS * sec)):
            self._write(img)

    # ── Title Screen ─────────────────────────────────────────

    def render_title(self, scenes: list[DemoScene], duration: float = 5.0):
        fh1 = get_font(H1_SZ, bold=True)
        fh2 = get_font(H2_SZ)
        fs = get_font(FONT_SZ)

        n = len(scenes)
        n_text = sum(1 for s in scenes if s.modality == "text")
        n_vision = sum(1 for s in scenes if s.modality == "vision")
        cats = sorted(set(s.category for s in scenes))

        for i in range(int(FPS * duration)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.8))

            draw.rectangle([0, 0, 6, HEIGHT], fill=alpha_color(BLUE, a))

            draw.text((MARGIN, 60), "\U0001f916 Pinocchio \u591a\u6a21\u6001\u667a\u80fd\u4f53\u6d4b\u8bd5\u6f14\u793a",
                       fill=alpha_color(BLUE, a), font=fh1)
            draw.text((MARGIN, 120), "\u8986\u76d6\u6587\u672c\u00b7\u56fe\u50cf\u00b7\u591a\u8f6e\u5bf9\u8bdd\u00b7\u8fb9\u754c\u573a\u666f \u2014 \u771f\u5b9e LLM \u4ea4\u4e92",
                       fill=alpha_color(TEXT_COL, a), font=fh2)

            y = 200
            stats = [
                f"\U0001f4cb  {n} \u4e2a\u6d4b\u8bd5\u573a\u666f",
                f"\U0001f4dd  {n_text} \u4e2a\u6587\u672c\u573a\u666f   \U0001f5bc\ufe0f  {n_vision} \u4e2a\u89c6\u89c9\u573a\u666f",
                f"\U0001f4c2  {len(cats)} \u4e2a\u6d4b\u8bd5\u7c7b\u522b: {', '.join(cats)}",
            ]
            for line in stats:
                draw.text((MARGIN + 20, y), line,
                           fill=alpha_color(LAVENDER, a), font=fh2)
                y += 42

            y += 20
            cat_colors = {
                "\u6587\u672c\u5904\u7406": TEAL,
                "\u56fe\u50cf\u7406\u89e3": PEACH,
                "\u591a\u8f6e\u5bf9\u8bdd": LAVENDER,
                "\u8fb9\u754c\u573a\u666f": YELLOW,
                "\u8ba4\u77e5\u5faa\u73af": MAUVE,
            }
            cat_icons = {
                "\u6587\u672c\u5904\u7406": "\U0001f4dd",
                "\u56fe\u50cf\u7406\u89e3": "\U0001f5bc\ufe0f",
                "\u591a\u8f6e\u5bf9\u8bdd": "\U0001f9e0",
                "\u8fb9\u754c\u573a\u666f": "\u26a1",
                "\u8ba4\u77e5\u5faa\u73af": "\U0001faa9",
            }
            for cat in cats:
                cnt = sum(1 for s in scenes if s.category == cat)
                icon = cat_icons.get(cat, "\u2022")
                color = cat_colors.get(cat, TEXT_COL)
                draw.text((MARGIN + 40, y), f"{icon}  {cat}  \u00d7{cnt}",
                           fill=alpha_color(color, a), font=fs)
                y += 32

            y += 30
            features = [
                "\u2705  \u516d\u9636\u6bb5\u8ba4\u77e5\u5faa\u73af: \u611f\u77e5 \u2192 \u7b56\u7565 \u2192 \u6267\u884c \u2192 \u8bc4\u4f30 \u2192 \u5b66\u4e60 \u2192 \u5143\u53cd\u601d",
                "\U0001f6e1\ufe0f  \u9c81\u68d2\u6027\u52a0\u56fa: \u679a\u4e3e\u5b89\u5168 \u00b7 \u5206\u6570\u9497\u4f4d \u00b7 \u7ebf\u7a0b\u5b89\u5168 \u00b7 \u9519\u8bef\u9694\u79bb",
                "\u26a1  Powered by Ollama + qwen3-vl:8b",
            ]
            for feat in features:
                draw.text((MARGIN + 20, y), feat,
                           fill=alpha_color(DIM, a), font=fs)
                y += 30

            foot = "Pinocchio Multimodal Agent  \u00b7  2026-03-02  \u00b7  Live LLM Demo"
            fbx = draw.textbbox((0, 0), foot, font=fs)
            draw.text(((WIDTH - fbx[2] + fbx[0]) // 2, HEIGHT - 50),
                       foot, fill=alpha_color(DIM, a), font=fs)

            self._write(img)

    # ── Category Transition ──────────────────────────────────

    def render_category_transition(self, category: str, cat_scenes: list[DemoScene]):
        fh1 = get_font(H1_SZ, bold=True)
        fs = get_font(FONT_SZ)

        cat_icons = {
            "\u6587\u672c\u5904\u7406": "\U0001f4dd",
            "\u56fe\u50cf\u7406\u89e3": "\U0001f5bc\ufe0f",
            "\u591a\u8f6e\u5bf9\u8bdd": "\U0001f9e0",
            "\u8fb9\u754c\u573a\u666f": "\u26a1",
            "\u8ba4\u77e5\u5faa\u73af": "\U0001faa9",
        }
        cat_colors = {
            "\u6587\u672c\u5904\u7406": TEAL,
            "\u56fe\u50cf\u7406\u89e3": PEACH,
            "\u591a\u8f6e\u5bf9\u8bdd": LAVENDER,
            "\u8fb9\u754c\u573a\u666f": YELLOW,
            "\u8ba4\u77e5\u5faa\u73af": MAUVE,
        }
        icon = cat_icons.get(category, "\u2022")
        color = cat_colors.get(category, BLUE)

        for i in range(int(FPS * 2.5)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.4))

            title = f"{icon}  {category}"
            tbx = draw.textbbox((0, 0), title, font=fh1)
            tw = tbx[2] - tbx[0]
            draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 80), title,
                       fill=alpha_color(color, a), font=fh1)

            lw = int(400 * min(1.0, i / (FPS * 0.8)))
            cx = WIDTH // 2
            draw.line([(cx - lw // 2, HEIGHT // 2 - 30), (cx + lw // 2, HEIGHT // 2 - 30)],
                       fill=alpha_color(color, a * 0.5), width=2)

            y = HEIGHT // 2 + 10
            for j, s in enumerate(cat_scenes):
                sa = min(1.0, max(0, (i - FPS * 0.5 - j * 4) / (FPS * 0.3)))
                if sa <= 0:
                    continue
                draw.text((WIDTH // 2 - 200, y), f"{s.icon}  {s.title}",
                           fill=alpha_color(TEXT_COL, sa * a), font=fs)
                y += 30

            self._write(img)

    # ── Scene Transition Card ────────────────────────────────

    def render_scene_transition(self, scene: DemoScene, num: int, total: int):
        fh2 = get_font(H2_SZ, bold=True)
        fs = get_font(FONT_SZ)

        badge_colors = {"text": TEAL, "vision": PEACH, "multi": YELLOW}
        bc = badge_colors.get(scene.modality, TEXT_COL)

        for i in range(int(FPS * 1.5)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.4))

            draw.rectangle([0, 0, 8, HEIGHT], fill=alpha_color(BLUE, a))

            draw.text((MARGIN, 40), f"Scene {num}/{total}",
                       fill=alpha_color(DIM, a), font=fs)

            draw.text((MARGIN, HEIGHT // 2 - 60),
                       f"{scene.icon}  {scene.title}",
                       fill=alpha_color(BLUE, a), font=fh2)

            draw.text((MARGIN, HEIGHT // 2), scene.description,
                       fill=alpha_color(TEXT_COL, a), font=fs)

            badge = f"[{scene.modality.upper()}]"
            draw.text((MARGIN, HEIGHT // 2 + 40), badge,
                       fill=alpha_color(bc, a), font=fs)

            self._write(img)

    # ── Main Scene Rendering (typed chat) ────────────────────

    def render_scene(self, scene: DemoScene, scene_num: int):
        font = get_font(FONT_SZ)
        font_bold = get_font(FONT_SZ, bold=True)
        font_sm = get_font(16)

        header_h = 50
        chat_top = header_h + 10

        user_lines = wrap_text(
            scene.user_input if scene.user_input != "__STATUS__" else "(agent.status() \u547d\u4ee4)",
            max_chars=100,
        )
        resp_lines = wrap_text(scene.agent_response, max_chars=100)
        if len(resp_lines) > 32:
            resp_lines = resp_lines[:30] + ["...", f"(\u5171 {len(wrap_text(scene.agent_response, 100))} \u884c)"]

        # === Phase 1: Type user message ===
        total_uchars = sum(len(l) for l in user_lines)
        uchars_per_frame = max(3, int(total_uchars // (FPS * 1.5)))

        typed = 0
        while typed <= total_uchars:
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_sm)

            thumb_offset = 0
            if scene.thumbnail:
                thumb_offset = self._draw_thumbnail(img, draw, scene.thumbnail, chat_top + 10)

            y = chat_top + 10 + thumb_offset
            draw.text((MARGIN, y), "\U0001f9d1 User:", fill=TEAL, font=font_bold)
            y += LINE_H + 4

            chars_left = int(typed)
            for line in user_lines:
                if chars_left <= 0:
                    break
                visible = line[:int(chars_left)]
                draw.text((MARGIN + 20, y), visible, fill=TEXT_COL, font=font)
                chars_left -= len(line)
                y += LINE_H

            if (typed // 3) % 2 == 0:
                draw.text((MARGIN + 20, y - LINE_H), "\u258a", fill=GREEN, font=font)

            self._draw_footer(draw, scene, font_sm)
            self._write(img)
            typed += uchars_per_frame

        # Hold complete user message
        img_u, draw_u = make_frame()
        self._draw_header(draw_u, scene, scene_num, font_bold, font_sm)
        y = chat_top + 10
        t_off = 0
        if scene.thumbnail:
            t_off = self._draw_thumbnail(img_u, draw_u, scene.thumbnail, y)
        y += t_off
        draw_u.text((MARGIN, y), "\U0001f9d1 User:", fill=TEAL, font=font_bold)
        y += LINE_H + 4
        for line in user_lines:
            draw_u.text((MARGIN + 20, y), line, fill=TEXT_COL, font=font)
            y += LINE_H
        self._draw_footer(draw_u, scene, font_sm)
        self._hold(img_u, 0.6)

        # === Phase 2: Thinking animation ===
        think_frames = int(FPS * min(2.0, max(1.0, scene.elapsed)))
        for i in range(think_frames):
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_sm)
            y = chat_top + 10
            t_off = 0
            if scene.thumbnail:
                t_off = self._draw_thumbnail(img, draw, scene.thumbnail, y)
            y += t_off
            draw.text((MARGIN, y), "\U0001f9d1 User:", fill=TEAL, font=font_bold)
            y += LINE_H + 4
            for line in user_lines:
                draw.text((MARGIN + 20, y), line, fill=TEXT_COL, font=font)
                y += LINE_H

            y += LINE_H
            draw.text((MARGIN, y), "\U0001f916 Pinocchio:", fill=PEACH, font=font_bold)
            y += LINE_H + 4

            dots = "\u00b7" * ((i // 8) % 4 + 1)
            phases = ["\u611f\u77e5\u4e2d", "\u7b56\u7565\u89c4\u5212", "\u6267\u884c\u4e2d", "\u8bc4\u4f30\u4e2d", "\u5b66\u4e60\u4e2d", "\u53cd\u601d\u4e2d"]
            phase = phases[(i // 15) % len(phases)]
            draw.text((MARGIN + 20, y), f"\U0001f504 \u8ba4\u77e5\u5faa\u73af \u2014 {phase} {dots}",
                       fill=YELLOW, font=font)

            elapsed_show = min(scene.elapsed, i / FPS)
            draw.text((WIDTH - 150, y), f"{elapsed_show:.1f}s", fill=DIM, font=font_sm)

            self._draw_footer(draw, scene, font_sm)
            self._write(img)

        # === Phase 3: Stream agent response ===
        total_rchars = sum(len(l) for l in resp_lines)
        rchars_per_frame = max(3, int(total_rchars // (FPS * 5)))

        typed = 0
        while typed <= total_rchars:
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_sm)

            y = chat_top + 10
            t_off = 0
            if scene.thumbnail:
                t_off = self._draw_thumbnail(img, draw, scene.thumbnail, y)
            y += t_off

            draw.text((MARGIN, y), "\U0001f9d1 User:", fill=TEAL, font=font_bold)
            y += LINE_H + 4
            user_show = user_lines[:3]
            if len(user_lines) > 3:
                user_show = user_lines[:2] + [f"\u2026 ({len(user_lines)} \u884c)"]
            for line in user_show:
                draw.text((MARGIN + 20, y), line, fill=DIM, font=font)
                y += LINE_H

            y += LINE_H
            draw.text((MARGIN, y), "\U0001f916 Pinocchio:", fill=PEACH, font=font_bold)
            y += LINE_H + 4

            max_vis_lines = (HEIGHT - y - 80) // LINE_H
            chars_left = int(typed)
            vis_lines: list[str] = []
            for line in resp_lines:
                if chars_left <= 0:
                    break
                vis_lines.append(line[:int(chars_left)])
                chars_left -= len(line)

            start = max(0, len(vis_lines) - max_vis_lines)
            for line in vis_lines[start:]:
                if y > HEIGHT - 80:
                    break
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "import ", "from ")):
                    draw.text((MARGIN + 20, y), line, fill=BLUE, font=font)
                elif stripped.startswith("#"):
                    draw.text((MARGIN + 20, y), line, fill=DIM, font=font)
                elif stripped.startswith(("return ", "if ", "for ", "while ", "elif ", "else:")):
                    draw.text((MARGIN + 20, y), line, fill=MAUVE, font=font)
                elif '"""' in line or "'''" in line:
                    draw.text((MARGIN + 20, y), line, fill=GREEN, font=font)
                elif line.startswith("{") or line.startswith("}") or '": ' in line:
                    draw.text((MARGIN + 20, y), line, fill=SKY, font=font)
                else:
                    draw.text((MARGIN + 20, y), line, fill=TEXT_COL, font=font)
                y += LINE_H

            self._draw_footer(draw, scene, font_sm)
            self._write(img)
            typed += rchars_per_frame

        # === Phase 4: Hold final frame ===
        self._hold(img, 2.0)

    # ── Header / Footer helpers ──────────────────────────────

    def _draw_header(self, draw, scene, num, font_bold, font_sm):
        draw.rectangle([0, 0, WIDTH, 46], fill=SURFACE)
        draw.text((MARGIN, 12), f"{scene.icon} {scene.title}", fill=BLUE, font=font_bold)

        badge_colors = {"text": TEAL, "vision": PEACH, "multi": YELLOW}
        bc = badge_colors.get(scene.modality, TEXT_COL)
        draw.text((WIDTH - 300, 14), f"[{scene.modality.upper()}]  Scene {num}",
                   fill=bc, font=font_sm)

    def _draw_footer(self, draw, scene, font_sm):
        draw.rectangle([0, HEIGHT - 36, WIDTH, HEIGHT], fill=SURFACE)
        draw.text((MARGIN, HEIGHT - 28), f"\u23f1 {scene.elapsed:.1f}s", fill=DIM, font=font_sm)
        cat_label = f"{scene.category}  \u00b7  Pinocchio \u00b7 qwen3-vl:8b \u00b7 \u8ba4\u77e5\u5faa\u73af"
        draw.text((WIDTH // 2 - 200, HEIGHT - 28), cat_label, fill=DIM, font=font_sm)

    def _draw_thumbnail(self, img, draw, thumb, y_start) -> int:
        max_w, max_h = 350, 250
        tw, th = thumb.size
        scale = min(max_w / tw, max_h / th)
        new_w, new_h = int(tw * scale), int(th * scale)
        resized = thumb.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = WIDTH - new_w - MARGIN
        img.paste(resized, (x, y_start))
        draw.rectangle([x - 2, y_start - 2, x + new_w + 2, y_start + new_h + 2],
                       outline=SURFACE, width=2)
        return 0

    # ── Summary Screen ───────────────────────────────────────

    def render_summary(self, scenes: list[DemoScene], duration: float = 6.0):
        fh1 = get_font(H1_SZ, bold=True)
        fh2 = get_font(H2_SZ)
        fs = get_font(FONT_SZ)

        total_time = sum(s.elapsed for s in scenes)
        n = len(scenes)
        n_ok = sum(1 for s in scenes if "[Error" not in s.agent_response)
        n_fail = n - n_ok
        cats = sorted(set(s.category for s in scenes))

        for i in range(int(FPS * duration)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.6))

            draw.rectangle([0, 0, 6, HEIGHT], fill=alpha_color(GREEN, a))

            draw.text((MARGIN, 30), "\U0001f4cb \u6d4b\u8bd5\u6f14\u793a\u603b\u7ed3",
                       fill=alpha_color(BLUE, a), font=fh1)

            y = 100
            stats = [
                (f"\u603b\u573a\u666f: {n}", TEAL),
                (f"\u6210\u529f: {n_ok}  |  \u5931\u8d25: {n_fail}", GREEN if n_fail == 0 else RED),
                (f"\u603b\u8017\u65f6: {total_time:.1f}s  |  \u5e73\u5747: {total_time / max(n, 1):.1f}s/\u573a\u666f", LAVENDER),
            ]
            for text, color in stats:
                draw.text((MARGIN, y), text,
                           fill=alpha_color(color, a), font=fh2)
                y += 42

            y += 10
            draw.text((MARGIN, y), "\u573a\u666f\u56de\u987e:", fill=alpha_color(DIM, a), font=fs)
            y += 32

            for j, scene in enumerate(scenes):
                if y > HEIGHT - 70:
                    draw.text((MARGIN + 20, y), f"\u2026 \u8fd8\u6709 {n - j} \u4e2a\u573a\u666f",
                               fill=alpha_color(DIM, a), font=fs)
                    break
                ok = "\u2705" if "[Error" not in scene.agent_response else "\u274c"
                line = f"{ok}  {scene.icon}  {scene.title}  \u2014  {scene.elapsed:.1f}s"
                row_a = min(1.0, max(0, (i - j * 2) / (FPS * 0.3)))
                if row_a > 0:
                    draw.text((MARGIN + 20, y), line,
                               fill=alpha_color(TEXT_COL, row_a * a), font=fs)
                y += LINE_H

            foot = "Pinocchio Multimodal Agent  \u00b7  Test Demo  \u00b7  2026-03-02"
            fbx = draw.textbbox((0, 0), foot, font=fs)
            draw.text(((WIDTH - fbx[2] + fbx[0]) // 2, HEIGHT - 50),
                       foot, fill=alpha_color(DIM, a), font=fs)

            self._write(img)


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("\U0001f3ac Pinocchio \u591a\u6a21\u6001\u6d4b\u8bd5\u7528\u4f8b\u6f14\u793a\u89c6\u9891\u751f\u6210\u5668")
    print("=" * 60)

    render_only = "--render-only" in sys.argv

    scenes = None
    if render_only:
        scenes = load_cache()
        if scenes is None:
            print("\u26a0\ufe0f  No cache found, falling back to full run")

    if scenes is None:
        scenes = build_all_scenarios()
        scenes = run_all_scenarios(scenes)
        save_cache(scenes)

    print(f"\n\U0001f3ac Rendering {len(scenes)} scenes into video \u2026\n")
    renderer = VideoRenderer()

    print("   \U0001f4fc  Title screen")
    renderer.render_title(scenes, duration=5.0)

    categories_order = [
        "\u6587\u672c\u5904\u7406",
        "\u56fe\u50cf\u7406\u89e3",
        "\u591a\u8f6e\u5bf9\u8bdd",
        "\u8fb9\u754c\u573a\u666f",
        "\u8ba4\u77e5\u5faa\u73af",
    ]
    scene_num = 0

    for category in categories_order:
        cat_scenes = [s for s in scenes if s.category == category]
        if not cat_scenes:
            continue

        print(f"   \U0001f4fc  Category: {category} ({len(cat_scenes)} scenes)")
        renderer.render_category_transition(category, cat_scenes)

        for scene in cat_scenes:
            scene_num += 1
            print(f"      \U0001f39e  [{scene_num}/{len(scenes)}] {scene.title}")
            renderer.render_scene_transition(scene, scene_num, len(scenes))
            renderer.render_scene(scene, scene_num)

    print("   \U0001f4fc  Summary")
    renderer.render_summary(scenes, duration=6.0)
    renderer.close()

    video_sec = renderer.frame_count / FPS
    total_time = sum(s.elapsed for s in scenes)
    n_ok = sum(1 for s in scenes if "[Error" not in s.agent_response)

    print(f"\n{'=' * 60}")
    print(f"\u2705 \u6f14\u793a\u89c6\u9891\u5df2\u751f\u6210: {OUTPUT_PATH}")
    print(f"   \u5206\u8fa8\u7387: {WIDTH}\u00d7{HEIGHT} @ {FPS}fps")
    print(f"   \u89c6\u9891\u65f6\u957f: {video_sec:.0f}s ({video_sec / 60:.1f}min)")
    print(f"   \u5e27\u6570: {renderer.frame_count}")
    print(f"   \u6d4b\u8bd5\u573a\u666f: {len(scenes)} (\u6210\u529f {n_ok})")
    print(f"   LLM \u603b\u8017\u65f6: {total_time:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n\U0001f4a1 \u518d\u6b21\u6e32\u67d3(\u65e0\u9700\u91cd\u65b0\u8fd0\u884cLLM): python3 {__file__} --render-only")


if __name__ == "__main__":
    main()
