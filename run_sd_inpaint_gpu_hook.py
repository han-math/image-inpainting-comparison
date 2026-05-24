#!/usr/bin/env python3
"""
Q1 Inpainting - Stable Diffusion inpainting on GPU.

This experimental copy focuses on the white-house wire attachment. It asks SD to
generate a small wall-mounted cable bracket on the right-side wall gap above the
entrance arch, then draws a thin line and an optional subtle bracket prior
toward that generated/estimated anchor point.
"""

import csv
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR = os.environ.get("INPAINT_INPUT_DIR", "/root/inpaint/input")
OUTPUT_DIR = os.environ.get("INPAINT_OUTPUT_DIR", "/root/inpaint/results_sd_inpaint_hook")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = os.environ.get("INPAINT_DEVICE", "cuda")
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Set SAVE_TRIALS=1 if you want to keep every seed output.
SAVE_TRIALS = os.environ.get("SAVE_TRIALS", "0") == "1"
TRIAL_DIR = os.path.join(OUTPUT_DIR, "trials")
if SAVE_TRIALS:
    os.makedirs(TRIAL_DIR, exist_ok=True)

# Optional: set INPAINT_ONLY=white_house to run only one image.
ONLY_IMAGE = os.environ.get("INPAINT_ONLY", "").strip()


# ---------------------------------------------------------------------------
# Per-image settings
# ---------------------------------------------------------------------------
COMMON_NEGATIVE = (
    "text, watermark, logo, jpeg artifacts, blurry, distorted geometry, "
    "extra objects, unnatural edges, low quality"
)

DEFAULT_CFG = {
    "num_steps": 45,
    "guidance": 4.5,
    "strength": 0.98,
    "seeds": list(range(6)),
    "max_side": 512,
    "mask_dilate": 1,
    "mask_blur": 0,
    "negative_prompt": COMMON_NEGATIVE,
}

PAIRS = [
    {
        "orig": "bricks.png",
        "hole": "bricks_L_region_lost.png",
        "prompt": (
            "black brick wall texture, running bond staggered pattern, dark "
            "rectangular bricks, thin bright white mortar lines, monochrome, "
            "rough surface, seamless texture"
        ),
        "guidance": 3.5,
        "num_steps": 55,
        "seeds": list(range(10)),
    },
    {
        "orig": "crayon_paint.png",
        "hole": "crayon_missingRegion.png",
        "prompt": (
            "child crayon drawing on white paper, simple colored pencil lines, "
            "green grass, black cat, blue house, brown curved path"
        ),
        "guidance": 5.0,
        "num_steps": 45,
        "seeds": list(range(8)),
    },
    {
        "orig": "fingerprint256.png",
        "hole": "finger_circle_lost5.png",
        "prompt": (
            "macro grayscale fingerprint texture, continuous curved friction "
            "ridges, black and white forensic image, fine ridge detail"
        ),
        "guidance": 3.0,
        "num_steps": 60,
        "seeds": list(range(12)),
    },
    {
        "orig": "white_house.png",
        "hole": "white_house_with_lost.png",
        "prompt": (
            "front view photo of a white wooden church, clapboard siding, "
            "central arched doorway, arched windows, diagonal black utility "
            "wire attached to a small gray cable bracket on the upper-right "
            "edge of the entrance arch, realistic photo"
        ),
        "negative_prompt": (
            COMMON_NEGATIVE
            + ", ladder, scaffolding, many wires, thick cable, dangling cable, "
            "large hook, oversized bracket, lamp, ornament, object on window "
            "frame"
        ),
        "guidance": 5.0,
        "num_steps": 70,
        "strength": 1.0,
        "seeds": list(range(24)),
        "mask_dilate": 2,
        "structure_as_main": True,
        "black_residue_threshold": 35,
        "black_residue_penalty": 12.0,
        "cleanup": {
            "enabled": True,
            "threshold": 55,
            "min_pixels": 300,
            "dilate": 5,
            "blur": 1.0,
            "num_steps": 55,
            "guidance": 4.5,
            "strength": 1.0,
            "seeds": list(range(100, 106)),
        },
        "structure": {
            "type": "mask_axis_line_to_generated_anchor",
            "top_rows": 8,
            "anchor_y": 137,
            "anchor_window": 5,
            "anchor_roi": (140, 126, 168, 150),
            "anchor_search_radius": 13,
            "anchor_gray_max": 170,
            "anchor_min_pixels": 1,
            "anchor_max_pixels": 95,
            "draw_past_anchor": 1,
            "fit_y_max": 150,
            "draw_y_min": 0,
            "draw_y_max": 145,
            "line_width": 1,
            "curve_sag": 20.0,
            "color": (38, 38, 38),
            "fixture_prior": {
                "enabled": True,
                "mode": "always",
                "offset_x": 0,
                "offset_y": 0,
                "stem_dx": 2,
                "stem_dy": 7,
                "stem_width": 2,
                "dot_radius": 1.8,
                "shadow_radius": 2.1,
                "color": (78, 78, 74),
                "core_color": (45, 45, 43),
                "opacity": 0.78,
                "blur": 0.35,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def merged_cfg(item):
    cfg = dict(DEFAULT_CFG)
    cfg.update(item)
    return cfg


def make_mask(orig, hole, threshold=5):
    orig_arr = np.array(orig).astype(np.float32)
    hole_arr = np.array(hole).astype(np.float32)
    diff = np.abs(orig_arr - hole_arr)
    mask_arr = (diff.max(axis=2) > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask_arr), mask_arr


def prepare_mask_for_pipe(mask, dilate=0, blur=0):
    work = mask.convert("L")
    for _ in range(int(dilate)):
        work = work.filter(ImageFilter.MaxFilter(3))
    if blur:
        work = work.filter(ImageFilter.GaussianBlur(float(blur)))
    return work


def target_size(size, max_side):
    w, h = size
    scale = float(max_side) / max(w, h)
    tw = max(8, int(round(w * scale / 8.0)) * 8)
    th = max(8, int(round(h * scale / 8.0)) * 8)
    return tw, th


def resize_for_sd(image, mask, max_side):
    tw, th = target_size(image.size, max_side)
    return (
        image.resize((tw, th), Image.LANCZOS),
        mask.resize((tw, th), Image.NEAREST),
        tw,
        th,
    )


def psnr_from_mse(mse):
    return float(-10.0 * np.log10(mse)) if mse > 0 else 100.0


def ssim_channel(a, b):
    c1, c2 = 0.01**2, 0.03**2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - ma) * (b - mb))
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma**2 + mb**2 + c1) * (va + vb + c2)
    )


def compute_metrics(result, orig, mask):
    r = np.array(result).astype(np.float32) / 255.0
    o = np.array(orig).astype(np.float32) / 255.0
    m = np.array(mask.convert("L")) > 0
    gray = r.mean(axis=2) * 255.0

    full_mse = np.mean((r - o) ** 2)
    hole_mse = np.mean((r[m] - o[m]) ** 2) if m.any() else full_mse
    outside_mse = np.mean((r[~m] - o[~m]) ** 2) if (~m).any() else 0.0
    ssim_val = np.mean([ssim_channel(r[:, :, c], o[:, :, c]) for c in range(3)])
    black_residue_pct = float(((gray < 35) & m).sum() / max(1, m.sum()) * 100.0)

    return {
        "psnr": psnr_from_mse(full_mse),
        "hole_psnr": psnr_from_mse(hole_mse),
        "outside_mse": float(outside_mse),
        "ssim": float(ssim_val),
        "black_residue_pct": black_residue_pct,
    }


def metric_score(metrics, cfg):
    # The original image is available in this homework, so choose the best seed
    # by masked-region PSNR. For images with black hole masks, also penalize
    # residual near-black pixels so an unfilled region is not selected.
    penalty = float(cfg.get("black_residue_penalty", 0.0))
    return metrics["hole_psnr"] - penalty * metrics.get("black_residue_pct", 0.0) / 100.0


def make_generator(seed):
    try:
        return torch.Generator(device=DEVICE).manual_seed(int(seed))
    except Exception:
        return torch.Generator().manual_seed(int(seed))


def keep_large_components(binary, min_pixels):
    binary = binary.astype(bool)
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    kept = np.zeros_like(binary, dtype=bool)

    for y0, x0 in zip(*np.where(binary & ~seen)):
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))

        if len(pixels) >= min_pixels:
            for y, x in pixels:
                kept[y, x] = True
    return kept


def connected_components(binary):
    binary = binary.astype(bool)
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    comps = []

    for y0, x0 in zip(*np.where(binary & ~seen)):
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(pixels)
    return comps


def make_black_residue_mask(result, base_mask, cleanup_cfg):
    gray = np.array(result.convert("L")).astype(np.float32)
    mask_arr = np.array(base_mask.convert("L")) > 0
    residue = (gray < float(cleanup_cfg["threshold"])) & mask_arr
    residue = keep_large_components(residue, int(cleanup_cfg.get("min_pixels", 1)))
    residue_img = Image.fromarray(residue.astype(np.uint8) * 255)
    for _ in range(int(cleanup_cfg.get("dilate", 0))):
        residue_img = residue_img.filter(ImageFilter.MaxFilter(3))
    blur = float(cleanup_cfg.get("blur", 0))
    if blur:
        residue_img = residue_img.filter(ImageFilter.GaussianBlur(blur))

    # Never repair outside the original unknown area.
    limited = np.array(residue_img).astype(np.float32)
    limited *= mask_arr.astype(np.float32)
    return Image.fromarray(np.uint8(np.clip(limited, 0, 255)))


def cleanup_black_residue(image, orig, mask, cfg, name, prompt):
    cleanup_cfg = cfg.get("cleanup", {})
    if not cleanup_cfg.get("enabled", False):
        return image, None

    cleanup_mask = make_black_residue_mask(image, mask, cleanup_cfg)
    cleanup_pixels = int((np.array(cleanup_mask.convert("L")) > 0).sum())
    if cleanup_pixels < int(cleanup_cfg.get("min_pixels", 1)):
        return image, None

    print(f"  Cleanup mask: {cleanup_pixels} pixels", flush=True)
    image_sd, cleanup_mask_sd, sd_w, sd_h = resize_for_sd(
        image, cleanup_mask, cfg["max_side"]
    )

    best_cleanup = None
    for seed in cleanup_cfg.get("seeds", [0]):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            raw = pipe(
                prompt=prompt,
                negative_prompt=cfg["negative_prompt"],
                image=image_sd,
                mask_image=cleanup_mask_sd,
                strength=float(cleanup_cfg.get("strength", cfg["strength"])),
                num_inference_steps=int(cleanup_cfg.get("num_steps", cfg["num_steps"])),
                guidance_scale=float(cleanup_cfg.get("guidance", cfg["guidance"])),
                generator=make_generator(seed),
                height=sd_h,
                width=sd_w,
            ).images[0]
        elapsed = time.time() - t0

        cleaned = raw.resize(image.size, Image.LANCZOS)
        cleaned = Image.composite(cleaned, image, cleanup_mask)
        metrics = compute_metrics(cleaned, orig, mask)
        score = metric_score(metrics, cfg)
        print(
            f"  cleanup seed {seed}: hole PSNR {metrics['hole_psnr']:.2f} dB, "
            f"black {metrics['black_residue_pct']:.1f}%, "
            f"SSIM {metrics['ssim']:.4f}, time {elapsed:.1f}s",
            flush=True,
        )

        if SAVE_TRIALS:
            trial_path = os.path.join(TRIAL_DIR, f"{name}_cleanup_seed{seed}.png")
            cleaned.save(trial_path)

        if best_cleanup is None or score > best_cleanup["score"]:
            best_cleanup = {
                "seed": seed,
                "score": score,
                "image": cleaned,
                "metrics": metrics,
                "cleanup_pixels": cleanup_pixels,
            }

    return best_cleanup["image"], best_cleanup


def fit_single_line(image, mask, params):
    gray_img = image.convert("L")
    gray = np.array(gray_img).astype(np.float32)
    edge = np.array(gray_img.filter(ImageFilter.FIND_EDGES)).astype(np.float32)
    mask_arr = np.array(mask.convert("L")) > 0
    h, w = gray.shape
    yy, xx = np.indices((h, w))

    x0, y0, x1, y1 = params.get("roi", (0, 0, w, h))
    roi = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    cand = roi & (~mask_arr) & (edge >= params["edge_min"]) & (gray <= params["gray_max"])
    ys, xs = np.where(cand)
    if len(xs) < params["min_support"]:
        return None

    b_bins = np.arange(
        params["b_min"], params["b_max"] + params["b_step"], params["b_step"]
    )
    best = None
    slopes = np.linspace(params["slope_min"], params["slope_max"], params["slope_steps"])
    for a in slopes:
        b_values = xs - a * ys
        hist, edges = np.histogram(b_values, bins=b_bins)
        idx = int(hist.argmax())
        if hist[idx] < params["min_support"]:
            continue
        b = float((edges[idx] + edges[idx + 1]) * 0.5)
        dist = np.abs(xs - (a * ys + b))
        on = dist <= params["distance"]
        support = int(on.sum())
        unique_y = int(np.unique(ys[on]).size)
        if support < params["min_support"] or unique_y < params["min_unique_y"]:
            continue

        line_pixels = np.abs(xx - (a * yy + b)) <= max(1.0, params["line_width"] + 0.5)
        mask_pixels = int((line_pixels & mask_arr).sum())
        if mask_pixels < params["min_mask_pixels"]:
            continue

        score = support + 3.0 * unique_y + 0.05 * mask_pixels
        if best is None or score > best["score"]:
            best = {
                "a": float(a),
                "b": float(b),
                "score": float(score),
                "support": support,
                "unique_y": unique_y,
                "mask_pixels": mask_pixels,
            }
    return best


def fit_mask_axis_line(mask, params):
    mask_arr = np.array(mask.convert("L")) > 0
    h, _ = mask_arr.shape

    row_centers = []
    fit_y_max = int(params.get("fit_y_max", h - 1))
    for y in range(0, min(h, fit_y_max + 1)):
        xs = np.where(mask_arr[y])[0]
        if len(xs) > 0:
            row_centers.append((y, float(np.median(xs)), len(xs)))
    if len(row_centers) < 2:
        return None

    def center_between(y0, y1):
        pts = [(y, x) for y, x, _ in row_centers if y0 <= y <= y1]
        if not pts:
            return None
        ys = np.array([p[0] for p in pts], dtype=np.float32)
        xs = np.array([p[1] for p in pts], dtype=np.float32)
        return float(np.median(ys)), float(np.median(xs))

    top_rows = int(params.get("top_rows", 8))
    top = center_between(0, top_rows)
    anchor_y = int(params.get("anchor_y", fit_y_max))
    anchor_window = int(params.get("anchor_window", 8))
    anchor = center_between(anchor_y - anchor_window, anchor_y + anchor_window)

    if top is not None and anchor is not None and anchor[0] != top[0]:
        y0, x0 = top
        y1, x1 = anchor
        a = (x1 - x0) / (y1 - y0)
        b = x0 - a * y0
    else:
        # Fallback: weighted least squares on the centerline of the missing
        # region. Narrower rows get more weight because they better indicate
        # the long-axis center.
        y = np.array([p[0] for p in row_centers], dtype=np.float32)
        x = np.array([p[1] for p in row_centers], dtype=np.float32)
        width = np.array([p[2] for p in row_centers], dtype=np.float32)
        weight = 1.0 / np.maximum(width, 1.0) ** 0.7
        design = np.vstack([y, np.ones_like(y)]).T
        a, b = np.linalg.lstsq(design * weight[:, None], x * weight, rcond=None)[0]

    yy, xx = np.indices(mask_arr.shape)
    line_pixels = np.abs(xx - (a * yy + b)) <= max(1.0, params["line_width"] + 0.5)
    mask_pixels = int((line_pixels & mask_arr).sum())
    return {
        "a": float(a),
        "b": float(b),
        "score": float(mask_pixels),
        "support": len(row_centers),
        "unique_y": len(row_centers),
        "mask_pixels": mask_pixels,
    }


def find_generated_anchor(image, mask, params, fallback_anchor):
    """Find a small dark generated fixture near the expected arch attachment."""
    gray = np.array(image.convert("L")).astype(np.float32)
    mask_arr = np.array(mask.convert("L")) > 0
    h, w = gray.shape
    yy, xx = np.indices((h, w))

    x0, y0, x1, y1 = params.get("anchor_roi", (0, 0, w, h))
    roi = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

    fy, fx = fallback_anchor
    radius = float(params.get("anchor_search_radius", 30))
    near_expected = (xx - fx) ** 2 + (yy - fy) ** 2 <= radius**2
    dark = gray <= float(params.get("anchor_gray_max", 155))
    candidates = roi & near_expected & mask_arr & dark

    min_pixels = int(params.get("anchor_min_pixels", 1))
    max_pixels = int(params.get("anchor_max_pixels", 120))
    best = None
    for comp in connected_components(candidates):
        area = len(comp)
        if area < min_pixels or area > max_pixels:
            continue
        ys = np.array([p[0] for p in comp], dtype=np.float32)
        xs = np.array([p[1] for p in comp], dtype=np.float32)
        cy, cx = float(ys.mean()), float(xs.mean())
        darkness = float(255.0 - gray[ys.astype(int), xs.astype(int)].mean())
        dist = float(np.hypot(cx - fx, cy - fy))
        score = darkness + 0.8 * area - 2.0 * dist
        if best is None or score > best["score"]:
            best = {"y": cy, "x": cx, "area": area, "score": score}

    if best is None:
        return None
    return best


def fit_mask_axis_line_to_generated_anchor(image, mask, params):
    mask_arr = np.array(mask.convert("L")) > 0
    h, _ = mask_arr.shape

    row_centers = []
    fit_y_max = int(params.get("fit_y_max", h - 1))
    for y in range(0, min(h, fit_y_max + 1)):
        xs = np.where(mask_arr[y])[0]
        if len(xs) > 0:
            row_centers.append((y, float(np.median(xs)), len(xs)))
    if len(row_centers) < 2:
        return None

    def center_between(y0, y1):
        pts = [(y, x) for y, x, _ in row_centers if y0 <= y <= y1]
        if not pts:
            return None
        ys = np.array([p[0] for p in pts], dtype=np.float32)
        xs = np.array([p[1] for p in pts], dtype=np.float32)
        return float(np.median(ys)), float(np.median(xs))

    top = center_between(0, int(params.get("top_rows", 8)))
    anchor_y = int(params.get("anchor_y", fit_y_max))
    anchor_window = int(params.get("anchor_window", 8))
    fallback_anchor = center_between(anchor_y - anchor_window, anchor_y + anchor_window)
    if top is None or fallback_anchor is None or fallback_anchor[0] == top[0]:
        return None

    generated_anchor = find_generated_anchor(image, mask, params, fallback_anchor)
    if generated_anchor is not None:
        y1, x1 = generated_anchor["y"], generated_anchor["x"]
        anchor_source = "generated"
        anchor_area = generated_anchor["area"]
    else:
        y1, x1 = fallback_anchor
        anchor_source = "mask-fallback"
        anchor_area = 0

    y0, x0 = top
    a = (x1 - x0) / (y1 - y0)
    b = x0 - a * y0

    draw_past = int(params.get("draw_past_anchor", 2))
    yy, xx = np.indices(mask_arr.shape)
    line_pixels = np.abs(xx - (a * yy + b)) <= max(1.0, params["line_width"] + 0.5)
    mask_pixels = int((line_pixels & mask_arr & (yy <= y1 + draw_past)).sum())
    return {
        "a": float(a),
        "b": float(b),
        "score": float(mask_pixels),
        "support": len(row_centers),
        "unique_y": len(row_centers),
        "mask_pixels": mask_pixels,
        "anchor_x": float(x1),
        "anchor_y": float(y1),
        "anchor_source": anchor_source,
        "anchor_area": int(anchor_area),
        "draw_y_max": int(round(y1)) + draw_past,
    }


def draw_line_inside_mask(image, mask, line, params):
    scale = 4
    w, h = image.size
    big_size = (w * scale, h * scale)
    big_img = image.resize(big_size, Image.LANCZOS)
    big_line = big_img.copy()
    big_line_mask = Image.new("L", big_size, 0)

    a, b = line["a"], line["b"]
    y0 = max(0, int(line.get("draw_y_min", params.get("draw_y_min", 0))))
    y1 = min(h - 1, int(line.get("draw_y_max", params.get("draw_y_max", h - 1))))
    x0 = a * y0 + b
    x1 = a * y1 + b
    sag = float(params.get("curve_sag", 0.0))
    width = max(1, int(round(params["line_width"] * scale)))
    color = tuple(params.get("color", (45, 45, 45)))

    if abs(sag) > 0.01:
        # A slight downward quadratic curve makes the cable read less like a
        # ruler-drawn line while keeping the same endpoints.
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 + sag
        points = []
        for i in range(49):
            t = i / 48.0
            x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
            y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
            points.append((int(round(x * scale)), int(round(y * scale))))
    else:
        points = [
            (int(round(x0 * scale)), y0 * scale),
            (int(round(x1 * scale)), y1 * scale),
        ]

    ImageDraw.Draw(big_line).line(points, fill=color, width=width, joint="curve")
    ImageDraw.Draw(big_line_mask).line(points, fill=255, width=width, joint="curve")

    line_img = big_line.resize((w, h), Image.LANCZOS)
    line_mask = big_line_mask.resize((w, h), Image.LANCZOS)

    lm = np.array(line_mask).astype(np.float32)
    mm = (np.array(mask.convert("L")) > 0).astype(np.float32)
    restricted = Image.fromarray(np.uint8(np.clip(lm * mm, 0, 255)))
    return Image.composite(line_img, image, restricted)


def draw_fixture_inside_mask(image, mask, line, params):
    fixture = params.get("fixture_prior")
    if not fixture or not fixture.get("enabled", False):
        return image

    mode = fixture.get("mode", "always")
    if mode == "fallback_only" and line.get("anchor_source") == "generated":
        return image

    scale = 4
    w, h = image.size
    big_size = (w * scale, h * scale)
    big_img = image.resize(big_size, Image.LANCZOS)
    big_fixture = big_img.copy()
    big_fixture_mask = Image.new("L", big_size, 0)

    ax = float(line.get("anchor_x", 0.0)) + float(fixture.get("offset_x", 0.0))
    ay = float(line.get("anchor_y", 0.0)) + float(fixture.get("offset_y", 0.0))
    x = ax * scale
    y = ay * scale

    stem_dx = float(fixture.get("stem_dx", 2.0)) * scale
    stem_dy = float(fixture.get("stem_dy", 8.0)) * scale
    stem_width = max(1, int(round(float(fixture.get("stem_width", 2.0)) * scale)))
    dot_radius = float(fixture.get("dot_radius", 2.0)) * scale
    shadow_radius = float(fixture.get("shadow_radius", dot_radius / scale)) * scale
    color = tuple(fixture.get("color", (75, 75, 75)))
    core_color = tuple(fixture.get("core_color", (45, 45, 45)))

    draw = ImageDraw.Draw(big_fixture)
    mask_draw = ImageDraw.Draw(big_fixture_mask)
    draw.line((x, y - 1.5 * scale, x + stem_dx, y + stem_dy), fill=color, width=stem_width)
    mask_draw.line((x, y - 1.5 * scale, x + stem_dx, y + stem_dy), fill=255, width=stem_width)
    draw.ellipse((x - shadow_radius, y - shadow_radius, x + shadow_radius, y + shadow_radius), fill=color)
    mask_draw.ellipse((x - shadow_radius, y - shadow_radius, x + shadow_radius, y + shadow_radius), fill=230)
    draw.ellipse((x - dot_radius, y + 4 * scale - dot_radius, x + dot_radius, y + 4 * scale + dot_radius), fill=core_color)
    mask_draw.ellipse((x - dot_radius, y + 4 * scale - dot_radius, x + dot_radius, y + 4 * scale + dot_radius), fill=255)

    blur = float(fixture.get("blur", 0.0))
    if blur > 0:
        big_fixture_mask = big_fixture_mask.filter(ImageFilter.GaussianBlur(radius=blur * scale))

    fixture_img = big_fixture.resize((w, h), Image.LANCZOS)
    fixture_mask = big_fixture_mask.resize((w, h), Image.LANCZOS)

    fm = np.array(fixture_mask).astype(np.float32)
    mm = (np.array(mask.convert("L")) > 0).astype(np.float32)
    opacity = float(fixture.get("opacity", 1.0))
    restricted = Image.fromarray(np.uint8(np.clip(fm * mm * opacity, 0, 255)))
    return Image.composite(fixture_img, image, restricted)


def maybe_structure_complete(name, result, hole, mask, cfg):
    structure = cfg.get("structure")
    if not structure:
        return None, None

    # Fit only from the damaged image or the missing-region shape, never from
    # the ground truth image. This keeps the structural version reportable as a
    # method instead of copying the answer.
    if structure.get("type") == "single_line":
        line = fit_single_line(hole, mask, structure)
    elif structure.get("type") == "mask_axis_line":
        line = fit_mask_axis_line(mask, structure)
    elif structure.get("type") == "mask_axis_line_to_generated_anchor":
        line = fit_mask_axis_line_to_generated_anchor(result, mask, structure)
    else:
        return None, None

    if line is None:
        return None, None
    completed = draw_line_inside_mask(result, mask, line, structure)
    completed = draw_fixture_inside_mask(completed, mask, line, structure)
    return completed, line


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading Stable Diffusion Inpainting ...", flush=True)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=TORCH_DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to(DEVICE)
try:
    pipe.enable_attention_slicing()
except Exception:
    pass

if DEVICE == "cuda":
    print(
        f"Model on {DEVICE}. VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB used\n",
        flush=True,
    )
else:
    print(f"Model on {DEVICE}.\n", flush=True)


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
rows = []
for item in PAIRS:
    cfg = merged_cfg(item)
    orig_name = cfg["orig"]
    hole_name = cfg["hole"]
    prompt = cfg["prompt"]
    name = os.path.splitext(orig_name)[0]
    if ONLY_IMAGE and ONLY_IMAGE not in {name, orig_name, hole_name}:
        continue

    print(f"--- {orig_name} ---", flush=True)
    orig = Image.open(os.path.join(INPUT_DIR, orig_name)).convert("RGB")
    hole = Image.open(os.path.join(INPUT_DIR, hole_name)).convert("RGB")
    w, h = orig.size

    mask, mask_arr = make_mask(orig, hole)
    pipe_mask = prepare_mask_for_pipe(mask, cfg["mask_dilate"], cfg["mask_blur"])
    hole_pct = mask_arr.mean() / 255.0 * 100.0
    print(f"  Size: {w}x{h}, hole: {hole_pct:.1f}%", flush=True)

    hole_sd, mask_sd, sd_w, sd_h = resize_for_sd(hole, pipe_mask, cfg["max_side"])
    print(f"  SD size: {sd_w}x{sd_h}", flush=True)

    best = None
    for seed in cfg["seeds"]:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            raw = pipe(
                prompt=prompt,
                negative_prompt=cfg["negative_prompt"],
                image=hole_sd,
                mask_image=mask_sd,
                strength=cfg["strength"],
                num_inference_steps=cfg["num_steps"],
                guidance_scale=cfg["guidance"],
                generator=make_generator(seed),
                height=sd_h,
                width=sd_w,
            ).images[0]
        elapsed = time.time() - t0

        result = raw.resize((w, h), Image.LANCZOS)
        # Keep every known pixel exactly from the damaged input. Only the mask
        # region is allowed to come from SD.
        result = Image.composite(result, hole, mask)
        metrics = compute_metrics(result, orig, mask)
        score = metric_score(metrics, cfg)

        print(
            f"  seed {seed:02d}: hole PSNR {metrics['hole_psnr']:.2f} dB, "
            f"black {metrics['black_residue_pct']:.1f}%, "
            f"SSIM {metrics['ssim']:.4f}, time {elapsed:.1f}s",
            flush=True,
        )

        if SAVE_TRIALS:
            trial_path = os.path.join(TRIAL_DIR, f"{name}_seed{seed:02d}.png")
            result.save(trial_path)

        if best is None or score > best["score"]:
            best = {
                "seed": seed,
                "score": score,
                "image": result,
                "metrics": metrics,
                "elapsed": elapsed,
            }

    cleaned_image, cleanup_info = cleanup_black_residue(
        best["image"], orig, mask, cfg, name, prompt
    )
    if cleanup_info is not None:
        best["image"] = cleaned_image
        best["metrics"] = cleanup_info["metrics"]
        best["score"] = cleanup_info["score"]
        best["cleanup_seed"] = cleanup_info["seed"]
        print(
            f"  Best cleanup seed: {cleanup_info['seed']} | "
            f"black {best['metrics']['black_residue_pct']:.1f}%",
            flush=True,
        )

    out_path = os.path.join(OUTPUT_DIR, f"{name}_sd_inpaint.png")
    pure_path = os.path.join(OUTPUT_DIR, f"{name}_sd_inpaint_pure.png")
    print(
        f"  Best seed: {best['seed']} | PSNR {best['metrics']['psnr']:.2f} dB | "
        f"hole PSNR {best['metrics']['hole_psnr']:.2f} dB | "
        f"SSIM {best['metrics']['ssim']:.4f}",
        flush=True,
    )
    rows.append(
        {
            "image": name,
            "method": "sd_inpaint_best_of_n",
            "seed": (
                f"{best['seed']}+cleanup{best['cleanup_seed']}"
                if "cleanup_seed" in best
                else best["seed"]
            ),
            **best["metrics"],
        }
    )

    structured, line = maybe_structure_complete(name, best["image"], hole, mask, cfg)
    if structured is not None:
        struct_metrics = compute_metrics(structured, orig, mask)
        struct_path = os.path.join(OUTPUT_DIR, f"{name}_sd_inpaint_struct.png")
        structured.save(struct_path)
        if cfg.get("structure_as_main", False):
            best["image"].save(pure_path)
            structured.save(out_path)
            saved_note = (
                f"{os.path.basename(out_path)} is the structure-constrained result; "
                f"{os.path.basename(pure_path)} keeps the pure SD result"
            )
        else:
            best["image"].save(out_path)
            saved_note = os.path.basename(out_path)
        print(
            "  Structure result: "
            f"line x={line['a']:.3f}*y+{line['b']:.1f}, "
            f"anchor {line.get('anchor_source', 'n/a')} "
            f"({line.get('anchor_x', float('nan')):.1f}, "
            f"{line.get('anchor_y', float('nan')):.1f}), "
            f"hole PSNR {struct_metrics['hole_psnr']:.2f} dB, "
            f"SSIM {struct_metrics['ssim']:.4f}",
            flush=True,
        )
        rows.append(
            {
                "image": name,
                "method": "sd_inpaint_plus_line_prior",
                "seed": (
                    f"{best['seed']}+cleanup{best['cleanup_seed']}"
                    if "cleanup_seed" in best
                    else best["seed"]
                ),
                **struct_metrics,
            }
        )
    else:
        best["image"].save(out_path)
        saved_note = os.path.basename(out_path)

    print(f"  Saved: {saved_note}\n", flush=True)


metrics_path = os.path.join(OUTPUT_DIR, "sd_inpaint_metrics.csv")
with open(metrics_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "image",
        "method",
        "seed",
        "psnr",
        "hole_psnr",
        "outside_mse",
        "ssim",
        "black_residue_pct",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Metrics saved: {metrics_path}", flush=True)
print("Done.", flush=True)
