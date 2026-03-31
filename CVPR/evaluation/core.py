import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_PYIQA_NIQE = None
_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"

if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def resize_rgb(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    # ITU-R BT.601 luminance approximation.
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def compute_eme(gray: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    h, w = gray.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    if blocks_h == 0 or blocks_w == 0:
        return float("nan")

    total = 0.0
    count = 0
    for i in range(blocks_h):
        for j in range(blocks_w):
            y0 = i * block_size
            x0 = j * block_size
            block = gray[y0:y0 + block_size, x0:x0 + block_size]
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            total += math.log((bmax + eps) / (bmin + eps))
            count += 1

    return (2.0 / count) * total if count > 0 else float("nan")


def _downsample_gray(gray: np.ndarray, max_side: int = 50) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return gray
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _ensure_scipy_misc_imresize() -> None:
    try:
        import scipy.misc as misc  # type: ignore
    except Exception:
        return

    if hasattr(misc, "imresize"):
        return

    def _imresize(arr: np.ndarray, size: Tuple[int, int], interp: str = "bilinear", mode: Optional[str] = None) -> np.ndarray:
        if isinstance(size, tuple):
            out_h, out_w = size
        elif isinstance(size, (int, float)):
            scale = float(size)
            if scale > 10.0:
                scale /= 100.0
            out_h = max(1, int(round(arr.shape[0] * scale)))
            out_w = max(1, int(round(arr.shape[1] * scale)))
        else:
            raise TypeError("scipy.misc.imresize compatibility shim expects a tuple or scalar scale")

        if mode == "F":
            work = arr.astype(np.float32)
        elif arr.dtype != np.uint8:
            work = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            work = arr

        resample_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resample = resample_map.get(interp, Image.BILINEAR)
        pil = Image.fromarray(work, mode=mode)
        resized = pil.resize((out_w, out_h), resample=resample)
        result = np.asarray(resized)
        if mode == "F":
            return result.astype(np.float32)
        return result

    misc.imresize = _imresize  # type: ignore[attr-defined]


def compute_loe(input_gray: np.ndarray, enhanced_gray: np.ndarray, max_side: int = 50) -> float:
    a = _downsample_gray(input_gray, max_side=max_side).reshape(-1)
    b = _downsample_gray(enhanced_gray, max_side=max_side).reshape(-1)
    if a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

    ord_a = a[:, None] >= a[None, :]
    ord_b = b[:, None] >= b[None, :]
    diff = ord_a != ord_b
    return float(np.mean(np.sum(diff, axis=1)))


def try_compute_niqe(gray: np.ndarray) -> Tuple[float, Optional[str], Optional[str]]:
    global _PYIQA_NIQE

    # Prefer pyiqa implementation for compatibility and stability.
    try:
        import torch
        import pyiqa  # type: ignore

        if _PYIQA_NIQE is None:
            _PYIQA_NIQE = pyiqa.create_metric('niqe', device='cpu')

        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
        score = float(_PYIQA_NIQE(x).item())
        return score, "pyiqa", None
    except Exception:
        pass

    # Prefer PIQ implementation for compatibility with modern SciPy.
    try:
        import torch
        import piq  # type: ignore

        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
        score = float(piq.niqe(x, data_range=1.0).item())
        return score, "piq", None
    except Exception:
        pass

    # Prefer skimage implementation when available.
    try:
        from skimage.metrics import niqe as skimage_niqe  # type: ignore

        score = float(skimage_niqe((gray * 255.0).astype(np.float32)))
        return score, "skimage", None
    except Exception:
        pass

    # Fallback to skvideo implementation.
    try:
        _ensure_scipy_misc_imresize()
        from skvideo.measure import niqe as skvideo_niqe  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency optional
        return float("nan"), None, f"NIQE unavailable: {exc}"

    try:
        # skvideo expects image in [0, 255] or [0, 1] float; use [0,255] for stability.
        score = float(skvideo_niqe((gray * 255.0).astype(np.float32)))
        return score, "skvideo", None
    except Exception as exc:
        return float("nan"), None, f"NIQE failed: {exc}"
