"""Shared intensity correction and contour-limit resolution for detection."""

from functools import lru_cache
import cv2
import numpy as np


@lru_cache(maxsize=32)
def _gamma_lut(gamma):
    return np.round(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)


def gamma_correct(image, gamma=1.0):
    """Return an 8-bit gamma-corrected copy, or the input array for gamma 1."""
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Tracking gamma must be a finite positive number")
    image = np.asarray(image)
    return image if gamma == 1.0 else cv2.LUT(image, _gamma_lut(gamma))


def aspect_limits(settings, config=None):
    """Threshold-specific limits override project defaults, then 1.4 / 10."""
    limits = []
    for key, default in (("min_aspect_ratio", 1.4), ("max_aspect_ratio", 10.0)):
        fallback = getattr(config, key, None) if config is not None else None
        value = settings.get(key)
        limits.append(float(value if value is not None else fallback if fallback is not None else default))
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high:
        raise ValueError("Aspect-ratio limits must be finite and satisfy 0 <= min < max")
    return low, high
