#! /usr/bin/env python

import os
import json

PREFS_PATH = os.path.expanduser("~/.atracker/editor_prefs.json")

DEFAULTS = {
    "drawing_mode": "rectangle",
    "hue": 120,
    "image_transparency": 100,
    "point_opacity": 100,
    "mask_opacity": 80,
    "point_size": 6,
    "blackwhite": False,
    "show_mask": False,
    "invert_mask": False,
    "show_zones": False,
    "helperlines": False,
    "crosshair": False,
    "show_loop": False,
    "show_points": True,
    "show_lines": False,
    "show_framenrs": False,
    "show_nearframe": False,
    "show_arrows": False,
    "show_angle_lines": False,
    "arrow_length": 20,
    "visible_range": 100,
    "scope_mode": "any",
    "edit_mode": "draw",
    "point_type_idx": 0,
    "highlight_current": True,
    "range_direction": "all",
    "fade_points": False,
    "fps": 25,
}


def load_prefs():
    try:
        with open(PREFS_PATH) as f:
            data = json.load(f)
        return {**DEFAULTS, **data}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(DEFAULTS)


def save_prefs(prefs):
    try:
        os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)
        with open(PREFS_PATH, "w") as f:
            json.dump(prefs, f, indent=2)
    except OSError:
        pass
