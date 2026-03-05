#! /usr/bin/env python

"""
barcoding.py

Functions to generate, visualise, save, and load barcodes for ATracker.

Supports:
- Generation with orientation and shape filtering
- Minimum pixel difference filtering
- Clean visualisation with IDs
- Saving and loading barcodes as compressed .npz files
"""

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

# ---- Generation utilities ----

def all_raw_tags(tag_shape):
    H, W = tag_shape
    for bits in itertools.product([0, 1], repeat=H*W):
        yield np.array(bits, bool).reshape(H, W)

def rotations(tag):
    """Return 4 rotations."""
    return [np.rot90(tag, k) for k in range(4)]

def canonical_rotation(tag):
    """Pick lex smallest rotation."""
    flats = [tuple(np.rot90(tag, k).flatten()) for k in range(4)]
    return np.array(min(flats), bool)

def shape_signature(tag):
    """Return black pixel pattern ignoring absolute position."""
    pos = np.array(np.where(~tag)).T
    if len(pos) == 0:
        return ()
    pos = pos - pos.min(axis=0)
    pos = sorted(tuple(p) for p in pos)
    return tuple(pos)

def black_count(tag):
    return (~tag).sum()

def sort_tags(tags):
    """Sort by number of black pixels and lex order."""
    return sorted(tags, key=lambda t: (black_count(t), tuple(t.flatten())))

# ---- Core functions ----

def generate_tags(tag_shape=(3,3), orient_strict=False, shape_strict=False, mindiff=None):
    """Generate tags with options."""
    tags = []
    seen_orients = set()

    raw_tags = list(all_raw_tags(tag_shape))
    print(f"Generated {len(raw_tags)} raw barcodes.")

    for raw in raw_tags:
        can = canonical_rotation(raw) if orient_strict else raw.flatten()
        key = tuple(can)
        if key in seen_orients:
            continue
        seen_orients.add(key)
        tags.append(raw)

    print(f"After orientation filtering: {len(tags)} barcodes.")

    if shape_strict:
        seen_shapes = set()
        tags2 = []
        for tag in tags:
            sig = shape_signature(tag)
            if sig not in seen_shapes:
                seen_shapes.add(sig)
                tags2.append(tag)
        tags = tags2
        print(f"After shape filtering: {len(tags)} barcodes.")

    if mindiff is not None and mindiff > 1:
        selected = []
        flats = []
        for tag in tags:
            flat = tag.flatten()
            if flats:
                D = pairwise_distances([flat], flats, metric="cityblock")
                if D.min() < mindiff:
                    continue
            selected.append(tag)
            flats.append(flat)
        tags = selected
        print(f"After mindiff filtering (≥{mindiff}): {len(tags)} barcodes.")

    tags = sort_tags(tags)

    # Add white border (1px)
    tags = [np.pad(tag, 1, constant_values=True) for tag in tags]

    ids = np.arange(len(tags))
    tags = np.array(tags, dtype=bool)

    return tags, ids

def visualise_tags(tags, ids=None, ncols=10, tile_size=1.0, pdf_path=None):
    """Visualise tags."""
    if ids is None:
        ids = np.arange(1, len(tags)+1)

    N = len(tags)
    nrows = math.ceil(N / ncols)
    figsize = (ncols * tile_size, nrows * tile_size)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=200)
    axes = np.array(axes, ndmin=1).flatten()

    for ax, tag, tid in zip(axes, tags, ids):
        h, w = tag.shape
        # Coordinates for each pixel edge
        X, Y = np.meshgrid(np.arange(w+1)-0.5, np.arange(h+1)-0.5)

        # Draw pixels with pcolormesh
        ax.pcolormesh(X, Y, tag.astype(int), cmap='gray', edgecolors='face', linewidth=0)

        # Draw crisp outer black frame
        ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h,
                                fill=False, edgecolor='black', linewidth=0.3))

        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)
        ax.set_aspect('equal')

        # Label
        ax.set_xlabel(f"{tid+1:03d}", fontsize=6)
        ax.xaxis.set_label_coords(0.5, -0.2)

    for ax in axes[N:]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.08)

    if pdf_path:
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def save_tags(filename, tags, ids):
    """Save tags + ids to npz."""
    np.savez_compressed(filename, tags=tags, ids=ids)

def load_tags(filename):
    """Load tags + ids from npz."""
    data = np.load(filename)
    return data['tags'], data['ids']
