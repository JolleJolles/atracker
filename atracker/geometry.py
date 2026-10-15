"""Portable, full-frame geometry exports from saved project images."""
import ast
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _polygons(binary, epsilon):
    """Represent disconnected regions, holes and nested islands without flattening."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    tree = hierarchy[0]

    def ring(index):
        contour = contours[index]
        if epsilon:
            contour = cv2.approxPolyDP(contour, epsilon, True)
        coords = contour.reshape(-1, 2).tolist()
        if len(coords) < 3 or cv2.contourArea(contour) == 0:
            return None
        return coords + [coords[0]]

    polygons = []
    for index in range(len(contours)):
        depth, parent = 0, tree[index][3]
        while parent >= 0:
            depth += 1
            parent = tree[parent][3]
        if depth % 2:
            continue
        exterior = ring(index)
        if exterior is None:
            continue
        rings = [exterior]
        child = tree[index][2]
        while child >= 0:
            interior = ring(child)
            if interior is not None:
                rings.append(interior)
            child = tree[child][0]
        polygons.append(rings)
    return polygons


def _number(value):
    return None if value is None or pd.isna(value) else float(value)


def export_geometry(project, names=None, inds=None, query=None, cats=None,
                    output=None, zone_names=None, epsilon=0, overwrite=False):
    """Export one geometry JSON per selected overview row; return its file paths.

    Files are GeoJSON FeatureCollections with pixel-coordinate metadata. Mask,
    wall and mask-zone dark regions use a <=10 grayscale threshold, without
    processing erosion. Zones use the processing palette's hue/saturation/value
    thresholds. epsilon optionally simplifies contours, in pixels (0 preserves
    raster outlines). Full-frame images must match the overview resolution.
    zone_names maps numeric palette IDs to labels; otherwise names are zone_1,
    zone_2, etc. Original images remain the tracking/processing inputs.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be non-negative pixels")
    if names is not None:
        inds = project.get_inds(names)
    selected, _ = project.get_files(inds=inds, query=query, cats=cats)
    originals = Path(project.dirs["originals"])
    destination = Path(output).expanduser() if output else originals
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in selected:
        row = project.overview.loc[index]
        video = str(row.video)
        region = row.get("region")
        multiple = (project.overview.video == video).sum() > 1
        if multiple and (region is None or pd.isna(region)):
            raise ValueError(f"Missing region for overview row {index}")
        basename = f"{video}_R{int(region)}" if multiple else video
        path = destination / f"{basename}.geometry.json"
        if path.exists() and not overwrite:
            print(f"Existing geometry retained: {path}")
            paths.append(str(path))
            continue
        resolution = row.get("resolution")
        if isinstance(resolution, str):
            try:
                resolution = ast.literal_eval(resolution)
            except (ValueError, SyntaxError):
                match = re.fullmatch(r"(\d+)x(\d+)", resolution.strip())
                if not match:
                    raise ValueError(f"Invalid resolution for {video}: {resolution}")
                resolution = tuple(map(int, match.groups()))
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError(f"Missing width/height resolution for {video}")
        width, height = map(int, resolution)
        roi = row.get("roi")
        if isinstance(roi, str) and roi.strip():
            roi = ast.literal_eval(roi)
        if not isinstance(roi, (tuple, list)):
            roi = ((0, 0), (width, height))
        (x0, y0), (x1, y1) = roi
        features = [dict(type="Feature", properties=dict(kind="roi", name="roi"),
                         geometry=dict(type="Polygon", coordinates=[[
                             [int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)],
                             [int(x0), int(y1)], [int(x0), int(y0)]]]))]
        source_images = {}
        for column, kind in (("bgimg", "background"), ("maskimg", "mask"),
                             ("wallimg", "walls"), ("maskzoneimg", "maskzone"), ("zoneimg", "zone")):
            filename = row.get(column)
            if not isinstance(filename, str) or not filename.strip():
                continue
            source = originals / filename
            source_images[kind] = os.path.relpath(source, path.parent)
            if kind == "background":
                continue
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Cannot read {kind} image: {source}")
            if image.shape[:2] != (height, width):
                raise ValueError(f"{source} must be full-frame ({width}x{height})")
            if kind == "zone":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                regions = [(zone_id, cv2.inRange(hsv, np.array([(hue-20)//2, 200, 200]),
                            np.array([(hue+20)//2, 255, 255])))
                           for zone_id, hue in enumerate(range(0, 360, 36), 1)]
            else:
                regions = [(None, (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) <= 10).astype(np.uint8) * 255)]
            for zone_id, binary in regions:
                polygons = _polygons(binary, epsilon)
                if not polygons:
                    continue
                properties = dict(kind=kind, name=kind, source=source_images[kind])
                if zone_id is not None:
                    properties.update(zone_id=zone_id,
                        name=(zone_names or {}).get(zone_id, f"zone_{zone_id}"))
                features.append(dict(type="Feature", properties=properties,
                                     geometry=dict(type="MultiPolygon", coordinates=polygons)))
        associated = {}
        for key, directory, suffix in (("tracked_csv", "tracked", ".csv"),
                                        ("tracking_video", "tracked", "_TR.mp4"),
                                        ("processed_csv", "processed", "_F.csv")):
            associated[key] = os.path.relpath(Path(project.dirs[directory]) / (basename+suffix), path.parent)
        document = dict(type="FeatureCollection", schema_version=1, video=video,
            region=None if region is None or pd.isna(region) else int(region),
            image=dict(width=width, height=height),
            coordinate_system=dict(units="pixels", origin="top_left", x="right", y="down",
                reference="full_frame", contour_coordinates="pixel_centres",
                roi_bounds="crop bounds; right and bottom are exclusive"),
            calibration=dict(mm_per_pixel=_number(row.get("conv"))),
            tracking=dict(fps=_number(row.get("fps")), start_frame=_number(row.get("frame_start")),
                          end_frame=_number(row.get("frame_stop")), objects=_number(row.get("objects"))),
            provenance=dict(method="raster_contours", simplification_pixels=epsilon,
                dark_threshold=10, processing_erosion_applied=False),
            source_images=source_images, associated_files=associated, features=features)
        path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        paths.append(str(path))
        print(f"Geometry saved: {path}")
    return paths
