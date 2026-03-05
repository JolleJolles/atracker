#! /usr/bin/env python3

import os
import pandas as pd
from pythutils.sysutils import lineprint
from atracker.visual_editor import annotation_gui

def batch_measure(
    folder=".",
    filetypes=("jpg", "png", "jpeg", "bmp", "tif", "tiff", "mp4", "avi"),
    ask_id=True,
    px_per_mm=None,
    conversion_image_mm=None,
    out_csv=None,
    ids=None,
    skip_first=True,
    conversion="first",
    variables=None,
):
    if variables is None:
        variables = ["SL_mm", "BH_mm", "eye_bottom_mm", "eye_top_mm"]

    def get_conversion_or_skip(file, label):
        """Handle calibration and return (status, px_per_mm)."""
        if conversion_image_mm is None:
            raise ValueError("conversion_image_mm must be provided for calibration.")
        lineprint(f"[{i+1}/{len(files)}] File: {basename} - Calibration! Draw line of {conversion_image_mm} mm...", newline=False)
        res = annotation_gui(media_file=file, mode="measure")
        if res == "exit":
            lineprint(" Canceled!")
            return "exit", None
        if isinstance(res, tuple):
            if res[0] == "length":
                length_px = res[1]
            elif res[0] == "polygon":
                length_px = res[2]
            else:
                lineprint(" Skipped!")
                return "skip", None
            conv = length_px / conversion_image_mm
            lineprint(f" Done! 1 mm = {conv:.1f} px")
            return "ok", conv
        else:
            lineprint(" Skipped!")
            return "skip", None
        
    def calculate_polygon_area(points):
        """Calculate the area of a polygon using the Shoelace formula."""
        n = len(points)
        if n < 4:
            return None
        area = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - y1 * x2
        return abs(area) / 2

    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(filetypes) and not f.startswith('.')]
    files = sorted(files)
    if not files:
        lineprint("No matching files found.")
        return

    already_done = set()
    if out_csv and os.path.exists(out_csv):
        try:
            df_existing = pd.read_csv(out_csv)
            already_done = set(df_existing['file'].values)
            lineprint(f"Found existing CSV with {len(already_done)} files. Skipping already processed files.")
        except Exception as e:
            lineprint(f"Could not read existing CSV: {e}")

    if ids is not None and len(ids) != len(files):
        raise ValueError(f"Length of IDs ({len(ids)}) does not match number of files ({len(files)})")

    results = []
    conv_global = px_per_mm
    per_file_calibration = (conversion == "all" and px_per_mm is None)
    area_columns_needed = False

    for i, file in enumerate(files):
        basename = os.path.basename(file)
        if basename in already_done:
            continue
        ID = ids[i] if ids is not None else basename

        # --- Determine the px-per-mm for this file ---
        if px_per_mm is not None:
            conv_this = px_per_mm
        elif per_file_calibration:
            status, conv_this = get_conversion_or_skip(file, label="per-file")
        else:
            if conv_global is None:
                status, conv_global = get_conversion_or_skip(file, label="first file")
                if status == "ok" and skip_first:
                    continue
            status, conv_this = "ok", conv_global

        if status == "exit":
            break
        elif status == "skip":
            continue

        # --- Measurement step ---
        result_row = {"ID": ID, "file": basename}
        for variable in variables:
            lineprint(f"Draw measurements for {variable}...", newline=False)
            res = annotation_gui(media_file=file, mode="measure")
            if isinstance(res, tuple) and res[0] == "polygon":
                points = res[1]
                length_px = res[2]
                length_mm = round(length_px / conv_this, 3) if conv_this else None
                area_px = calculate_polygon_area(points)
                area_mm = round(area_px / (conv_this ** 2), 3) if area_px and conv_this else None

                result_row[f"{variable}_length_px"] = length_px
                result_row[f"{variable}_length_mm"] = length_mm
                result_row[f"{variable}_area_px"] = area_px
                result_row[f"{variable}_area_mm"] = area_mm

                if area_px:
                    area_columns_needed = True
                print(f"Length: {int(length_mm)} mm, Area: {int(area_mm) if area_mm else 'N/A'} mm²")
            elif res == "exit" or res is None:
                print(" Canceled!")
                break
            else:
                print(" Skipped!")
                result_row[f"{variable}_length_px"] = None
                result_row[f"{variable}_length_mm"] = None
                result_row[f"{variable}_area_px"] = None
                result_row[f"{variable}_area_mm"] = None
        results.append(result_row)

    df = pd.DataFrame(results)
    if not area_columns_needed:
        df = df.drop(columns=[col for col in df.columns if "area" in col], errors="ignore")

    lineprint("All measurements complete.")
    if out_csv:
        if os.path.exists(out_csv):
            try:
                df_existing = pd.read_csv(out_csv)
                # Combine and drop duplicates by file name
                df_combined = pd.concat([df_existing, df], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset="file", keep="first")
                df_combined.to_csv(out_csv, index=False)
                lineprint(f"Appended new data and saved updated CSV: {out_csv}")
            except Exception as e:
                lineprint(f"Error updating existing CSV ({out_csv}): {e}")
        else:
            df.to_csv(out_csv, index=False)
            lineprint(f"Saved new CSV: {out_csv}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch manual measurement using PyQt annotation GUI")
    parser.add_argument("folder", nargs="?", default=".", help="Folder with images/videos")
    parser.add_argument("--px_per_mm", type=float, help="Pixels per mm (skip calibration step)")
    parser.add_argument("--calib_mm", type=float, default=50.0, help="Known mm to use in calibration image")
    parser.add_argument("--noid", action="store_true", help="Do NOT ask for user ID (just use filename)")
    parser.add_argument("--csv", type=str, default="lengthdata.csv", help="Output CSV file")
    parser.add_argument("--measure-first", dest="skip_first", action="store_false",
                        help="Also measure the first file used for calibration (conversion='first').")
    parser.add_argument("--conversion", choices=["first", "all"], default="first",
                        help="Calibration mode: 'first' (one-time) or 'all' (per file).")
    args = parser.parse_args()

    batch_measure(
        folder=args.folder,
        px_per_mm=args.px_per_mm,
        conversion_image_mm=args.calib_mm,
        ask_id=not args.noid,
        out_csv=args.csv,
        skip_first=args.skip_first,
        conversion=args.conversion,
    )
