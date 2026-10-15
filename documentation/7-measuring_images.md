# Measuring images

[Guide overview](2-using_atracker.md)

Use `measure_images()` to draw lengths or outlines on images and save calibrated
measurements to CSV. It works without an ATracker project. The function was
previously named `batch_measure`; use `measure_images` in new and existing scripts.

## 1. Measure with a known scale

Put the images to measure in one folder. With a scale of 10 pixels per millimetre:

```python
from atracker import measure_images

measurements = measure_images("/path/to/images", px_per_mm=10,
    variables=["body_length", "body_height"], out_csv="measurements.csv")
```

Replace the scale and variable names with your own. Each image opens once per
variable, in sorted filename order. Images are selected from the immediate folder,
not recursively. Use `filetypes=("png", "jpg", "jpeg")` to restrict selection to
images; supported video files can also be opened for choosing a frame to measure.

`px_per_mm` is **pixels per millimetre**, the inverse of the `conv` value used for
tracking-data processing. Keep the same scale only if magnification and image
resizing are consistent across the images.

## 2. Draw the measurements

1. Click the start and end of a length, or add intermediate points along a curve.
2. Press **S** to save that measurement and move to the next variable or image.
3. For an area, draw an outline and return to the starting point before saving.

Length is the sum of consecutive segments in the drawn polyline. Area uses the
closed polygon and is recorded for outlines with at least four points. Explicitly
closing an outline also includes its closing segment in the saved length.
Place points consistently, and inspect the resulting CSV before measuring a large
collection. Cancelling a variable may leave a partial row; check that row before
resuming because existing filenames are skipped.

## 3. Calibrate interactively

If you do not know the pixel scale, draw a reference of known length:

```python
measurements = measure_images("/path/to/images", conversion_image_mm=50,
    conversion="first", skip_first=False,
    variables=["body_length", "body_height"], out_csv="measurements.csv")
```

Draw the 50-mm reference in the first image and press **S**. That calibration is
then used for every image. `skip_first=False` also measures the first image;
`skip_first=True` (the default) treats it as a calibration-only image.

If the scale changes between images:

```python
measurements = measure_images("/path/to/images", conversion_image_mm=50,
    conversion="all", variables=["body_length"], out_csv="measurements.csv")
```

This requests calibration for each image. Supplying `px_per_mm` bypasses
interactive calibration entirely.

## 4. Identify images and resume a session

Each result has the source filename in `file`. By default, `ID` also uses that
filename. To supply your own IDs:

```python
measurements = measure_images("/path/to/two_images", px_per_mm=10,
    ids=["F01", "F02"], variables=["body_length"], out_csv="measurements.csv")
```

The ID list must match the full sorted input-file list, including files already
measured. It does not prompt for IDs interactively.

Rerun the same call to resume: filenames already in `out_csv` are skipped and new
rows are appended. To remeasure an image, remove its existing CSV row first or
choose a different output CSV. Keep calibration settings explicit when resuming;
the measurement CSV does not reload them automatically.

## What is saved?

For `variables=["body_length"]`, output columns include:

| Column | Meaning |
| --- | --- |
| `ID` | Supplied identifier or source filename. |
| `file` | Source filename. |
| `body_length_length_px` | Drawn polyline length in pixels. |
| `body_length_length_mm` | Calibrated length in millimetres. |
| `body_length_area_px` | Polygon area in square pixels, when an area was drawn. |
| `body_length_area_mm` | Calibrated area in square millimetres. |

Area columns are omitted when no areas were measured in that session.
`measure_images()` returns a DataFrame of that session's new rows. `out_csv`
contains the accumulated results; omit it to return measurements without saving.
The drawn points themselves and calibration settings are not saved by this tool.

Use the [tracking editor](5-tracking_editor.md) for tracking-point correction or
candidate event review, or return to the [guide overview](2-using_atracker.md).
