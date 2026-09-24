# Tuning automated tracking

This guide explains how to tune detection before tracking a batch, which settings control each step, and how to recognise when a filter is removing real animals. There is no single best parameter set: resolution, animal size, posture, lighting and background all affect the result.

Use [manual tracking and correction](5-manual_tracking.md) for local corrections after tracking, and [processing tracking data](4-processing_data.md) for interpolation, smoothing and analysis. Those later operations cannot recover a fish that was consistently detected as the wrong object.

## 1. Prepare representative videos

Start with an ATracker project whose videos have been registered using your normal setup workflow. Check the background, ROI, masks, frame limits and object count before changing thresholds.

- **Background:** B/W detection assumes the animal is darker than the saved background. A fish captured in that background can leave a ghost or disappear when it overlaps the same location.
- **ROI:** include the area where the animal should be tracked. Shape and size filtering operate on image pixels inside this region.
- **Mask:** exclude known unusable areas rather than trying to remove them with increasingly restrictive shape thresholds. In B/W detection, dark mask pixels suppress detections. Zone overlays are annotations, not exclusion masks.
- **Frame limits and object count:** check the intended interval and number of animals in the overview.

Inspect several situations: normal swimming, turns, small and large apparent sizes, dark footage, proximity to a refuge, and lighting disturbances. A setting that works on one clear frame may fail throughout a turn.

## 2. Tune a named configuration in the Editor

```python
# Replace these indices with representative overview rows.
sample_inds = [0, 1, 2]
AT.editor(inds=sample_inds, purpose="thresholding", threshtypes=["bwdark"])
```

This loads `bwdark` if it exists, otherwise starts with defaults. The default new configuration is a starting point, not a recommendation for every video.

**Next / Prev keep your current threshold settings active**, including unsaved changes, for the same configuration name. Use that to compare several videos. **S / Save current** writes the named configuration into `AT.cfiles["threshinfo"]`; navigation itself does not save. **Close** prompts before discarding unsaved changes. Different configuration names retain separate session settings.

Saving updates every future tracking run that uses that configuration. It does not change any video's overview assignment.

### A useful tuning order

1. **Start with Tracking gamma = 1.00.** Adjust the threshold until the fish is visible in the detection preview with as little background noise as possible.
2. **Adjust blur and erosion sparingly.** Remove isolated noise while keeping the body connected. Inspect tails and small fish; aggressive erosion can erase them.
3. **For dark footage, compare increased tracking gamma with the original result.** Retune the threshold after changing gamma. Keep gamma only if detection improves across the representative frames.
4. **Set the area limits.** Include the observed range of real fish contours and exclude clearly smaller noise or larger background structures.
5. **Set aspect-ratio limits.** Include real postures, especially turns and curled fish. Check that rejected detections are actually unwanted.
6. **Save and run a short tracking trial.** Temporal checks such as flicker rejection, ID linking and the rolling size filter are not reproduced by the single-frame threshold preview.

Change one setting at a time when diagnosing a problem. Area and aspect-ratio limits cannot repair a fragmented fish contour: first improve the threshold image.

## 3. Understand the detection pipeline

### B/W configurations: names beginning with `bw`

The order is:

1. Apply tracking gamma to both the frame and background.
2. Subtract the frame from the background, clipping negative differences to zero. Only darker-than-background differences remain.
3. If enabled, check ROI-wide darkening for flicker rejection.
4. Apply the mask and convert the difference image to grayscale.
5. Apply Gaussian blur (`blur`), erosion (`erode`), and a second Gaussian blur (`blur2`).
6. Apply the binary `threshold`.
7. Extract contours and reject those outside the area or aspect-ratio limits.
8. Link detections to IDs; apply movement-jump and optional per-ID size checks.

The **threshold is applied to the processed background difference**, not directly to the original video's brightness.

### Colour configurations

Use `purpose="thresholding color"` for HSV detection. The tracker treats names not beginning with `bw` as colour configurations; use a supported colour name such as `red`.

Colour detection applies gamma, Gaussian blur, conversion to HSV, the colour bounds, and fixed erosion/dilation before extracting contours. It uses the same area and aspect-ratio limits. It does not use B/W background subtraction, the B/W `threshold`, `blur2`, configurable `erode`, or flicker rejection. The current colour branch also does not apply the B/W exclusion-mask operation.

HSV hue uses OpenCV's 0–179 scale; saturation and value use 0–255. The controls are `hue_lo`, `hue_hi`, `sat_lo`, `sat_hi`, `val_lo` and `val_hi`. A colour range that crosses the hue wraparound cannot be represented by a single ordinary low/high interval.

## 4. Threshold configuration settings

These values are stored per configuration in the threshold YAML file. The defaults below describe a newly created project-Editor configuration; existing configurations keep their saved values.

| Setting | New configuration | Meaning and tuning effect |
| --- | --- | --- |
| `gamma` | 1.0 | Values above 1 brighten shadows before subtraction/blur. Can also amplify noise; retune the threshold. The Editor offers 0.25–4.00. |
| `blur` | 9 | First Gaussian kernel width in pixels. Larger values smooth noise but also smear boundaries and can join nearby features. |
| `erode` | 1 | B/W erosion kernel width in pixels. Larger values suppress small bright features in the difference image and shrink detections. A width of 1 has no spatial erosion effect. |
| `blur2` | 1 | Second B/W Gaussian kernel width after erosion. A width of 1 has no spatial smoothing effect. |
| `threshold` | 50 | B/W cutoff on the processed 8-bit difference image. Lower admits weaker differences and more noise; higher may remove fish or fragment them. |
| `min_area` | 100 | Minimum contour area in pixels². Detections must be strictly larger. |
| `max_area` | 20000 | Maximum contour area in pixels². Detections must be strictly smaller. |
| `min_aspect_ratio` | Project default, otherwise 1.4 | Lower bound on rotated-box long side / short side. Detections must be strictly above it. |
| `max_aspect_ratio` | Project default, otherwise 10 | Upper bound on the same ratio. Detections must be strictly below it. |

Blur and erosion kernel widths are made positive and odd internally: an even value such as 4 becomes 5. These widths are not iteration counts. Area is measured from the extracted contour, so gamma, blur, erosion and threshold changes can all change its measured area.

The **Gamma** slider under Drawing functions is display-only. **Tracking gamma** in the threshold panel changes detection and is saved with the configuration. Neither changes source videos or stored background images.

### Aspect ratio and shape

Aspect ratio uses the minimum-area rotated bounding rectangle. A rectangle twice as long as it is wide has ratio 2, regardless of its orientation in the image. A compact or roughly round contour has a ratio near 1; a thin elongated contour has a high ratio. A degenerate contour with zero width is rejected.

The default strict interval `(1.4, 10)` rejects compact blobs and extremely thin shapes. It can also reject genuine fish during a tight turn, occlusion, or fragmented detection. Inspect those frames before tightening the interval. Lowering the minimum below 1 can admit approximately square contours; setting it to 1 still rejects a ratio exactly equal to 1.

Set project defaults with:

```python
AT.set_config(min_aspect_ratio=1.4, max_aspect_ratio=10.0)
```

The precedence is **explicit threshold configuration → project config → built-in defaults**. The Editor resolves the same values as tracking, exposes them in the threshold panel, and saves the displayed values as explicit configuration values. Consequently, changing the project default later does not override a configuration that already has those keys. Edit its limits in the Editor instead.

The preview and saved settings now preserve additional configuration fields rather than dropping them when a visible threshold control changes.

## 5. Reject shutter/lighting disturbances with `check_flicker`

```python
# Override for this run:
AT.track(inds=sample_inds, threshtype="bwdark", check_flicker=True)

# Or set the project default:
AT.set_config(check_flicker=True)
```

This option is off by default. It addresses sudden **darkening across much of the ROI** that could otherwise become a large false detection. It is not a general shutter correction, motion-blur correction or image reconstruction step.

Before tracking, ATracker samples up to 100 evenly spaced frames across the whole video. For each sample it gamma-corrects the cropped frame and background, calculates the positive background-minus-frame difference, converts it to grayscale, and takes its mean. The cutoff is **20 times the median of those sample means**. Sampling and detection use the same intensity correction and grayscale conversion. Configurations with different gamma values get separate cutoffs; equal gamma values reuse a cutoff. Sampling preserves the current video position.

During B/W tracking, a difference-image mean above that cutoff flags the threshold pass as flicker. That pass contributes no detections or linking updates for the frame. The image is not repaired, and a frame can still have detections from another configuration. Inspect the resulting coordinate gaps.

Important limits:

- The check is before masking and averages the entire ROI. Darkening in excluded areas can still trigger it; a small local disturbance may not.
- It targets darkening. A sudden bright flash may not be detected by background-minus-frame subtraction.
- Slow lighting drift or frequently disturbed footage can distort the sampled baseline.
- A median difference of zero produces a zero cutoff, which can be overly aggressive.
- Sampling uses the whole video, not just the selected tracking interval.
- HSV tracking does not use this check. The Editor threshold preview does not estimate or apply temporal flicker rejection.

The sample count (100) and multiplier (20) are currently implementation defaults in `estimate_flicker_baseline()`, not exposed arguments of `AT.track()` or `AT.set_config()`. The tracking log reports the cutoff and gamma. Inspect a short trial before enabling the option for a whole batch.

## 6. Per-ID rolling size filter

Global `min_area` / `max_area` reject implausible contours before linking. The optional rolling filter checks whether an already-linked detection has a plausible area **for that particular ID**.

```python
AT.set_config(
    size_filter=True,
    size_filter_tol=0.25,
    size_filter_memory=500,
)
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `size_filter` | False | Enable the per-ID rolling area check. |
| `size_filter_tol` | 0.25 | Accept area between `tol × median` and `median / tol`, inclusive. Use `0 < tol <= 1`. |
| `size_filter_memory` | 500 | Maximum accepted area observations retained per ID. Use an integer of at least 10 so the filter can finish its warmup. |

The check starts after **10 accepted observations** for an ID. Earlier observations pass this particular check. With the default tolerance, accepted area spans **25%–400%** of the median; a tolerance of 0.5 narrows it to **50%–200%**. Increasing the tolerance makes the check stricter. The history contains accepted observations, not elapsed video frames, so gaps and frame skipping change how much time the history spans.

Rejected detections receive missing centroid coordinates and do not update the area history. This occurs after ID linking; it is not a separate identity classifier. Advanced head/tail fields are stored separately and are not all cleared by the size check, so inspect centroid validity when interpreting those fields.

Use this filter when occasional detections have implausible areas relative to the animal. It cannot fix a consistently wrong contour, and a history built from wrong detections gives a wrong baseline. Tight tolerances can remove legitimate curled, partially occluded or changing-size animals. Tune the basic threshold image first.

The older names `shape_area_tol`, `shape_history_len`, and `contour_mode="dynamic"` are legacy configuration concepts. Use the three current settings above in new code.

## 7. Other tracking checks and outputs

| Setting | Default | Role |
| --- | --- | --- |
| `advanced` | False | Enables additional B/W head/tail, skeleton and shape calculations. It does not automatically add convexity or curvature rejection. |
| `max_framedist` | 200 | Live jump-filter allowance in pixels per elapsed frame. Implausible centroid jumps are rejected; movements between two positions near masked regions have special handling. |
| `skip_frames` | 0 | Skip frames during tracking. Nonzero values change temporal sampling and should be considered when assessing fast motion and filter history. |
| `create_vid` | Project config | Write a tracking video for review. |
| `show_tracking` | Project config | Show tracking live. |
| `draw_zones` | True | Draw zone fills and labels in tracking/visualisation output. Does not affect detection or zone processing. |

Area and aspect-ratio filtering operate in ordinary B/W and colour detection even when `advanced=False`. Advanced B/W mode computes quantities named inertia, convexity and curvature; the current detector does not apply rejection thresholds to those quantities. The variable called convexity is contour area divided by convex-hull area (often called solidity), not a direct curvature test. Experimental barcode handling is separate from the normal B/W/HSV workflow described here.

## 8. Validate a short run, then track the batch

```python
# bwdark must have been saved in the Editor first.
AT.set_config(create_vid=True, show_tracking=True, draw_zones=False)

# Use representative indices and an interval appropriate for your videos.
# A suffix keeps this trial's output separate from the usual output basename.
AT.track(
    inds=sample_inds,
    folder="originals",
    threshtype="bwdark",
    frame_start=1,
    frame_stop=500,
    suffix="tuning",
    overwrite=False,
    check_flicker=True,
)
```

`folder="originals"` explicitly selects the original videos; the default tracking folder is `todo`. Replace the example interval with frames present in your videos. Use a different suffix or explicitly choose overwrite if you need to rerun a trial. The example enables flicker rejection for evaluation; leave it off when it does not suit the footage.

Review real detections, false positives, identity continuity, contour sizes and missing centroids. Use the tracking overlay and final per-ID median area/aspect-ratio report to support inspection, not as proof that every frame is correct. The Editor coverage timeline can help locate gaps in the normal tracked CSVs; a suffixed trial CSV is a separate file from the default CSV that `AT.editor()` opens.

For the batch, either override the threshold name for that run:

```python
AT.track(inds=inds, threshtype="bwdark")
```

or save its assignment to selected overview rows:

```python
AT.update_overview("thresh_types", "bwdark", inds=inds)
AT.track(inds=inds)
```

The override uses singular **`threshtype`**; the Editor accepts plural **`threshtypes`**. Project settings live in `AT.cfiles["config"]`, named threshold parameters in `AT.cfiles["threshinfo"]`, and video assignments in the overview. Saving a named configuration does not assign it to videos.

## Troubleshooting

| Observation | First things to inspect |
| --- | --- |
| Dark fish is missing | Background contrast, threshold, then tracking gamma; inspect whether area/ratio filters reject the contour. |
| Fish breaks into small pieces | Threshold too high, excessive erosion, uneven contrast. Improve the binary image before lowering the area minimum. |
| Large background structures are detected | ROI/mask and background quality, then threshold and maximum area. |
| Fish disappears during turns | Minimum aspect ratio or area bounds may be too restrictive; the rolling size filter may also reject compact postures. |
| Many gaps appear after enabling size filtering | Inspect the initial history, tolerance and legitimate area variation. Compare against `size_filter=False`. |
| Whole-ROI dark disturbances produce false tracks | Evaluate `check_flicker=True`, inspect its logged cutoff and resulting gaps. |
| Changing project aspect defaults has no effect | Check for explicit aspect-ratio keys in the selected threshold configuration. |
| Threshold preview looks good but tracking loses points | Inspect temporal checks, ID linking, frame skipping and the actual configuration/assignment used for tracking. |
| Gamma makes the displayed video clearer but detections do not change | Make sure you changed **Tracking gamma**, not the display-only **Gamma** slider, and saved the configuration. |
