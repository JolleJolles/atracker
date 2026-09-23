# Changelog

## TODO
- Perhaps besides storing the default tracking file and tracked video for each file, we could store an associated file that has coord data of all relevant files for that video, such as of the mask, the walls, and zones.  


## Unreleased
- Added an imageio/FFmpeg H.264 fallback when OpenCV cannot open visualisation output, with encoder errors reporting the target path and video resources released on rendering errors.
- Aligned full-frame backgrounds and wall/zone overlays with cropped, resized visualisation videos.
- Padded odd-sized visualisation frames to even MP4 dimensions (for example, a half-size ROI of 316 × 485 becomes 316 × 486) and added a frame-size check to prevent silent writer mismatches.
- Included video dimensions, FPS and OpenCV version in visualisation writer errors to help diagnose environment-specific output failures.
- Made visualisation report the absolute output path or explicitly state that saving is disabled, and raise an error if the video writer fails to open or no output is written.
- Fixed overlay opacity accumulating where pending zone shapes overlap; zones/previews and mask outlines are now composited before applying the shared display opacity.
- Fixed pandas `SettingWithCopyWarning` during processed-output rounding by explicitly copying the selected output columns.
- Fixed tracked points and point-editing controls not appearing/working in the multi-file editor's "Coordinate data" mode; moved filenames above navigation buttons and added full-name tooltips/window titles.
- Fixed signed ROI-distance comparisons so `roi_edge_margin` targets positions near/outside the boundary instead of the arena interior, for centroid gap handling and head/tail filtering. Short gaps still join according to `max_traj_gap`.
- Clarified the processing guide with ATracker-native file selection, a complete default settings dictionary and explanations of every `process()` parameter.
- Split manual tracking into its own guide and rewrote processing documentation around the current API, with runnable examples, parameter-selection advice and output-unit caveats.
- Fixed the interactive editor resetting the last-used drawing tool when opening the next video; window size, position and maximized/fullscreen state are now saved with personal editor preferences.
- Extended the mask opacity slider to zones and pending zone previews, and renamed it "Mask / Zones Opacity".
- Fixed adding zones to saved images to select an unused palette colour; saving now colours pending shapes automatically instead of storing dark previews.
- Fixed saved zones images appearing blank when reopened with `set_interactive(zones=True)` by converting loaded RGB images to the canvas's ARGB format.
- Video frames are now written in a separate thread via AsyncVideoWriter, preventing video writing from blocking the tracking loop.
- Fixed race condition where shared Tracker instance attributes were overwritten by concurrent workers, causing missing or misplaced output files and CSVs.
- Added skip_frames parameter to track(). Set to 0 (default) to process every frame, or higher to skip frames and increase tracking speed (e.g. skip_frames=1 processes every 2nd frame).
- Fixed infinite loop where already-tracked files were not removed from the job queue before pooled tracking started.
- Reworked `filter_tracking_jumps` mask handling: jumps between two positions both near a masked region are now accepted unconditionally (fish traversed under mask), replacing the previous straight-line path-crossing check which failed when both endpoints were on the same side of the mask.
- Fixed `filter_tracking_jumps` to use filtered (post-jump-check) positions when updating `movedat`, preventing noise contours rejected by the jump filter from overwriting the ID's reference position and causing persistent ID mis-assignment in subsequent frames.
- Added `contour_mode` setting to `set_config()`. Set to `"dynamic"` to enable a per-ID rolling area consistency filter that rejects detections deviating strongly (below 25% or above 400%) from the ID's rolling median area, suppressing noise contours that pass the global thresholds but are inconsistent with the tracked object's history.
- Tracking completion now prints per-ID median area and aspect ratio for accepted detections.
- Replaced O(N²) trajectory drawing with per-ID deques: drawing now reads from a fixed-length deque instead of scanning all accumulated fulldat, eliminating a bottleneck that grew throughout long videos.
- Removed `sys.stdout.flush()` from the per-frame threshold loop.
- Cached all drawing color values (`eval()`, `namedcols()`) before the tracking loop so they are computed once per video instead of once per frame.
- Eliminated redundant `cv2.contourArea` call per contour: areas are now computed once during sorting in `process()` and passed directly to `processcon()`.
- Fixed mask shape comparison in `preprocess_bw_mode` to compare spatial dimensions only, avoiding a no-op `cv2.resize` call every frame.
- Updated `processcon` function to get the aspect ratio of the rotated rectangle and always long side divided by short side, and set better default parameters for tracking fish
- Fixed checking of firstframe and lastframe columns of the overview to deal with empty strings
- When drawing rectangle in roi, mask, or measure mode in `set_interactive` mode, it now also shows the surface area.
- Added `__citation__` to `__init__.py`.
- Updated `README.md` with zenodo doi and moved citation to the top.
- Cleaned `CHANGELOG.md` structure and standardize dates
- Split monolithic `utils.py` into eight focused modules: `geometry.py` (spatial math), `contour_utils.py` (contour operations), `angles.py` (angle/orientation math), `trajectory.py` (movement analysis), `tracking_filters.py` (real-time filters), `media.py` (video/image I/O), `qt_utils.py` (Qt/screen utilities), and `data_utils.py` (data handling). `utils.py` is now a thin backward-compatibility shim so all existing code continues to work unchanged.
- Replaced `from .utils import *` in `tracker.py` and `process_image.py` with explicit imports from the new specialized modules for cleaner dependency tracking.
- Deleted obsolete `tracker_man.py` (manual tracking superseded by `visual_editor.py` / `manual_tracker()` wrapper).
- Renamed `processor.py` to `post_processor.py` to clearly distinguish post-processing of tracked data from image processing; `__init__.py` updated accordingly.
- Rewrote `__init__.py` with explicit named exports (`ATracker`, `manual_tracker`, `batch_measure`, `Processor`, `__citation__`) instead of star imports.
- Fixed `warp_barcode_patch` being defined inside the `ProcessImage` class body instead of as a module-level function, which would have caused a `NameError` at runtime.
- Fixed "colname does not exist" message printed during tracking: `namedcols()` was called for `"bw"` threshold types which are not color names; these are now correctly skipped when building the per-threshtype color lookup.
- Fixed OpenCV `rectangle` crash (`Can't parse 'pt2'. Sequence item with index 1 has a wrong type`) in the tracking overlay caused by `pythutils.draw_text` returning float pixel coordinates; replaced all per-frame overlay label drawing with direct `cv2.getTextSize` / `cv2.rectangle` / `cv2.putText` calls using explicit integer arithmetic.
- Added `drawinfobox` option to the tracking overlay and `visualiser`: when enabled, draws a compact white infobox in the top-left corner showing frame number and per-ID area and aspect ratio, replacing the previous fragmented per-line labels. The standalone frame-number label is suppressed automatically when the infobox is active.

## v1.0.0 — 2026-03-05 
First public release of ATracker.
- Updated the package tagline and description, including in `setup.py`.
- Updated `README.md` file with further explanation and citation section.
- Added `CITATION.cff` file to provide information how to cite the pacage.
- Added `LICENSE` file with the Apache licence text.
- Removed the temp and tutorial video folders to facilitate release

---

## Pre-release development history

### 2025-12-09 
- Fixed the dist_to_rect function to correctly compute distance measures when coordinates never overlap with a zone
- Fixed orientation pipeline: head coordinates (hx, hy) are now correctly mapped to fx, fy and used for orientation when orientfrombw = False and clarified orientation modes: orientfrombw = True now consistently triggers angle-based orientation and removes head-coordinate columns (fx, fy).

### 2025-11-27 
- Updated `get_inds` function to work with regions as well.
- Updated `check_interactive` to load correct datafile when working with regions.
- Updated the `get_bgfiles` function to work correctly with regions.
- Fixed `annotation_gui` to be able to load correctly with empty datafiles.
- Fixed `prep` processor function to remove any data from the file beyond the start and stopframes.
- Added `force_single_traj` option to trajectories processor to fix data into single trajectories.

### 2025-10-20 
- Fixed batch_measure function to append to file if it exists rather than skipping the files already done but still overwriting the file.

### 2025-09-17 
- Fixed the `load_and_convert_tracking_dataframe` function to also load properly new files without detailed orientation data.
- Set "highlight current" checkbox of the visual_editor to True to by default show the current locations.
- Fix default trajectory display when switching to "Current ID" mode: force-set tp_current_id = 0 to show ID1 correctly.
- Add support for undoing multi-point deletions by updating undoLastDelete() to handle actions-based undo buffer format.
- Fixed the loop not showing in visual_editor.

### 2025-09-12 
- Added an "event" mode to the `set_interactive` function, enabling the user to set a list of tuple pairs of event changes using the "k" key and to remove the last keypress with the "l" key, automatically stored in the AT.overview file.
- Fixed "measure" mode accordingly so it would work with the updated results output as input.

### 2025-09-12 
- Improved the `get_filepart` utils function to provide specific file extensions.
- Fixed `batch_measure` function not escaping fully when pressing the escape key.
- Enabled `batch_measure` to load and append to file, if it exists.
- Added deleting of last drawn point functionality using the `d` key to the visual editor for measuring mode.
- Updated `batch_measure` to store area information of polygon if more than 3 points were drawn.
- Improved `bath_measure` with option of user to provide a list of variables to measure.

### 2025-08-25 
- Improved `batch_measure` function with "skip_first" parameter to also include the first image and "conversion" parameter to allow user to only use first or all images for conversion.
- Added `get_filepart` utils function to quickly get a list of parts of filenames.
- Updated the `smooth` utils function to ensure target columns are float dtype.
- Improvedd the `processor` script to correctly stop post-processing when "emptydat" is True.

### 2025-08-01 
- Fixed date column of AT.overview being automatically converted to date with added time and then onverted to string anyway.
- Added overwrite parameter to the `track` function that overrules the same parameter in the config file, for easier changing of the setting.
- Updated the `showinfo` function with "return_inds" parameter to be able to return a list of indices for easy subsequent further use.
- Added `update_overview` function to the ATracker class for easy updating of the overview file.
- Addded point opacity slider to the `set_interative` function to make it easier to fix tracking data by still being able to see the tracked object behind the tracking data.
- Fixed timepoints mode to enforce one point per ID per frame; clicking now moves the point instead of adding new ones.
- Improved undo support for moved and edited points in timepoints mode.
- Fixed some issues that had come up due to the new integrations with `check_interactive` and `filter_tracking_jumps` to make sure it also works across tracking multiple videos.
 
### 2025-07-31 
- Fixed that video out of tracking uses the fps as provided in the `fps` column of the fileinfo extraction of AT.overview
- Implemented a `filter_tracking_jumps` utils function that we call in the main `tracker.py` file to on-the-fly exclude tracking of contours that are beyond a set threshold distance measure that grows with each frame. This can be set with the `max_framedist` parameter in the `track` function.
- Fixed potential errors for drawing trajectories due to nan's in the data in the `tracker.py` file.
- Added a new main `check_interactive` functionality to the ATracker class with which one can now load files via inds and names like the other ATracker functions and thereby interactively manipulate tracking data stored in tracking files in the "3tracked" folder while automatically the original accompanying video is displayed with roi cropped out.

### 2025-07-24 
- Fixed wrong key explanation for zones drawing (from 'o' key to 'z' key).
- Fixed custom treshtypes not defaulting to normal tresholding in set_interactive when starting with "bw".
- Added note in the guide and tutorial about providing custom treshnames and custom treshfiles.
- Fixed tracking not conclusively being able to work with custom treshtypes starting with "bw"

### 2025-07-22 
- Included computing point distance computations in the `processor` function file, which was apparently missing after all considerable revisions of the function.
- Fixed conversion accidentally removing roi[0] again while tracking was already based on that, so updated the `convert` utils function with parameter `already_relative`.
- Unified all distance computations by shifting cx, cy to full-image coordinates (xs_full, ys_full) before calling dist_to_*()
- Ensured roi, mask, wall, zone, and point distances are now computed in consistent coordinate space

### 2025-07-16 
- Fixed spelling of tresh** throughout the package (to thresh**)

### 2025-07-15 
- Fixed mask not being drawn always when using the set_interactive mode for thresholding.
- Improved thresholding mode in set_interactive to take into account the mask.
- Fixed overlay semi-transparency such that contours are clearly visible.
- Fixed saving and escaping in fullscreen going back to normal window size rather than exiting the current instance.
 
### 2025-07-14 
- Updated the `showinfo` and `get_files` functions. Former now displays the AT.overview subset based on the indices or filenames provided and the latter now has parameter to hide extension or not.
- Fixed in zones mode accidentally the background image being used as zone default rather than a white background.s
- For zones mode now by default the zones mask is being displayed.
- For thresholding bars in the visual_editor now also added spinbox so the user can type in a value rather than only using the scrollbar, especially useful for small values on bars with large possible range.
- Fixed the color threshold bars being shown for normal thresholding and all threshold bars being shown for color thresholding.
- Updated the installation guide with optional installation of virtualenvwrapper.

### 2025-07-11 
- Updated `using_atracker` tutorial to better explain users can update the `threshinfo.yml` file manually if needed.
- Fixed issue with naming of zone and wallimg columns in the overview.
- Added new `compute_zone_distances_auto` utils function that uses three different approaches to calculate the distance to a zone, and depending on its complexity chooses the fastest one. It always uses the conv conversion.
- Improved and streamlined the various functionalities for computing distances to points, rectangles, polygons and masks.
- Made considerable improvements to the `processor` file, particularly focused on the distance computations, loading of masks, streamlining logic for roi,mask,walls,zones-related calculations, and using converted and flipped data.

### 2025-07-09 
- `setup_files` now populates the overview for videos even if conversion to .mp4 is not performed (autoconvert=False), helpful to set variables such as fps already.  
- Improved file setup to automatically handle video filenames with or without version suffixes (_vXX), adding a vidseq column to the overview for the version if present. Users don't need to specify a version field in fname_vars; version detection and overview handling are automatic.
- updated the using_atracker guide to have a brief mention one can use the AT.save() command to save the overview to disk when the user updated in Python for example.
- updated the convert_h264_to_mp4 function to have as input a list of fps values

### 2025-07-08 
- Added a poolatracker.py file to the tutorial folder and some further explanation to the tracking jupyter tutorial file how to use the pool. 
- Measure mode now supports polyline measurement: users can click multiple times to define a path, and the total length is calculated on Save (S key). The drawing mode selector is automatically disabled in Measure mode to prevent confusion.
- Support for a new ids parameter in batch_measure, allowing users to supply a list of custom IDs (same length as the file list). Each result row will use these IDs if provided.

### 2025-06-28  
Updated the processing function to efficiently compute distances to each zone when a zones mask is provided and name and round them correctly.
- Added coordsfromzones utils function to efficiently get contours from colors on zones image.
- Updated the add_kdtree_distances utils function to also work with multiple contours and provide parameter to add coordinates of the nearest point on the mask/zone contour. 
- Integrated that zone distance is 0 when the object is inside the zone, which doesnt work with kdtree as distance is always positive.
- Made sure the same nr of columns with "z" followed by a numeric are added as there are zones and these come after the mdist and rdist columns and rounded to 1 decimal.

### 2025-06-27  
- Fixed small issue with "total_frames" missing in set_interactive.
- Added possibility to show/hide a loop that shows a zoomed-in area at the mouse pointer.
- Reorganised the keyPressEvent function of the visual_editor and added new keys to change mode (U-key) and to go up and down IDs (I and O keys).
- Fixed qwerty keys to not function when working with image.
- Fixed the ID number and frame number for showing 1 too many using the "near" toggle.
- Added generate_distinct_colors utils function, which will be used for unique color drawing of IDs.
- Improved showing ID colors as a list of colored squares, as many as there are IDs, and integrated ability for user to click IDs when "Current ID" is selected, which facilitates quickly jumping between IDs. The list will expand and push the other interface options down when needed.
- Fixed angles in visual_editor to be stored in the right coordinate system.
- Improved the draw_arrow_head utils function to be able to draw it also at the origin
- Added a scrollbar to dynamically change the length of the arrow line being shown, so arrow heads can for example also be shown at the origin.
- Moved position of Highlight current frame to next to mode selector and unchecked as default.

### 2025-06-18 
- Fixed set_interactive mode to now store information to allinds based on the query and category
- Integrated a much improved load (old) tracking data function and a separate function to convert it to dictionaries to work with visual_editor
- Made various improvements to visual_editor to be able to work with older, complex files, such as with mixed IDs

### 2025-06-17
- Switched arrowhead angle logic to fix direction rendering near 0 degrees.
- Changed visible range indexing to start at 0 to allow showing just the current frame.
- Adjusted point filtering logic to correctly include current frame in visible range.
- Made frame numbers visible even when points are hidden.
- Added background transparency slider below “Drawing Hue” and renamed section to “Drawing functions”.
- Modified paintEvent to draw the current frame’s point and arrow thicker in black for emphasis.
- Scoped point and angle display to current ID if selector is active, using shared scope_mode.
- Ensured scope_mode and tp_current_id are synced to the drawing widget.
- Fixed logic in tracker.py to increment point IDs by 1 so they align with GUI display
- Moved the "Highlight current frame" checkbox inside the Timepoints group (below Angles), and connected it so toggling instantly updates the drawing area.
- Make editing of angle data possibly by mouse press events
- Updated saving of timepoints to make it work again now with all point types and angle.
- Integrated ability for user to load datafile without mediafile and set width and height in annotation_gui and manual_tracker.
  
### 2025-06-13 
- Removed the process.py file, which was long time ago overwritten by the expanded imgprocessor.py file, which was now renamed to image_process.py.
- Fixed image processing to wrongly store trail and icom columns as numpy
- Removed the "icom" variable everywhere and the centroid based on the skeleton is now written directly to com to simplify things.
- Tracking and overlay output now use separate cx,cy, hx,hy, and tx,ty columns instead of tuple-based com, head, and tail for clearer downstream processing and video rendering.
- Included load_tracking_data function in visual_editor to be able to load and work with tracking data from ATracker.
- Fixed showing info of nearest point of any ID in visual editor.
- Integrated ability to load and work with head,tail,angle data in the visual_editor.
- Integrated numerous improvements in the timepoints options in the sidebar of the visual_editor.
- Reorganised and cleaned the code for the whole drawing interface with the visual_editor.
- Integrated drawing of arrows and in colors based on the angle value in the visual_editor.

### 2025-06-10
Made various improvements to the visual-editor:
- Added ability to crop (delete) points inside any temporary shape (rectangle, polygon, circle, ellipse) in timepoints mode without needing to press “A”; shape remains for repeated use. Also Improved polygon shape finalization and display logic for cropping/deleting.
- Improved undo logic: avoids KeyError when undo buffer is empty or called multiple times.
- Enhanced nearest frame indicator to show both ID and frame number (e.g., #4-112) for all IDs, with a clear white outline for readability.
- Fixed: “Current ID” selector now properly syncs with the number of IDs, disables invalid selections, and never allows selection above maximum.
- Integrated “Move points” with radio buttons with option to select “None", “Any ID” and “Current ID”.
- Improved ID color display: supports up to 20 IDs, displays color squares above visible range slider, and highlights current ID.
- Improved logic so points from IDs above current NR IDs are hidden but not deleted.
- Added option to draw helperlines across the screen with both h-key and checkbox in the functions panel
- Added checkbox to toggle visibility of the crosshair in the functions panel
- General UI/UX improvements throughout timepoints mode, including consistent color coding and clearer text rendering.
- Removed the ivideo.py file as became obsolete as all functionalities have been overtaken and improved with PyQT in the visual_editor.py file.

### 2025-06-10
- Added batch_measure.py script for efficient batch manual measurement of image and video files via the ATracker annotation GUI, with results saved to CSV or returned as a pandas DataFrame.

### 2025-06-10
- Updated the installation guide further to work better with virtual environments on Mac and resolves potential issues with MVS and kernels
- Updated media type detection by introducing a dedicated get_media_type function, ensuring images and videos are handled appropriately and preventing OpenCV video stream errors for image files.
- Updated the using_atracker Jupyter tutorial to include new documentation and guidance on the batch measurement workflow and related improvements.

### 2025-06-02 v0.1.6
- Updated installation guide with platform-specific Python setup, virtual environments, and ffmpeg installation instructions.
- Added hybrid .h264 → .mp4 conversion with automatic fallback from ffmpeg to imageio for full portability.
- Improved path handling in ATracker() to support drag-and-drop and mixed slashes across Windows, macOS, and Linux.
- Updated the atracker installation and usage guide to make installation simpler and install in developer mode (`-e`).
- Fully tested installing atracker on Windows and made some further improvements to the code and tutorials. 

### 2025-05-14 
- Updated set_objects function to change all rows if indices are not provided.
- Fixed atracker without filedir to correctly load current working directory of notebook and avoid writing files starting with dot.
- Updated instances where np.Inf was called to np.inf, according with numpy 2.x.
- Improved processor function by moving all coordinate unpacking (icom, head, tail) to prep(), dropped unused object columns to prevent dtype warnings.
- Fixed FutureWarning by setting dtype=object in newdat DataFrame used for filling missing frames.
- Fixed crash in checknearmask by coercing coordinate columns to numeric before applying np.isinf.
- Removed dependency on 'icom' in fillmissing(); it now uses the cx/cy columns passed via colpair, matching the updated coordinate setup in Processor.process().
- Improved robustness of movement variable calculations in Processor.process() by coercing cx_c and cy_c to numeric before using them in calcudiff(), avoiding ufunc errors.
- Updated differentiate() to always use pd.to_numeric() and return NumPy arrays, ensuring compatibility with vectorized operations in calcudiff() and turn metrics.
- Added safety checks in trajectory processing to ensure cx/cy-derived columns are clean before computing displacement, speed, and orientation.
- Introduced series_to_point_tuple() as a helper to extract (x, y) coordinate arrays from DataFrames, simplifying angle calculations with points_to_angle().
- Fixed potential crash in final post-processing by coercing cx_c to numeric before checking for NaNs or using np.isinf(), improving error resilience.
- Refactored wall and mask distance calculations in Processor.process() into a reusable function, add_kdtree_distances(), using KDTree queries.
- Centralized coordinate cleaning, filtering, and distance assignment logic to avoid repetition and ensure consistency across structure distance calculations.
- Ensured all cx_c and cy_c values are properly cleaned before KDTree queries, preventing type-related crashes during distance computations.
- Standardized output columns for structure distances (e.g., mdist/mx/my and wdist/wx/wy) for easier downstream analysis and plotting.
- Cleaned up the final data export step in Processor.process(): grouped rounding by precision, used safe column selection, and streamlined column ordering.
- Simplified removenearmask(): it now only removes coordinate and tracking data near the mask and leaves trajectory assignment and labeling to later steps.
- Reorganized the object-level loop in Processor.process() for clarity and logical flow: now removes in-mask data first, then interpolates, then assigns trajectories.
- Refactored head/tail correction logic within each trajectory, merging swapping, exclusion, and interpolation steps into one coherent block.
- Updated coordinate conversion logic to safely handle optional fx/fy columns and to fall back on cx/cy when conversion is not needed.
- Streamlined movement variable calculations (e.g., displacement, speed, heading) by using clear indexing and separating per-ID and per-traj steps.
- Updated lit_converter() function to robustly handle NaNs, valid tuples, and malformed strings when parsing coordinate tuples from CSV.
- Unified trajectory processing with new process_trajectories() function; replaces and simplifies masking, interpolation, and short trajectory removal.
- Ensured consistency between inmask and traj; all position data is now correctly cleared when fish are in cover.
- Updated _unpack_coords() to ensure unpacked coordinate columns are explicitly cast to float, preventing dtype issues during smoothing and interpolation.
- Ensured inmask and traj columns are not duplicated during processing by de-duplicating columns after concatenation or merging.
- Replaced .set_index("frame") with .index = data["frame"] to preserve the frame column while aligning the index.
- Fixed handling of empty datasets by guaranteeing essential columns (cx, cy, traj, inmask, etc.) are always present and properly initialized.
- Prevented KeyError during assignment by ensuring columns like mdist exist before setting values conditionally.
- Added a check after outlier removal in prep() to flag datasets as empty when all rows are discarded.

### 2025-05-07 
- Fixed track function that somehow missed part of the code
- Made max size in visual_editor 10000 instead of 5000. Will want to integrate a way for user to set these values.
- Added drawing of centroid to visual_editor thresholding mode

# v0.1.2 - 2025-04-29 
- Implemented new barcoding functionality to generate barcodes with various patemeters, visualise, save and load them

# v0.1.1 - 2025-04-28 
- Changed max size in visual editor from 2000 to 5000 to work better with larger objects/higher resolution (should implement it as a parameter in the future)
- Clarified note in tracking guide for thresholding that user should aim for blue not red shapes
- Fixed "erode" wrongly called "erosion in the tracker.py file
- Fixed storage of roi coordinates as only topleft and bottomright corner in visual_editor.py

# v0.1.0 - 2025-04-17
- Made huge upgrades to the interactive functionalities over the past many weeks so now for that rely on pyqt5 rather than opencv, which is much faster and user friendly but required a complete overhaul. I did not include any detailed changelogs here or a full description of what has been implemented, but will do that from now on now the atracker.set_interactive functionality calls the new visual_editor file and the old set_interactive and ivideo functionality is obsolete.
- Removed a lot of dependencies across atracker and thereby considerably simplified install_requires of the setup file, including removing pirecorde, scikit-learn, pathos, pyyaml, future, xlrd, xlwt, some python2 related stuff and specific versions. Also further cleaned up the setup file, reorganised, added comments etc.
- Fixed AT.reload function to also reload the config file when manually updated
- Added section to installation guide for manually installing latest version of pythutils
- Integrated double fast escape key for exiting all videos in the set_interactive function
- Fully integrated drawing of zones, which creates a white mask with multiple colored shapes representing different zones.
- Fixed mask and zone images not loading with correct size as original image or video
- Fixed getpts functionality to store a range of values to point columns
- Properly integrated thresholding into set_interactive, with ability to load a list of threshtypes
- Made final changes to using-atracker jupyter tutorial file
- Added robust video output handling using imageio with automatic resizing to ensure even dimensions, improving compatibility with ffmpeg and preventing encoder errors.
- Fixed Box error that resulted when AT.config was loaded with track instance and an error occurred, preventing the same code cell to be rerun\
- Added error handling for missing threshtype entries in threshinfo.yml, preventing crashes and printing a clear skip message.
- Fixed region suffix formatting by converting float regions (e.g., 1.0) to integer format (1) in output filenames.
- Improved color thresholding support by adapting threshinfo parsing to new HSV-based config format, and added blur/min_area/max_area sliders to the visual editor for color modes.
- Show general thresholding sliders (blur, min/max area) also in color mode of visual_editor.
- Fix KeyError when storing angle in fulldat during multi-color threshold tracking.
- Improved linkIDs() function to assign contour IDs more consistently over time using Hungarian matching and fallback logic based on distance thresholds. Prevents random reassignment after merges or splits.
- Added compact tutorial Jupyter notebook atracker_tutorial_compact.ipynb with minimal steps and clear annotations for setting up and tracking tutorial videos.
 - Updated README with clearer description of ATracker’s core functionalities, interactive modes, configuration and thresholding system, and added references to the documentation and tutorial folder.
- Updated the ATracker installation guide for clarity and reliability, replacing pyenv with virtualenvwrapper, consolidating Python install steps, and improving beginner usability.

# v0.0.9
- 2025-03-28 Improved imgprocessor function by making sure blur and erode values are always odd
- 2025-04-10 Removed the bg_extract file and moved the corresponding function to the utils
- 2025-04-10 Improved the get_bgfiles function to better check if bgfiles exist and added overwrite parameter
- 2025-04-10 Completely updated the big section 5 in the using atracker jupyter tutorial document

# v0.0.8
- 2025-03-18 Rewrote the setting-up guide from scratch and made it very organised and clear, also for beginners, and included all steps for mad and windows.
- 2025-03-18 Reorganised the documentation files, moved the tutorial jupyter file to documentation for working with and setting everything up with atracker, and integrated the documentation and jupyter file, making considerable updates throughout
- 2025-03-18 Created a new tracking.ipynb tutorial file with all steps for running atracker and specifically related to the tutorial files
- 2025-03-18 For get_files function set cdir default to "originals"

# v0.0.7
- 2025-03-12 Improved explanation in 1-setting-up.md 
- 2025-02-28 Added ceate_dummy_video.py file to the tutorial to create a white dummy video to create manual tracking for testing
- 2025-02-28 Added a video to the documentation how to use the manual tracker
- 2025-02-28 Included comment to set fname_vars to False when loading new videos added to 0originals folders that may have different name parameters
- 2025-02-28 Added some commands to the tutorial jupyter file for processing

# v0.0.6 - 250131
- 2025-01-31 Added/completely fixed pooling for processing using threading instead of multiprocessing
- 2025-01-31 Also added print statements with idnr if processed file and made sure print statements were flushed (important when pooling)
- 2025-02-01 Removed printing invalid rows detected from walls-related computations in the processing function as wrongly also showed when data was empty
 
# v0.0.5 - 250131
- 2025-01-31 Added get_inds function to more easily get the indices of a list of files to be tracked and updated corresponding documentation and also incorporatied a `names` parameter in the track function to call this directly
- 2025-01-31 Fixed processor function not considering files with no tracking data, not it runs all processing and returns a file with empty dataframe

### 250127
- 2025-01-27 Further clarified the functioning of the `get_files()` function in the `3-using-atracker.md` documentation
- 2025-01-28 Included note in setting-up documentation how to install ffmpeg on older mac systems
- 2025-01-28 Fixed getting the list of trackedfiles for process() function while checking for _E files working also on windows 
- 2025-01-29 Included brief explanation what the `simple` parameter does in the atracker tutorial file
- 2025-01-29 Added explanation to the atracker documention how to use the pools parameter
- 2025-01-29 Added that by default an "exclude" column is created in the overview file and that for tracking (and more generally the get_files function) it ignores files that have exclude=1. Also included a short explanation about the exclude parameter in the tutorial jupyter file.
- 2025-01-29 Added printing of circle radius and rectangle area in the `ivideo` function to better draw masks of the same size across multiple videos
- 2025-01-29 Fixed the set_config function documentation as a number of parameters were not shown
- 2025-01-29 Fixed drymode setting pools temporary in the set_config function, while that does not have that parameter, only the track and process functions

### 250122 - v0.0.3
- Implemented possibility to create custom threshinfo file with the set_interactive() function and to load it in the track() function
- Start updating the version number in __version__.py in line with changes here and categories for added functionality, improved functionality, and fixes
- Added calc_borderdistvec() function for more optimised use of checking distance to borders for excluding data in the processor() function
- Added some further clarifications to the tutorial jupyter file about regions among other things
- Improved the layout of the changelog
- Tracking older dataset "testpilots exp" and using orientbw = True encountered an error to do with angle variable being double generated, so fixed that in tracker.py file
- Fixed fixheadtail() utils function using data.angle for computation but should be data.orientation based on recent changes
- Fixed wallcoords computations in the processor function to better deal with missing data
- Fixed issues with NA rows at start and end of data due to smoothing in processing, so removed those rows directly after smoothing now
- Fixed issue with NA rows when checking if coordinates were near a mask in the nearmask() utils function 

### 250121
- Updated documentation for setting up python, virtual environments, etc and installing atracker
- Improved the processcon function of the imgprocessor file to check for contour ratio to thereby better exclude noise such as very long contour shapes

### 250117
- For tracker_man fixed screen size of video window as could be overlapping beyong the screen height
- For tracker_man Made sure that the coords drawn in the info panel are of the final data, thus including newly added data, and also relect com or fx,fy depending on the type shown (currently it always showed com)
- Fixed the order of drawing for tracker_man, such that the centroid and heading angle are always on top, and mouse pointer and drawing rectangle and mouse coordinate on top of that
- Further improved drawing in the tracker_man such that the show_both properly works and the centroid data is shown on top of the arrows, but then only the line is drawn rather than also the points, and changed color to black for better visibility
- Fixed issue with display in tracker_man, that when I would swap individual or change the time window, it would show the corresponding centroid data but would only update the orientation data upon changing the frame
- cleaned-up the doc for the keypress function of tracker_man
- ! Incorporated a major improvement by integrating some functions in the utils file to read a powermate nob that can be used for scrolling through the video of the tracker_man function. Further improving this functionality we can now scroll back and forwards by frame or by second by simply toggling by pressing the powermate

### 250110
- For tracker_man improved drawing of angles by colouring based on absolute angle, and from red to green, making it easier to distinguish outlier values
- For tracker_man integrated functionality to smooth angles to better understand how much specific outlier values matter, which can be shown for specific frames using the g-key.
- For tracker_man fixed when deleting an angle arrow it would only show it when the frame was changed, both for single frame and using the rectangle selection
- For tracker_man fixed that rectangle selection works when angle arrows are displayed as it wrongly selected the fx,fy columns instead
- For tracker_man made sure that the orientation is drawn irrespective if arrow angles are shown or com data
- For tracker_man integrated possibility to toggle drawing of coordinates size based on speed on or off with keypress (j) as for the former points are only drawn when there is data two subsequent frames
- For tracker_man fixed referring to coords for plotting by creating another variable plotcoords

### 241220
- Fixed 'b' key in tracker_man file not being explained in the keypresses documentation
- Added new display option in the tracker_man file to show the coordinates of the current mouse position, which can be toggled with the 'h' key
- Added parameter delcontdata to processor file to remove contour data, which is helpful for when tracking colored data and bw contours were tracked for angle only

### 241205
- Updated utils with some new functions to get the screen resolution as screen_info somehow didn't work in some virtual environment I created so now it should be more failsafe. Updated ivideo file accordingly.

### 241009
- Updated internal documentation and tutorial about storing and using thresh_types
- Integrated simple `save` function to store the overview file
- Integrated easy way to create multiple bw threshtypes in the same threshinfo file, simply add a suffix and it will work automatically. For this I updated the imgprocessor, tracker, atracker, and ivideo files
- fixed all pathing from mac-specific (i.e. "/") to system general using os.path.join mainly in all relevant files, so now atracker is fully functional in windows as well!
- Wrote processing doc text with details about all variables. Renamed a number of them and reordered based on appearance.
- Wrote detailed processing documentation, including for manual tracking, and extended tutorial file with processing steps

### 241008
- Created five dedicated animated videos for the atracker tutorial for learning to track single objects with simple and advanced (skeleton etc) characterisation, pairs of objects, setting the roi and regions for separate objects, groups of objects for bw and color tracking, and group of barcoded objects.
- fixed that atracker rightly also loads and renames .m4v files
- Removed the fixdat column as now obsolete with fully functioning mantracking function
- Fixed lineprint for 'no files to convert' statement in atracker file
- Removed some not-needed import statements
- Improved printing during background extraction
- Fixed renaming `region` to `regions` in tutorial for setting configuration
- Added set_regions function to easily add regions information to the overview file
- Fixed indexing of files for the set_interactive function from iloc to loc
- Improve the using-atracker documentation by further explanations, restructuring, and adapting for improvements with updated tutorial videos
- Improved the zone drawing functionality to be able to also create a single zone image just like a mask (`maskzone`) and extended explanation in the tutorial documents
- Integrated simple get_objects function to add number of objects to track to the overview file
- fixed print statement for the threshtypes parameter of the `set_interactive` function
- resized and repositioned wrongly positioned trackbars of the ivideo class functionality and maximised size of infopanel
- included Hue infopanel to help select the right hue values, which dynamically shows the range selected by darkening the rest
- fixed threshold panels for colors in the ivideo class as min and max values were not correct
- in the tracker.py file fixed tracking stopping 1 frame too early
- Created `check_maxframe` function in the core `AT` class to check all videos frame by frame for missing frame and fix the maxframe if there are missing frames, and integrated in the tutorial document using atracker
- playing with the check_maxframe function using while loop and the `cap.set` function, I noticed a discrepancy of 1 frame, so I set the fcount in the atracker and ivideo classes -1.
- After more work I discovered the discrepancy can sometimes even be 2. I realised I can use the cap.set function backwards very quickly to find the last working frame so I updated the check_maxframe function. And then I decided to remove it completely by implementing the new utils `find_max_working_pyframe` function directly into the setup_files function, much easier. Also updated the tracker_man file with the same changes to work correctly with the "pyframe", which is the frame number-1.

### 241001
- I added a version of numpy to make sure it would not through errors (1.26) and modified the warnings command in the utils file
- Changed python-opencv version to an older one and to python-contrib-opencv (3.4) so atracker is still installable on mac OS 10.13 and working fast, as some newer versions were very slow on some systems I tried
- fixed getting the right index when using the roi parameter in the set_interactive function
- fixed mistake in imgprocessor function comparing the shape of an image with the mask

### 240925
- Updated setup file to automatically install python opencv, changed scikit to scikit-learn, and changed numpy to get it working
- Wrote extensive readme with detailed steps how to get atracker up and running on a mac and windows system
- Created documentation folder and moved much of the (new) contents of the readme to four separate documentation files, including setting-up, installing and loading, using atracker, and manual tracking
- Integrated the reload function in initiation
- Tried out various python versions and openv and numpy versions to make ATracker work out of the box after installing with no manual installation of dependencies anymore needed
- Write complete guide how to use atracker and all its functionalities, stored in documentation folder, thereby used parts of the old guide.ipynb file and deleted it
- fixed some things in the ivideo folder regarding max frames

### 240906
- Integrated computing turning acceleration variable in processing
- Fixed some small issue to do with computing orientation correctly
- Improved the `smooth` utils function to also work with non-series data
- Integrated parameters in 'manfix' to already set the frameloc, thresholdspeed and timewindow, so quickly all the relevant data can be shown without needing to scroll through the bars
- In the manfix function, integrated option to print the currently displayed trajectory data, helpful for further checking the data displayed. Including in-script computed speed variable, key for many tracking issues.
- Added smoothwin trackbar to the manfix functionality to dynamically change coordinate smoothing to see its effects on the trajectories and computed speed

### 240905
- Further smoothlined the orientation improvement functions of the processing script and associated functions, including `smooth` and `fillmissing` to return a series rather than store in the original series, to avoid issues with subsetting
- Implemented parameter in processing for the fillmissing function to set the max length of missing data to fill in, such as to be ale to fill in large gaps of missing orientation data, which arise after manual tracking only times with significant orientation change.

### 240904
- Fixed processing script not processing _E files and some issues with conversion as well as correctly storing the newly processed file (_F.csv)
- Enhanced `removenearmask` function to fill missing trajectory IDs between the first and last frame of each trajectory and to remove trajectories shorter than a specified threshold, ensuring better trajectory consistency and data integrity.
- Fixed new improve orientation function from not working well with indices

### 240903
- Added manonly parameter to processing to indicate if only manual fixing should be done or full processing
- Renamed hx,hy columns to fx,fy in the manual processing function to correspond to this naming in the main processing file
- Implemented some fixes and additions to update the processing script to better work with heading and/or orientation data together, including computing an optimized orientation variable based on heading above a speed threshold
- Considerably reordened the processing steps to optimaly calculate orientation based on heading andd speed threshold

### 240902
- Fixed timewindow showing 1 frame too little and changed Frame position bar to show "Frame+1" to indicate how python shows numbers
- Fixed the speed threshold computations to work properly and be in mm/sec, such that now with the threshold bar one can easily select a reasonable threshold, such as 5 (mm/s), which will then show all coordinate data at less that speed in red, to easily go there and fix orientation data manually
- Added showing radius of the coordinate data based on min and max user provided speed values to better distinguish slow and fast motion
- Created create_composite.py file for quickly creating a composite image of all changes observed during a video so easily any patterns can be detected, such as something blocking part of the camera view. Script is not yet very dynamic and integrated but the functionality is there.

### 240830
- Fixed an error resulting when displaying the tracking data when the current(ly selected) data has length 0
- Integrated orientation to be drawn as a series of arrows
- Changed colouring of arrows based on where they are in the frame series to recognise time
- Integrated option to hide centroid coordinate data when showing orientation arrows (with l-key)
- Integrated a speed threshold trackbar to dynamically show the speed the fish are moving and color points either orange (above it) or red (below it), to help focus on the slow-moving parts as at higher speeds we can simply use heading to compute orientations

### 240731
Made a further number of improvements to the manual tracker:
- Integrated working with dataframe with an "orient" column that is then converted to head coordinates
- Improved ptypes option to include "orient"
- Enhanced display of tracking data by simultaneously showing centroid and heading coordinates
- Enabled cropping and deleting of heading coordinates in same way as centroid
- Calculated heading positions at standard distance and integrated to dynamically show them based on free angle drawing tool, even when at that frame the point is drawn at a very different distance
- Enabled drawing of coordinates with size scaled on inversed speed to be able to more easily find times where animals are moving slow (clusters of larger points)
- Improved saving function to work with the added heading coordinates data as well as showing number of changes made

### 240730
Made a number of fixes and improvements to the manual tracker:

- Fixed that data shown with the time window bar is also actually the data that is deleted when selected with the rectangle rather than of the whole data
- Fixed and improved printing of changes made to the data, including adding, editing, and removing of frames, and showing the list of frame numbers.
- Add point vs rectangle selector to info panel
- Fix that coordinates of drawn rectangle always starts with the top-left coordinates even when drawn from another corner
- Fix that frame position and time window slider are same width as video plus info panel (also when resized)
- make frame position ar show total numer of frames rather than 200 steps so the scrollbar is actually meaningful
- make sure the time window slider starts at 1 frame difference and does not show all at 0 but at the end only
- Add frame range to the info panel
- Improve resolution and size of the info panel

### 240513
- Improved showing of framenumbers in manual tracker with option to only include
  the nearest frame to the mouse position with white background and within the screen window

### 240417
- Fixed video resizing for manual fix function
- Integrated drawing of rectangle in manual tracking function to delete coordinates
  within the rectangle, which is helpful for outliers
- Improved checking for tracking files if there exist edited files already select those

### 221101
- Fixed processed files no mx and my colums being written when no coordinate data
- Fixed processor accidentally using converted nearmaskdis and reconverting it
220918
- Made a range of improvements and fixes to the manual tracking function to make
  it work properly now it is integrated in atracker
- Fixed existonly=False setting in the track function as otherwise indices would
  be wrong
- Integrated stand-alone mantrack package into atracker. Version 3.2.4 Version
  history is provided below and from now integrated with that of atracker
- Improved get_bgfiles function with functionality to provide a list of indices, a
  list of startframes, and a list of stopframes to get more specific background
  files, such as when needed when an animal is frozen for a long time
- Updated showinfo function to work with files that have regions
- Made temporary fix so old files with regions (now named zones) can still be
  automatically processed

### 220520
- Fixed processor working with blank mask files
- Fixed processor ignoring startframe of configfile when no startframe provided
  in overview
- Fixed issue with colouring of trajectories in visualise function
- Fixed issue with data processing not being correct for cover trials

### 220517
- integrated using startframe as set in the config file when no startframe
  provided in excel overview file
- Changed naming of video window to actual name of current video file
- Fixed issue with disthreshold value in tracker file

### 220325
- Incorporated checks into processing function to work with overview files where
  the ID and wall mask columns are missing
- Fixed showinfo function to work with single files as string input
- Added checkonschange parameter to turn off the sophisticated but very slow
  contour splitting function when not needed
- Added linkdisthreshold parameter to use with tracker function to exclude linking
  of IDs too far away
- Improved tracker function to not link identities that are too far away when
  the focal contour is not properly detected

### 220324
- Fixed issue with drymode working with a copy of AT.config, thereby falsely
  not back overwriting the original settings

### 220323
- Improved tracker function to work with merged contours in groups with more
  than two objects. Still to incorporate is the scenario when in one frame more
  than 2 objects merge.
- Integrated variable mergedmindist to config file that will be used as a
  condition to determine if a very large contour is a merged contour of two
  previous contours
- Implemented custhreshtypes and cusobjects parameters to the tracking class to
  more easily quickly try out different tracking parameters
- Fixed linkID function to link IDs for multiple BW contours (since changes to
  the code for colored tracking)

### 220322
- Some small fixes to have atracker work with older config and thresh files where
  there is no blur2 or orientfrombw parameters
- Fixed fix_roi function in line with change on 220312

### 220315
- Integrated orientfrombw parameter in config file
- Improved drawing of angle for color threshtypes
- Integrated linking of col threshtypes and nearest bw contour to calculate object
  angles
- Adapted simple mode in tracking function to work with coloured tracking

### 220314
- Automatically set tracking mode of color thresh_types to simple
- Made printing of ids different for BW and color thresh_types
- Fixed linkids for bw to only check bw contours in movedat
- Fixed tracker to work with bw and color tracking simultaneously
- Integrated check in setuptracking function if threshtype exists in threshinfo
- updated check_threshtypes function to convert tuple

### 220312
- Fixed roi such that the max coordinates are external such that images are not
  wrongly cropped 1 pixel inside of their dimensions at default

### 220303
- Added fix to processing function to work with datafiles without any data
- Integrated pooled processing, but with a limited pool based on all files in
  tracked folder
- Improved printing of processing function
- Fixed an issue with addtrajsnmaskstate when trial ended under cover
- Added function to check if code is executed from notebook or not
- Fixed error of processing of headtail position at end of trajectory
- Implemented get_size function, which is a simpler version of the ivideo script
  dedicated to measuring the size of animals based on a folder of images
- fixed pooltracking to quickly skip files currently being tracked
- fixed pooltracking to work with already tracked files
- improved showinfo function to work with list of files
- fixed tracking error where skelpts where empty list
- upgraded get_files function with parameter to only list files that exist
- upgraded geom_tocoord function to work with shapely 2.0

### 220301
- Changes checking of skip variable for not being 1 to not necessarily have to
  fill in the column with zero's
- Fixed masked images stored with wrong size and implemented check in ivideo and
  tracker to resize mask when required

### 220225
- With a lot of effort creating multiprocessing function that checks a pool of
  existing files, skipping videos that are currently tracked for other regions
  and picking them up later, complete with proper notifications.
- Improved tracker function to work with pool of indices still to do rather than
  a static list of files
- Included printing of nr of files tracked and still to do in tracker function
- Added ability to print frame with identifier during tracking
- Fixed tracker not working properly yet with regions
- Fixed wrongly naming of bg, mask, wall files

### 220224
- Fixed clean exit of pool process with ^C
- Did some checks and pooled tracking is only possible from the terminal, so
  integrated a check before pooling to avoid hanging in jupyter
- Integrated better checking when tracking cannot be performed because the video
  is already currently tracked and indices placed at end of tracking list
- Fixed bgextractor not taking the right bgframe start and stopframes
- Integrated bgextractor to use skip filter information
- Fixed ivideo accidentally drawing roi in case of multiple regions in same video
- Added standards for drawing for the various interactive modes to further speed
  up the drawing process

### 220223
- Fixed computing of missing data where multiple trajectories are combined and
  added parameter to be able to set this
- Fixed getindsections to work with gaps where multiple trajectories are
  combined
- Improved interactive conversion setting by enabling a single or multiple pre-
  determined measures to be set that will give a single conversion measure.
- Made it default that nearmaskdis and inmaskdis parameters are provided in
  real dimensions rather than pixels to more easily provide relevant parameters
- Integrated nearmaskdis and inmaskdis parameters for respectively checking if
  missing data is because the object is within the mask and for removing data
  too near the mask.

### 220222
- Integrated usage of roi in ivideo mode to use for getting threshold parameters

### 220127
- Fixed ragged edges in masked image visualiser function
- Get visualise function to work properly for videos tracked with color thresholding
- Fixed hiding background and drawing coloured IDs

### 220124
- Updated tracker.linkIDs function to work with color threshtypes. Currently only
  implemented to work with a single contour per color.
- Updated tracker function so now multiple different color threshtypes can be
  accurately tracked.

### 220121
- Fixed thresholded image and contour info not showing in interactive mode when
  using simple.
- Incorporated interactive color thresholding in the set_interactive function
- Included a check for threshtype and included larger max size for contour size
  in color thresholding
- Got initial color tracking working with a single color. Further steps are to
  run color tracking in parallel temporally for multiple colors and combine with
  bw tracking for e.g. orientation calculation, as well as using color information
  in visualisation

### 220118
- For tracking, if nrobjects is not provided now automatically a value of 1 is used

### 220117
- Fixed some issues with tracking and processing not working when no roi or conv
  values are provided.
- Fixed processing to calculate turning variables when only heading is available
- Added abscumturn variable to processing

### 220113
- Updated checknearmask function to only require at least 1 frame that is within
  the required distance.
- Fixed the data not being sorted and having indices reset due to issue of
  adding untracked frames to the end of the dataframe, impacting some measures

### 220112
- Changed differentiation of speed, acceleration, and turning speed to backwards
  from forward differentiation and removed delNArows parameter from processing
- For processing fulldata parameter made sure to subtract 1 to have the proper
  full range of the data.
- Changed calculation of turning speed per second and added cumulative turn
  variable to see if tracked objects have a tendency to turn a certain direction
  over time.
- Added a lot of the draw_frame parameters to the visualise function
- Fixed the add canvas functionality of the visualise function to create canvas
  with right image type (np.uint8)

### 220110
- Added a guide.py and generated guide.html file with explanations and python
  code for working with ATracker.
- Fixed set_interactive crashing when the last frame of a video is corrupt
- Made helperinfo display a bit smaller in the set_interactive function to still
  fit on the screen
- linked simple mode from set_interactive to config simple rather than a
  separate option
- Removed track_nobj parameter from config file and integrated its use via the
  overview file to make it easier to track files with different object number
- Removed 'cover' parameter from processing as it can be automatically extracted
  from the overview file
- Renamed processing parameter 'window' to 'alonewindow' to make it clearer it
  is the time window for the removeoutliers functionality
- Fixes issue in processing file with wrongly checking if maskcoords were none

### 211201
- Added new functionality to draw zones using ivid, which stores a colored mask
  image with different zones that can be later used to analyse time spent in
  different zones
- Fixed findcontours function to work with all versions of opencv
- Fixed ivid to not draw objbox when nan
- Renamed zones, different areas to be tracked within a video separately, to
  regions to not mix with zones in terms of different areas within an arena

### 211123
- Fixed maskcoords not being cropped to roi for mask checking functions
- Fixed outofroi function as was using real roi rather than max coordinates
- Fixed checknearmask function for falsely checking win+1
- Made sure to exclude computing orientation-related information when such data
  is missing from datafile

### 210706
- Fixed issue with processor

### 210705
- Fixed another issue that resulted in a tracking error due to no IDs being
  available to link anymore when too much time has passed.
- Fixed an issue that resulted in a tracking error due to wrong ID allocation
  when more than the maximum number of contours were tracked.

### 210629
- Integrated data orientation into avgvel function
- Fixed issue with wrongly calculating distance matrix
- Integrated movedat in IDlinking to work with not just data from the previously
  tracked frame
- Fixed issue with wrongly allocating IDs for the first frame.
- Fixed weightedavg and avgvel functions to always at least give np.nan

### 210628
- Fixed an issue with processing of files with multiple ids

### 210625
- Fixed the smooth function as it was not writing the newly smoothed data correctly
- Made some small fixes to overcome errors for various processing functions
- Further improved prediction of current contours by calculating fishes' heading
  with a time delay
- Integrated creation of temporary movedat dictionary to be used for predicting
  future contour positions and allocations of IDs with little runtime required

### 210623
- Created temporary visual fixid function
- Added temporary fix regarding area checks
- Integrated previous contour predictions for contour splitting function
- Improved weighted average function
- Fixed issue with ID numbers being floats not integers

### 210622
- Fixed trajectory being wrongly drawn increasingly thick

### 210621
- Fixed issue with consplit function with drawing filled-in contours
- Got consplit function working properly into the tracker class

### 210616
- Improved filledconcoords and consplit functions further by blurring and dilating
  to not get weird points far away from main contour
- Further optimized allocation of points to contours based on ratio of previous
  points
- Improved consplit function by working with all points inside the contours rather
  than just points on contour edge
- Created filledconcoords function to get the coordinates of all points inside a
  contour, including the contour edge
- Created roifromcon function to get roi from con to be used for drawing a canvas
  image of just a contour

### 210615
- Improved split merged contour function by integrating (temporary) a predictive
  module of current contour positions based on previous speed and heading
- Improved split merged contour function by optimal allocating based on contour
  size
- Created working split merged contour function

### 210613
- Added function that determines if contours are merged or split, to be used for
  contour splitting and linkID checking functionalities
- Fixed LinkID function that wrongly allocated to previous IDs

### 210611
- Included config option to draw uniq nicely spaced colours for all IDs
- Created functions that in various ways split a merged contour in the most
  likely contours linked to two previous contours. Works okay but issue is contour
  is increasingly allocated to only one of the contours.

### 210609
- Fixed issue with smoothing multiplying nan values at start of trajectory
- Integrated another conditional rule in head-tail swapping that accounts for if
  that frame would be excluded because too close to the edges
- Improved checking for existing files for processing function
- Added rule to remove head and tail data near the roi
- Fixed that coordinate data continued to be linked even when going out of roi
- Finalised fixing that centroid is in the centre and that user provided point
  can be used if available
- Integrated some further improvements and fixes to processing function
- Made some fixes so visualiser function works properly with updated processing
  function

### 210603
- Fixed displaying wrong framenumber on tracked videos
- Added simple showvideo utility function
- Worked hard (again) on the fixheadtail function to better deal with various
  situations. Working better yet again, but it remains very difficult to get
  close to 100% good orientation data.
- Improved trajectory determination by integrating gap parameter
- Improved usage of fillmissing function by integrating ability to set time and
  distance windows
- Fixed borderdist function calculation as did wrongly take into account the roi
- Integrated additional rule for excluding orientation data based on turning speed

### 210601
- Further fixed fixheadtail function for working at the end of trajectories and
  with data where all area measures are nan
- Improved coordsfrommmask function to also add outer coordinates, such as when
  mask intersects with roi
- Fixed mistake in use of converting function
- Fixed calculation of roi distance
- Fixed and improved calculation of mask distance
- Fixed visualisation function not showing non-tracked frames
- Fix calculation of points on mask (which was wrong due to above convert issues)
- Fixed that missing frames (such as when object goes out roi) were not in data
- Added pt drawing to visualise functionality

### 210531
- Fixed swapheadtail function for only swapping the outer points
- Improved visualise functions to use a data dictionary instead

### 210527
- Improved speed of trajectory computation function
- Fully cleaned and updated visualisation function, which now enables visualising
  all kinds of data in an easy way
- Added coordsfrommask function

### 210526
- Integrated new processing functions for erroneous head and tail positions and
  swapping them where possible. Works a lot better than the original orientation
  fix functions.

### 210525
- Get basic visualise function working
- Fixed orientation calculation in processing function

### 210524
- Fixed trajectories not showing
- Get processor working with improved tracking files (except still for orientation)

### 210521
- Improved drawing of skeletons and added function for drawing coordlists in general
- Added lathom measurement to get an indication of longitudinal variation in
  the thickness of the object. This is usefull to determine when orientations
  are more likely to swap and be wrong, i.e. at low values.
- Added outerarea measure to help identify moments when fish make a sharp turn
  and curve their body

### 210520
- After some effort finally got a nice IDpairing and ID allocation function
  working and integrated that is relatively fast. Also works well for trials with
  multiple objects (not just noise).
- Piloted various solutions for overcoming overlaps and have a couple that sort-of
  work but not well enough yet to integrate them into atracker.

### 210519
- Added drymode function to quickly track a random number of sequences for a
  random selection of videos to help further check if the threshinfo values are
  optimally chosen. To be used before intensive full-scale tracking.
- Added alternative headpoint calculation for cases where the shapely function fails

### 210518
- Past week sped up tracking considerably by creating a vastly improved image
  processing function, better calculation of object tracking, better storing
  of temporary data, and removing prevlist and movelist objects
- Integrated calculation of object skeleton and thereby much improved determination
  of object head, tail, and inner centre of mass
- Improved calculation of skeleton to exclude potential other contours in same
  cropped region of interest
- Further improved calculation of tip of head by calculating point of intersection
  between polygon and line extending outwards from object's orientation
- Cleaned up image processing further by making it into a class with initialisation
- Fixed proper converting of various shapely geometries to coordinates
- Further fixed/improved working with shapely contours for calculating tip of head
- Improved writing of data to increase tracking speed

### 210517
- Fixed mask not being used in image processing
- Fixed display video not closing after stopping visualisation
- Fixed drawing of trajectory behind contour not working properly

### 210514
- Improved positioning and sizing of ivideo windows and prevented overlapping
- Improved creating and writing to conlist
- Fixed minimum box potentially having coordinates outside of video window
- Added ordering of contours from top to bottom and left to right
- Added significant improvements to imgprocess functionalities, including the
  calculation of internal centre of mass, very accurate head and tail calculation
  and skeleton calculation. Also considerably improved speed of image processing
  in various ways.
- Improved drawing of skeleton as clear single polyline
- Added drawing of individual boxes for the tracked objects in ivideo

### 210511
- added drawing of orientation calculation to ivideo
- fixed conditions for calculating angle for contour extraction

### 210504
- fixed bgfile extractor falsely skipping files that did not yet have a bgimg
- improved orientation tracking further by using the midpoint of the box rather
  than moment, then 99+% of the time orientaiton is correct and any further
  orientation comparisons are not necessary

### 210503
- removed delete files when overwriting as too dangerous..
- improved linking of identities over time and excluding duplicates. Worked on
  optimal matching of identities when multiple objects need to be tracked, but
  is not yet working properly.

### 210430
- incorporated option to temporarily overwrite start and stopframes for tracking
- incorporated option to skip files when they already exist for tracking

### 210427
- fixed tracker stopping upon missing frames, which are not skipped when near
  the start of the video, which happens with gopro videos when audio is not
  deleted. Better is to convert videos like that with ffmpeg, but very time-
  consuming (ffmpeg -i [videoname] -vcodec copy -an [videoname])
- fixed get_bgfiles function to not stop when issue with single file
- fixed get_bgfiles function to work with multiple zones in same video
- improved centralise processing function to both work with the roi and walls
- improved mask creation by integrating subtracting and adding additional shapes
- added helperlines functionality to interactive video mode
- integrated drawing of ellipse to interactive video mode and mask creation
- added roi_fix function to make sure the coordinates lie within the video limits
- fixed dataframe querying functionality when values contained both strings and numbers
- added reload function to reload the overview file when updated manually while
  the ATracker instance is still loaded
- interactive video mode now automaticall skips missing frames and notifies user
- integrated vid_displaysize option for interactive video mode
- fixed zones not working in updating the overview file
- changed excel file reading and writing to overcome xlwt warnings
- video files with capital extensions are automatically renamed with lower extensions

### 210426
- added walldist function
- considerably improved orientation and head and tail computations using centre of mass
- incorporated centre-of-mass inside polygon function
- added arena centroid distance processing function
- incorporated new centralise function that puts the coordinates at the centre
  of the arena at (0,0)

### 210422
- incorporated calculating distance to mask and visualising closest point on mask
- added first version of visualise function to visualise processed tracking data
- improved processing function to convert data after all other processing and
  compute _c variables to facilitate plotting of the raw data
- incorporated fillmissing of orientation data and front coordinates
- fixed orientation calculation in processing function

### 210416
- get function working
- integrated processing functionality to change the fps and convert coordinates,
- added interactive mode to add tupe of width and height conversion

### 210415
- got fixdat to work where data is set to nan when in the fixdat column of the overview
- added display of framenumber to tracked videos
- created fix functionality to prepare tracked files and make sure they
  have complete and appropriate row and column data
- created new processor class for the processing of tracked data

### 210414
- ivideo now has fullrange option that is used for setting the start and stopframes, otherwise
  video is temporally cropped to within the frame region
- fixed issue for ivideo function with goto key refreshing screen

### 210413
- now set_interactive uses existing threshold data when available
- with some effort got pool tracking to work, but this won't work in jupyter
- added overwrite mode if files should be skipped when already tracked
- integrated tracking of multiple zones and correct writing and moving of files

### 210409
- added ability to add and remove zones to configuration and use them accordingly
- made set_interactive work with videos with zones
- improved get_bgfiles function to only check for files that do not have a bg file yet
- fixed get_bgfiles from error about floats when start and stopframes were provided
- added conversion drawing mode to the interactive video function
- improved setup_files function to check for existing files, making it easy to add new files
- fixed and improve orientation calculation

### 210408
- made possible to draw trajectories below the fishes' contours
- fixed mask not incorporated into tracking
- made sure that visualisation of tracking single fish is working
- got tracking of single fish working while showing contour and ID, at ~50fps
- got video loading, writing, exiting, file moving to work

### 210407 - ! Start of changelog
- improved get_vids function to ignore inds not in dataframe (instead of giving an error)
- fixed not closing windows after using interactive video mode
- added changelog file
