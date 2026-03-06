# Processing data

Run manual tracking to fix erroneous tracking data if visual inspection of videos indicated this is the case. You can thereby provide a specific list of files if needed.
# Provide an overview of parameters for the processing function
print(AT.process.__doc__)
from pythutils.fileutils import listfiles 
listfiles(AT.dirs["tracked"], type=".csv", keepdir=True)
AT.process(names = ["animtest_solo_071024_S01_1413.csv"], manfix=True, manonly=True, man_types = ["c"], overwrite=True)
# Add dummy tracking data to the white video
AT.process(names = ["white_video.csv"], manfix=True, manonly=True, man_types = ["c"], overwrite=True)
s

## Checking the tracked videos

The first step after tracking videos is to check the tracked videos in the "4tracked" folder to see how well the tracking has worked. Scroll through the videos and look if there are many instances where tracking was lost, if wrong objects were tracked, if tracking often switched between a focus object and noise, etc.

In case of major tracking issues, across all videos, it indicates that the threshold parameters were not optimal, potentially the background file not perfect, or alternatively there were some issues with the video, such as changing light levels. Based on the severity and type of issues determined, it could be a solution to try and improve the threshold parameters. In some cases the best solution is to create a new thresh_info file and only use that on the selected videos with issues.

The automatic processing functionalities should be able to overcome most minor tracking issues. But ATracker also has a very useful **manual tracking** function (explained below) with which it is very easy to remove erroneous frames, edit tracking locations, or even manually track large sections of videos relatively quickly. Below I explain how.

## Overview of the processing function

To process the tracked `.csv` files, we use the `process()` function. A lot of parameters can be set:

```python
AT.process(pools=1, names=None, overwrite=False, fulldata=True, convert=True, removeoutliers=True, 
           alonewindow=5, manfix=False, man_vidresizeval=0.5, man_types=["c"], man_customstep=100, 
           manonly=True, man_frameloc=0, man_timewindow=0, man_threshold_speed=0, smoothwin=10, 
           changefps=None, addIDs=True, nearmaskdis=20, trajgap = 50, edgedis=10, 
           filllenthresh_com=500, inmaskdis=10, mintrajlength=10, filllenthresh_headtail=40, 
           headtoorientspeedthresh=1, fillmissingorientdiffthresh=100, centertype = "walls", 
           centralise=False)
```

To see the explanation of the different parameters simply run

```python
print(AT.process.__doc__)
```

In general, the processing function performs the following processing steps in order on the data. Key is that it will run a lot of checks and interpolations to make sure the tracking data is perfected and lots of relevant variables are automatically computed. In chronological order, the function will:

1. tracked .csv file is loaded and columns with tuples converted to values, relevant mask images checked and loaded
2. Short fragments of tracking data removed
3. Empty rows added for each frame and id combination for the full time series between first and last frame
4. Data is sorted by frame and (tracking) id
5. Option to manually fix the data (see below)
6. A time data column is added
7. Potentially the data can be subsetted to a lower fps
8. ID data from the overview file ID column is added to the data
9. Missing tracking data is filled in
10. Data column added with status if object is in or outside the region of interest
11. Data column with trajectory number added based on if the object dissappeared beyond the region of interest or under the mask
12. Data column added with status if object is under/out of the mask and frames under and within threshold value from the mask are removed
13. Where relevant, advanced computations are run to correct allocation of head and tail position and overcome potential swappings
14. Positional coordinates and orientation data are smoothed
15. Positional coordinates are converted to mm
16. Key movement variables are computed, including heading, speed, acceleration, turning speed, cumulative displacement
17. Structural distance measures are added, including to the boundaries of the region of interest, the mask, the walls, and the centre. Coordinates are added of the nearest point on the mask.
18. Data is centralised when desired, based on the region of interest, the walls, or a point as set with the set_interactive function.
19. Final data organisation, including re-ordering, resorting, and rounding the different variables. Removal of irrelevant columns and =last rows of data that have NANs due to differentiation.

It is possible to overwrite existing processing files with the *overwrite* parameter, and let processing process all data within the time boundaries of the existing tracking data or within the boundaries as provided by the overview file (*fulldata* parameter False/True).
Note that the *edgedis* and *maskdis* parameters should be set in pixels and so checked visually to be certain of appropriate values.

## Using the manual tracking function

After having checked all videos it may have become clear that before all processing is conducted that some tracking data needs to be removed, fixed, or added. For this we can use the manual tracking function. For this set the `manfix` setting to True and it is best to set `manonly` to True, such that it will loop through all videos in series but only does the manual fixing such that further processing can be done in a next step.

The manual tracking function is a very handy interactive tool that works with a number of keypresses and listens to mouse movements, similar as the `set_interactive` function. It opens a video window, a scrollbar for changing the video frame, a scrollbar for changing the time window for which tracking data should be shown, an info panel with information about the coordinates in focus, a scrollbar to change the speed threshold, and a scrollbar to change the smoothing window.

> **Note**: If using the names parameter, the processed csv file should be provided. For the interface it will show the tracked video, and manually-adjustable view of tracking data will be additionally shown on top.

### Controlling manual tracking

One can command the video in the same way as for the `set_interactive` function, and there are a number of additional important keypresses and commands:

```python
The keys q,w,e,r,t,y are used for controlling the video frame position:
q : go to first frame in video.
w : go one second back in time
e : go one frame back in time
r : go one frame forward in time
t : go one second forward in time
y : go to the last video frame
u : custom step size to go forward in time

Change selection:
c : change from point to rectangle drawing

Control the type of data tracked:
i : changes the animal ID to the next ID in the list
p : changes the point type to the next point type in the list
d : deletes the current point
f : removes the selected_points from memory

Change the animal state:
x : change state animal is in (0 <> 1). As the default state is NaN,
    make sure to press x twice to set state to 0!

Visualisations:
a : show/hide all currently tracked data as connected dots
z : show/hide the framenumbers for the currently tracked data points
` : show all framenumbers or only near the mouse
v : displays the changes made so far
l : show only orientation data

Storing and exiting:
n : saves the data in file with predefined name and creates new datafile
s : saves the data in file with predefined name and exits
esc : exits the program without saving
```

When the video is open, we can thus change the currently shown frame with the scrollbar or use the qwerty keys. To show the tracked data we can press the `a` key. If we want to show or edit only a specific window of the tracked data, we can adjust this with the time window scrollbar. 

To change the position of a tracking point, simply click with the mouse in the video and the new datapoint will be stored. If you need to delete a point, simply press `d`. If you want to delete multiple points, change the drawing option to rectangle with the `c` key, drag around the points you want to delete, adjust the time window if needed, and now when delete is pressed only the data currently visible (thus not that is before or after the timewindow) will be deleted.

We can flexibly change between the animal for which we need to change tracking data using the `i` key, and between types of tracking data, such as from centroid to head, using the `p` key.

To set the state of the animal we press the `x` key. For the first frame and last frame it is thus important to set this correctly, and for any change in state that should be recorded.

When happy with all changes, simply press the `s` key to save the data or `n` to create a duplicate file, or click `esc` to stop and not store the changes made.

For changing orientation data it may be helpful to differently plot the trajectory below a certain speed threshold, so it is clear what data needs to be deleted. For this you can use the speed threshold scrollbar. We can also set the smoothing window dynamically to find the window for which the data is optimally smoothed.

### Running man tracking only

We can run the manual tracker directly from the `process()` function or use it standalone. For our main interests we will in most cases do the former. Next we have to set the `man_types` parameter for the type of data to fix. In most cases (the default) this is centroid data ("c" or "com" for center of mass), but other options are "h" for head position, "t" for the tail position, or "o" for the orientation:

```python
AT.process(manfix=True, manonly=True, man_types = ["c"])
```

In addition, we can set a number of helpful parameters to run the manual tracker, including the frame to start with (`man_frameloc`), the time window to start with (`man_timewindow`), and the speed threshold (`man_threshold_speed`) below which data will be visualised differently.

### Getting the right files to manually track

To get started with manual tracking and only do it on the relevant videos, a good idea might be to set the `names` parameter. For example, to set the first video to check:

```python
trackedfiles = listfiles(AT.dirs["tracked"], type=".csv", keepdir=True)
names = trackedfiles[0:1]
```

to set it from a manual file:

```Python
names = [AT.dirs["tracked"] +"/WallA_E48_Corona.csv"]
```

or to start from the file you finished processing last:

```python
names = trackedfiles[(trackedfiles.index(os.joindirs(AT.dirs["tracked"]),"lastcheckedfile.csv"))+1:]
```

### Key considerations

1. Not every single frame needs to be perfect or tracked as various of the processing functions can overcome this. Key data that is harder to add automatically is where animals make turns or large changes in their speed and there are no points of the key parts of that movement. For example if an animal is moving from point A to point B between frames 110 and 146 in a relatively straight line with constant speed, the position and speed for those intermediate frames can be easily automatically computed.
2. For state variables, such as if an animals goes from the open to under a mask, it is key to make sure all state changes during a trial are correct. Most importantly, make sure that when the start frame of the trial has been found, immideately  set the state the animal is in for that frame.
3. When trials contain a refuge (and corresponding mas) it is important to get the frames of exit and enterring the refuge and mark those same frames with the right state change.
4. Make sure you always click on roughly the same spot on the animal's body for consistency and especially consider this when the animal's body is not visible as normal, such as when jumping or partly under cover.
5. If datacrop is set to True the data will be cropped to the first and last frame that have been tracked. Therefore if your videos start and end with a period that should not considered as trial time use datacrop=True and make sure that the frame you'd like to consider as first frame has a value scored, either the position of the animal or it's state.

### Stand alone manual tracking

It is also possible to use the manual tracking functionality stand-alone, i.e. outside of a project folder. For this we need to load the manual tracking class directly:

```python
from atracker.tracker_man import Tracker_man
```

Here as an example we will work with one of the tutorial videos, and track the video from scratch.

```python
import os
vid_folder = os.chdir("folder_with_copied_solo_video")
vid_file = "animtest_solo_071024_S01_1413.mp4"
```

For this tutorial we will create a new tracking file for an animal with ID "F01", and enable the manual tracking of the centre and frontpoint of the animal. We don't want to crop the datafile to the frame range we will manually track but for the range from frame 10-5000. We resize the video at 150% to make it easier to see. And finally, we include the tracking of the state of the animal, which we label as "outofcover". So when the animal is hidden we change the state to 0 and when it is visible we change it to 1.

```python
# Set up the manual tracking instance
TM = Tracker_man(vidfile, fileaction = "newfile", ids = ["F01"], ptypes = ["c","f"], 
                  datacrop = False, firstframe = 10, lastframe = 5000, 
                  resizeval = 1.5, statevar = "outofcover")

# Run the tracking instance
TM.track()
```

In the same way we could manually track a couple videos without videos needing to be loaded by ATracker in anyway. For this we need to simply run the function in a python loop. For example:

```python
# import the needed package
from pythutils.fileutils import listfiles

# Get video names automatically
vidlist = listfiles(filedir = ".", filetype = ".mp4", dirs = False, keepdir = False)
print("Nr of videos:", len(vidlist), end = " - ")

# Run a loop over the files to manually track each file independently
for vidfile in vidlist:
    
    # Ask user for the ids in the vide
    ids = input("Enter IDs of the animals separated by comma's or press 'x' to exit: ")
    ids = ids.split(',')
    if ids == ["x"]:
        break
        
    else:
        # Set up the manual tracking instance
        TM = mantrack.Track_Manual(vidfile, fileaction = "newfile", ids = list(ids), ptypes = ["c"], 
                      safecount = False, datacrop = True, resizeval = 1.5, statevar = "outofcover")

        # Run the tracking instance
        TM.track()
```

## Final data processing

When happy with all the tracked, and potentially manually further fixed, files, it is time to process the final data. You have to decide if you need full frame data to be included in the file or not (`fulldata parameter`), if data should be converted from pixels to mm (`convert`), if outliers should be removed or not and with what criteria (`removeoutliers` and `alonewindow`). You should also decide how much you will smooth the data (`smoothwin`), which you can test with the manual tracking function, if you want ID information from the overview file to be added to the datafile (`addIDs`), and if besides computing the distance to the roi and potential walls also the distance to the center should be computed (`centerype`) and if data should be shifted to have the center at the origin (`centralise`).

Importantly, some of the key processing functionalities is the interpolation of missing data, in particular in relation to the tank edges and masks. For this there are a number of additional parameters you can set to get the most optimal outcome. They are mostly to deal with the more tricky situation where there are refuges and/or when the tracking data is fragmented. You can leave these at the defaults for now, but if there are clear errors with interpolating, trajectories, or the assessment if fish are under refuges or not, then play with the following parameters:

- `nearmaskdis`
- `inmaskdis`
- `edgedis`
- `trajgap`
- `mintrajlength`
- `filllenthresh_com`
- `filllenthresh_headtail`
- `headtoorientspeedthresh`
- `fillmissingorientdiffthresh`

> **Note**: for the processing function I will likely make further updates in the near future and parameter options and names may change, so double check this tutorial file as I will keep this up-to-date.

Processing may need quite some time to run per file due to the many checks and computations. So it may be best to try it with a couple individual files before running the loop on all files. In that case it is also helpful to set the pools value to something higher than 1, and run the command from the terminal (pools does not work in jupyter), such that processing can be run on individual cores and thereby sped up 4x or 8x for example.

After running the process function, you can find the final processed files in the "4processed" folder with the "_F" suffix. You can then run further summarise functions in R or Python to get to the datasets for analysis!
