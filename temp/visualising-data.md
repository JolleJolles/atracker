# Visualising data

TO BE UPDATED

After processing the video, the processed data can be visualised using the *visualise* function. A lot of options exist:

- You can set custom start and stopframes using the *startframe* and *stopframe* parameters. If left at *None* then it will use the frames of the data 
- Do you want to see the video while it is being created? Set *showvideo* to True. If you don't want a video to be created set *writevideo* to False.
- You can set the video suffix with the *videosuffix* parameter (e.g. "_V").
- It is possible to skip frames and thereby get a video with a higher framerate using the *framestep* parameter and keeping the video at a lower size. Alternatively set the *fps* parameter, which has the same number of frames as the original video but just played at this speed.
- To show a cropped image to the region of interest (roi) set the *cropimg* parameter to True. Otherwise the full original video size will be used. To show the roi as a black border set the *drawroi* parameter to True.
- To resize the image to have a smaller or larger video set the *resize* parameter to a value smaller or larger than 1. You can use resizing to get a smoother image quality. To keep the original video dimensions and thus the video is only temporarily increased in size, set the *smoothresize* parameter to True.
- It is possible to just show the actual video frame, to hide the background, or to only show the tracking data on top of a white background. This can be done by setting the *img_bg* parameter to *None*, *img_bg* or to *"white"*.
- To draw all objects as black shape set the *drawobjects* to True. For this contour data should be provided with the tracking file (!This is not yet fully implemented)
- To draw trajectories the *drawtrajs* parameter should be true. Unique colors will be given to all tracked objects.
- It is possible to draw the trajectories behind the objects using the *drawtrajsbehind* parameter, however, for this the tresholded image should be provided (*img_tresh*).
