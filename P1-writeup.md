# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

Submitted: November 10, 2017 by Christopher Svec

The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report (you're reading it now)


[//]: # (Image References)

[orig]:./writeup-output-files/solidWhiteRight-0-orig.jpg "Original image"
[gray]:./writeup-output-files/solidWhiteRight-1-gray.jpg "Grayscale image"
[graymasked]:./writeup-output-files/solidWhiteRight-2-gray-masked.jpg "Masked grayscale image" 
[canny]:./writeup-output-files/solidWhiteRight-3-canny.jpg "Output of Canny transform"
[hough]:./writeup-output-files/solidWhiteRight-4-hough.jpg "All lines found by Hough transform"
[hough-oneline]:./writeup-output-files/solidWhiteRight-4-hough-oneline.jpg "Output of Hough transform with single line averages"
[histogram]:./writeup-output-files/solidWhiteRight-4-lines.jpg "Histogram of line parameters"
[final]:./writeup-output-files/solidWhiteRight-5-final.jpg "Final image"
[challenge-start]:./writeup-output-files/challenge-0-0-orig-final.jpg "Start of challenge video"
[challenge-bad]:./writeup-output-files/challenge-x31-0-orig-final.jpg "Challenge video starts to go badly"
[challenge-fail]:./writeup-output-files/challenge-fail.jpg "Challenge video fail"

---

## Reflection

My pipeline consisted of 5 steps:

1. Convert to grayscale
2. Remove unnecessary parts
3. Blur and run the Canny filter
4. Run Hough transform and identify lane lines
5. Draw lines on the original image

I'll describe each in turn, and I'll use the instructor-provided `solidWhiteRight.jpg` image as an example:

![alt_text][orig]

### 1. Convert to grayscale

The image was converted to grayscale using a simple OpenCV `cvtColor()` call.

This turned the 3-dimensional RGB image into a 1-dimensional grayscale image:

![alt_text][gray]

### 2. Remove unnecessary parts

Next I removed unnecessary parts of the image: we only care about the lane lines
right in front of the vehicle, so I removed the rest of image with a bounding
shape mask applied to the image. Any pixels outside of the bounding shape were
set to black:

![alt_text][graymasked]

The bounding shape dimensions was chosen by trial and error and by inspection.
The boundary lines locations in the shape were based on the image size.

### 3. Blur and run the Canny filter

Next I applied a Gaussian blur with `kernel=5` to provide a light blur to help
reduce noise and the appearance of non-line pixels.

Then I ran the Canny filter to find pixel color transitions (gradients) on the
image. The Canny filter output showed the parts of the image where colors
changed "quickly", where "quickly" was defined by the Canny filter parameters
`low_threshold = 50` and `high_threshold=150`.

In this Canny output image you can see that the lane lines are clearly visible,
along with some non-lane-lines above them:

![alt_text][canny]

The Canny parameters were chosen by trial and error and by inspection. It
resulted in visible lines with minimal noise.

### 4. Run Hough transform and identify lane lines

Next I ran the Hough transform to find the lines in the Canny output image.

The Hough transform finds all the line segments in the image based on these
Hough transform parameters:
```
    rho = 2              # distance resolution in pixels of the Hough grid
    theta = np.pi/180    # angular resolution in radians of the Hough grid
    threshold = 15       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 # minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
```

The Hough parameters were chosen by trial and error and by inspection. It
resulted in line segments that mostly matched lane lines.

This image shows the raw Hough output:

![alt_text][hough]

You can see that it matches the lane lines well. (There are some obvious
non-lane-lines at the top which I ignored using the algorithm described below.)

From the raw Hough output we wanted to find a single left and a single right
lane line. I thought that perhaps the slope of the lines would help me identify
the actual lane lines. I drew a histogram of the slopes of each of the lines found
by the Hough transform:

![alt_text][histogram]

The blue lines are the histogram data.

I saw that the lines that matched actual left lane lines tended to have slopes
between -0.9 and -0.5, and the lines that matched actual right lane lines tended
to have slopes between 0.5 and 0.7.

The red and black lines in the histogram are the average of the slopes within
those two ranges.

Here is the algorithm I used to find a single left line:

1. Extracted all lines with slopes between -0.9 and -0.5.
2. Take the average slope of all these lines, weighted evenly.
   * This is the slope of the single left lane line that we'll use.
3. For all the lines with slopes between -0.9 and -0.5, take the average of
   their y-intercepts, weighted evenly.
   * This is the y-intercept of the single left lane line that we'll use.
4. Draw the single left lane line using the slope and y-intercept:
   * The line starts at the bottom of the image, and goes up until the top of
     the bounding shape used in Step 2.

For the right lane, repeat the same procedure but take lines with slopes between
0.5 and 0.7.

The single left and right lane lines are shown here:

![alt_text][hough-oneline]

For video streams, the single lane slope and y-intercept were calculated using a
running average of the last 10 frames worth of slope and y-intercept data. This
produced a reasonably smooth and accurate lane line in a video.

### 5. Draw lines on the original image

The final step was to draw the single left and right lane lines on the original
image:

![alt_text][final]

Success!

## Results

My pipeline performed well for all the instructor-provided images, and the
`solidWhiteRight.mp4` and `solidYellowLeft.mp4` video files.

The lane lines found were relatively accurate.

The `challenge.mp4` video provided some, well, challenges.

The first part of `challenge.mp4` begins well enough, with my pipeline finding
the lanes reasonably well:

![alt_text][challenge-start]

But when the left lane becomes much lighter as the pavement changes, my pipeline
stops rising to the challenge:

![alt_text][challenge-bad]

It has a hard time seeing the light yellow lane line, and eventually it doesn't
find any lines from the Hough transform that are possible left lane candidates,
and it gives up:

![alt_text][challenge-fail]

## Possible Improvements

The biggest problem my pipeline had was finding low-contrast lane lines in the
challenge video. I experimented with different color filtering and color spaces
to try to get the light yellow lane to be more visible to the Canny filter, but
I didn't find a solution that worked.

## Conclusion

My lane finding pipeline worked reasonably well, and it taught me how to do
basic CV operations to process images.

Trial and error gave my pipeline good enough parameters to find lanes reasonably
well, but this project showed me that hand-tuning is probably insufficient for
processing all real-world situations.
