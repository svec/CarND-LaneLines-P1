#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import statistics
#%matplotlib inline

import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

g_top_of_mask = 0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def line_fit(x1,y1,x2,y2):
    rise = y2 - y1
    run  = x2 - x1

    slope = -100
    if run != 0:
        slope = rise / run

    # y = mx + b
    # b = y - mx
    y_intercept = y1 - (slope * x1)

    return (slope, y_intercept)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2, single_line=False):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #print("drawing %d lines" % (len(lines)))
    height, width, depth = img.shape

    stop_at = 100

    list_lengths = []
    list_slopes = []
    list_yints = []

    for index, line in enumerate(lines):
        #print("line #", index)
        for x1,y1,x2,y2 in line:
            slope, y_intercept = line_fit(x1,y1,x2,y2)
            line_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            #print("(%d, %d) (%d, %d) len: %5.2f slope: %5.2f y_int: %5.2f" % (x1, y1, x2, y2, line_len, slope, y_intercept))
            list_lengths.append(line_len)
            list_slopes.append(slope)
            list_yints.append(y_intercept)

            if single_line == False:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            #if index >= stop_at:
                #break
        #if index >= stop_at:
            #break

    if single_line == False:
        return


    global g_debug_frame
    global g_debug_output_filename_no_ext
    global g_debug_frame_count
    if g_debug_frame:
        output_file_prefix = g_debug_output_filename_no_ext
        if g_debug_frame_count != "":
            output_file_prefix = output_file_prefix + "-" + str(g_debug_frame_count)

        nbins = 100

        plt.figure()
        plt.subplot(3,1,1)
        plt.hist(list_slopes, nbins, range=(-0.1, 2))
        plt.title('slopes')
        plt.grid()

        plt.subplot(3,1,2)
        plt.hist(list_yints, nbins)
        plt.title('yints')
        plt.grid()
        
        plt.subplot(3,1,3)
        plt.hist(list_lengths, nbins)
        plt.title('lengths')
        plt.grid()
        


    left_index_slopes =  [(index, value) for index, value in enumerate(list_slopes) if ((value > -0.9) and (value < -0.5))]
    right_index_slopes = [(index, value) for index, value in enumerate(list_slopes) if ((value >  0.5) and (value < 0.7))]

    if (len(left_index_slopes) < 2) or (len(right_index_slopes) < 2):
        cv2.putText(img, "ERROR FINDING LANES", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=3)

        if (len(left_index_slopes) < 2): 
            print("ERROR: left_index_slopes too small ")
        if (len(right_index_slopes) < 2): 
            print("ERROR: right_index_slopes too small ")
        print("ERROR: list_lengths:" , list_lengths)
        print("ERROR: list_slopes:" , list_slopes)
        print("ERROR: list_yints:" , list_yints)
        if g_debug_frame:
            plt.savefig(output_file_prefix + "-4-lines-ERROR.jpg")

                
        return

    left_indices,  left_slopes  = zip(*left_index_slopes)
    right_indices, right_slopes = zip(*right_index_slopes)
    left_mean  = statistics.mean(left_slopes)
    right_mean = statistics.mean(right_slopes)

    left_y_ints = [list_yints[i] for i in left_indices]
    left_y_int_mean = statistics.mean(left_y_ints)
    right_y_ints = [list_yints[i] for i in right_indices]
    right_y_int_mean = statistics.mean(right_y_ints)

    if g_debug_frame:
        plt.subplot(3,1,1)

        plt.axvline(x=left_mean, linewidth=1, color='r')
        plt.axvline(x=right_mean, linewidth=1, color='k')

        plt.subplot(3,1,2)
        
        plt.axvline(x=left_y_int_mean, linewidth=1, color='r')
        plt.axvline(x=right_y_int_mean, linewidth=1, color='k')

        plt.savefig(output_file_prefix + "-4-lines.jpg")

    if single_line:
        global g_top_of_mask

        draw_full_line = False

        color = [0, 255, 0]
        # We now have the slope and y-intercept of the 2 lines we want to draw.
        if draw_full_line:
            x1 = 0
            y1 = int(round(left_y_int_mean))
            y2 = 0
            x2 = int(round((y2 - left_y_int_mean) / left_mean))
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], thickness)

            x1 = 0
            y1 = int(round(right_y_int_mean))
            y2 = height-1
            x2 = int(round((y2 - y1) / right_mean))
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], thickness)

        x1 = 0
        y1 = int(round(left_y_int_mean))
        y2 = g_top_of_mask
        x2 = int(round((y2 - left_y_int_mean) / left_mean))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        y1 = g_top_of_mask
        x1 = int(round((y1 - right_y_int_mean) / right_mean))
        y2 = height-1
        x2 = int(round((y2 - right_y_int_mean) / right_mean))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, single_line=False):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, color=[255,0,0], thickness=4, single_line=single_line)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def imshow_full_size(img, title=False, *args, **kwargs):
    dpi = 100
    margin = 0.05 # (5% of the width/height of the figure...)
    ypixels, xpixels = img.shape[0], img.shape[1]
    
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * xpixels / dpi, (1 + margin) * ypixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    ax.imshow(img, interpolation='none', *args, **kwargs)
    if title:
        plt.title(title)
    plt.show()

def main_loop():
    #reading in an image
    image = mpimg.imread('test_images/solidWhiteRight.jpg')

    height, width, depth = image.shape

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    #imshow_full_size(image)

    # To show multiple images, call plt.figure() start a new figure.
    # Subsequent calls to plt.imshow() will appear in a new window.

    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #imshow_full_size(gray, cmap='gray')

    use_hardcoded_for_960x540 = True

    if use_hardcoded_for_960x540:
        top_of_mask = 310
        lower_left  = (140, height-1)
        upper_left  = (475, top_of_mask)
        upper_right = (500, top_of_mask)
        lower_right = (860, height-1)
    else:
        top_of_mask = int(round(0.58 * height))
        lower_left  = (int(round(0.15 * width)), height-1)
        upper_left  = (int(round(0.45 * width)), top_of_mask)
        upper_right = (int(round(0.60 * width)), top_of_mask)
        lower_right = (int(round(0.91 * width)), height-1)

    bounding_shape = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
    masked_gray = region_of_interest(gray, bounding_shape)
    #imshow_full_size(masked_gray, cmap='gray')

    #kernels_loop(gray, bounding_shape)
    hough_loop(gray, bounding_shape)

def kernels_loop(gray, bounding_shape):
    plt.figure()
    # Define a kernel size and apply Gaussian smoothing
    kernels = [5]
    subplot_rows = 3
    subplot_cols = len(kernels)
    for index, kernel in enumerate(kernels):
        subplot_index = index+1
        plt.subplot(subplot_rows, subplot_cols,subplot_index)
        blur_gray = gaussian_blur(gray, kernel)
        plt.title(subplot_index)
        plt.imshow(blur_gray, cmap='gray')
        
        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = canny(blur_gray, low_threshold, high_threshold)
    
        #plt.subplot(subplot_rows, subplot_cols,index+(subplot_cols*1)+1)
        #plt.imshow(edges, cmap='gray')
        #imshow_full_size(edges, cmap='gray')
        #imshow_full_size(edges, cmap='Greys_r') # Udacity quiz uses Greys_r
    
        masked_canny = region_of_interest(edges, bounding_shape)
        plt.subplot(subplot_rows, subplot_cols,index+(subplot_cols*1)+1)
        plt.imshow(masked_canny, cmap='gray')
        #imshow_full_size(masked_canny, cmap='gray')
    
        # Define the Hough transform parameters
        # Original from offical lecture 'solution'
        #   rho = 2 # distance resolution in pixels of the Hough grid
        #   theta = np.pi/180 # angular resolution in radians of the Hough grid
        #   threshold = 15     # minimum number of votes (intersections in Hough grid cell)
        #   min_line_length = 30 #minimum number of pixels making up a line
        #   max_line_gap = 20    # maximum gap in pixels between connectable line segments
        #
        # My values determined by trial and error
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 15     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10 #minimum number of pixels making up a line
        max_line_gap = 10    # maximum gap in pixels between connectable line segments
        houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap)
    
        #plt.subplot(subplot_rows, subplot_cols,index+(subplot_cols*3)+1)
        #plt.imshow(houghed_img)
        #imshow_full_size(houghed_img)
    
        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges)) 
    
        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, houghed_img, 1, 0) 
        plt.subplot(subplot_rows, subplot_cols,index+(subplot_cols*2)+1)
        plt.imshow(lines_edges)
        #imshow_full_size(lines_edges)
    
    plt.subplots_adjust(left=0.0, bottom=0, right=1, top=1,
                    wspace=0.02, hspace=0.02)
    plt.show()

def hough_loop(gray, bounding_shape):
    kernel = 5
    blur_gray = gaussian_blur(gray, kernel)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_canny = region_of_interest(edges, bounding_shape)

    plt.figure()

    subplot_rows = 2
    params = [5,7,10,12,15]
    subplot_cols = 3
    subplot_index = 1

    plt.subplot(subplot_rows, subplot_cols, subplot_index)
    subplot_index = subplot_index + 1
    plt.imshow(masked_canny, cmap='gray')

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    for index, value in enumerate(params):
        houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap)

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, houghed_img, 1, 0) 
        plt.subplot(subplot_rows, subplot_cols, subplot_index)
        subplot_index = subplot_index + 1
        plt.imshow(lines_edges)
        plt.title(value)

    plt.subplots_adjust(left=0.0, bottom=0, right=1, top=1,
                    wspace=0.02, hspace=0.02)
    plt.show()

import os

def find_lanes_in_file(filename, output_dir_name):
    image = mpimg.imread(filename)
    height, width, depth = image.shape

    base_filename, file_ext = os.path.splitext(os.path.basename(filename))
    output_filename_no_ext = os.path.join(output_dir_name, base_filename)

    print("#", base_filename, "#", file_ext, "#", output_filename_no_ext, "#")

    global g_debug_frame
    global g_debug_output_filename_no_ext
    global g_debug_frame_count

    g_debug_frame = True
    g_debug_output_filename_no_ext = output_filename_no_ext
    g_debug_frame_count = ""

    process_image(image)


def file_loop():
    input_dir_name = "test_images"
    image_files = [f for f in os.listdir(input_dir_name) if os.path.isfile(os.path.join(input_dir_name, f))]
    image_files = [os.path.join(input_dir_name, f) for f in image_files]
    print(image_files)

    #find_lanes_in_file("test_images/solidWhiteRight.jpg", "test_images_output")
    #return

    for filename in image_files:
        find_lanes_in_file(filename, "test_images_output")

g_debug_frame = False
g_debug_output_filename_no_ext = ""
g_debug_frame_count = ""

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    height, width, depth = image.shape

    output_file_prefix = "FIXME"

    global g_debug_frame
    global g_debug_frame_count
    if g_debug_frame:
        output_file_prefix = g_debug_output_filename_no_ext
        if g_debug_frame_count != "":
            output_file_prefix = output_file_prefix + "-" + str(g_debug_frame_count)
        mpimg.imsave(output_file_prefix + "-0-orig.jpg", image)
    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if g_debug_frame:
        mpimg.imsave(output_file_prefix + "-1-gray.jpg", gray, cmap='gray')

    use_hardcoded_for_960x540 = False

    global g_top_of_mask
    if use_hardcoded_for_960x540:
        g_top_of_mask = 317
        lower_left  = (140, height-1)
        upper_left  = (450, g_top_of_mask)
        upper_right = (500, g_top_of_mask)
        lower_right = (900, height-1)
    else:
        g_top_of_mask = int(round(0.59 * height))
        lower_left  = (int(round(0.15 * width)), height-1)
        upper_left  = (int(round(0.40 * width)), g_top_of_mask)
        upper_right = (int(round(0.60 * width)), g_top_of_mask)
        lower_right = (int(round(0.91 * width)), height-1)

    bounding_shape = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
    masked_gray = region_of_interest(gray, bounding_shape)

    if g_debug_frame:
        mpimg.imsave(output_file_prefix + "-2-gray-masked.jpg", masked_gray, cmap='gray')

    kernel = 5
    blur_gray = gaussian_blur(gray, kernel)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_canny = region_of_interest(edges, bounding_shape)

    if g_debug_frame:
        mpimg.imsave(output_file_prefix + "-3-canny.jpg", masked_canny, cmap='gray')

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap, single_line=False)

    if g_debug_frame:
        mpimg.imsave(output_file_prefix + "-4-hough.jpg", houghed_img)

    houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap, single_line=True)

    if g_debug_frame:
        mpimg.imsave(output_file_prefix + "-4-hough-oneline.jpg", houghed_img)

    # Draw the lines on the original image
    lines_edges = cv2.addWeighted(image, 0.8, houghed_img, 1, 0) 

    if g_debug_frame:
        if g_debug_frame_count != "":
            g_debug_frame_count = g_debug_frame_count + 1
        mpimg.imsave(output_file_prefix + "-5-final.jpg", lines_edges)
        #g_debug_frame = False

    return lines_edges

def movie(filename, output_dir_name, debug=False):

    base_filename, file_ext = os.path.splitext(os.path.basename(filename))
    output_filename_no_ext = os.path.join(output_dir_name, base_filename)
    print("#", base_filename, "#", file_ext, "#", output_filename_no_ext, "#")

    output = output_filename_no_ext + file_ext

    global g_debug_frame
    global g_debug_output_filename_no_ext
    global g_debug_frame_count
    g_debug_frame = False

    if debug:
        g_debug_frame = True
        g_debug_output_filename_no_ext = output_filename_no_ext
        g_debug_frame_count = 0

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip = VideoFileClip(filename).subclip(0,5)
    #clip = VideoFileClip(filename)
    processed_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip.write_videofile(output, audio=False)

def movies_loop():
    input_dir_name = "test_videos"
    video_files = [f for f in os.listdir(input_dir_name) if os.path.isfile(os.path.join(input_dir_name, f))]
    video_files = [
                   #"test_videos/solidWhiteRight.mp4",
                   #"test_videos/solidYellowLeft.mp4",
                   "test_videos/challenge.mp4",
    ]
    print(video_files)

    for filename in video_files:
        movie(filename, "test_videos_output", debug=True)

#main_loop()
#kernels_loop()
file_loop()
#movies_loop()
