#!/usr/bin/env python3

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
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
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, color=[255,0,0], thickness=4)
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

    top_of_mask = 310
    lower_left  = (140, height-1)
    upper_left  = (475, top_of_mask)
    upper_right = (500, top_of_mask)
    lower_right = (860, height-1)

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
    output_filename_no_suffix = os.path.join(output_dir_name, base_filename)

    print("#", base_filename, "#", file_ext, "#", output_filename_no_suffix, "#")

    print('This image is:', type(image), 'with dimensions:', image.shape)
    #imshow_full_size(image, title=filename)
    mpimg.imsave(output_filename_no_suffix + "-0-orig" + file_ext, image)

    # To show multiple images, call plt.figure() start a new figure.
    # Subsequent calls to plt.imshow() will appear in a new window.

    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #imshow_full_size(gray, cmap='gray')
    mpimg.imsave(output_filename_no_suffix + "-1-gray" + file_ext, gray, cmap='gray')

    top_of_mask = 310
    lower_left  = (140, height-1)
    upper_left  = (475, top_of_mask)
    upper_right = (500, top_of_mask)
    lower_right = (900, height-1)

    bounding_shape = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
    masked_gray = region_of_interest(gray, bounding_shape)
    #imshow_full_size(masked_gray, cmap='gray')
    mpimg.imsave(output_filename_no_suffix + "-2-gray-masked" + file_ext, masked_gray, cmap='gray')

    kernel = 5
    blur_gray = gaussian_blur(gray, kernel)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_canny = region_of_interest(edges, bounding_shape)
    #imshow_full_size(masked_canny, cmap='gray')
    mpimg.imsave(output_filename_no_suffix + "-3-canny" + file_ext, masked_canny, cmap='gray')

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap)
    #imshow_full_size(houghed_img, cmap='gray')
    mpimg.imsave(output_filename_no_suffix + "-4-hough" + file_ext, houghed_img)

    # Draw the lines on the edge image
    #lines_edges = cv2.addWeighted(color_edges, 0.8, houghed_img, 1, 0) 
    #imshow_full_size(lines_edges, cmap='gray')
    lines_edges = cv2.addWeighted(image, 0.8, houghed_img, 1, 0) 
    #imshow_full_size(lines_edges);
    mpimg.imsave(output_filename_no_suffix + "-5-final" + file_ext, lines_edges);

    #plt.subplots_adjust(left=0.0, bottom=0, right=1, top=1,
                    #wspace=0.02, hspace=0.02)
    #plt.show()

def file_loop():
    input_dir_name = "test_images"
    image_files = [f for f in os.listdir(input_dir_name) if os.path.isfile(os.path.join(input_dir_name, f))]
    image_files = [os.path.join(input_dir_name, f) for f in image_files]
    print(image_files)

    for filename in image_files:
        find_lanes_in_file(filename, "test_images_output")

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    height, width, depth = image.shape

    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    top_of_mask = 317
    lower_left  = (140, height-1)
    upper_left  = (475, top_of_mask)
    upper_right = (500, top_of_mask)
    lower_right = (900, height-1)

    bounding_shape = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
    masked_gray = region_of_interest(gray, bounding_shape)

    kernel = 5
    blur_gray = gaussian_blur(gray, kernel)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    masked_canny = region_of_interest(edges, bounding_shape)

    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments

    houghed_img = hough_lines(masked_canny, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the original image
    lines_edges = cv2.addWeighted(image, 0.8, houghed_img, 1, 0) 

    return lines_edges

def movie(filename, output_dir_name):

    base_filename, file_ext = os.path.splitext(os.path.basename(filename))
    output_filename_no_suffix = os.path.join(output_dir_name, base_filename)
    print("#", base_filename, "#", file_ext, "#", output_filename_no_suffix, "#")

    output = output_filename_no_suffix + file_ext # 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip = VideoFileClip(filename)
    processed_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip.write_videofile(output, audio=False)

def movies_loop():
    input_dir_name = "test_videos"
    video_files = [f for f in os.listdir(input_dir_name) if os.path.isfile(os.path.join(input_dir_name, f))]
    video_files = [
                   "test_videos/solidWhiteRight.mp4",
                   "test_videos/solidYellowLeft.mp4",
                   "test_videos/challenge.mp4",
    ]
    print(video_files)

    for filename in video_files:
        movie(filename, "test_videos_output")

#main_loop()
#kernels_loop()
#file_loop()
movies_loop()

