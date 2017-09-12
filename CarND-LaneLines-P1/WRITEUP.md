# **Finding Lane Lines on the Road** 


## Lane Markings
Identifying the lane markins is one of the basic step to a self drive car. here is one of the method to detect the lane marking using python and OpenCV libraries

---
## **Finding Lane lines on the road**
The goals of this project are
* Make a pipeline to find the lane lines on road on a image
* Tune the parameters and overlay lane markings overlay on image
* Extend the functionality on videos
* use the temporal information in videos to make the algorithm more robust
* write up on the method of implementation (Posted on [Medium](https://review.udacity.com/#!/rubrics/322/view) )

---
### **1.Find Lane Lines on a image**
The pipe line consists of the following steps 
* Convert input image from color to Gray scale
* Smooth image using gaussian blur
* Apply canny edge detection 
* Apply a trapezoidal region of interest 
* apply hough transform to detect the lane lines
* categorize the lane lines wrt to left/right/none lane lines
* average the lane lines 
* interpolate the lane lines to extend from the bottom of image to region of interest
* Overlay the lane lines on original image with a weight

[inputImage]: ./test_images_output/solidWhiteCurve.jpg 
[blurImage]: ./test_images_output/whiteCarLaneSwitch_02blur.jpg 
[edgeImage]: ./test_images_output/whiteCarLaneSwitch_03edge.jpg 
[edgesLineImage]: ./test_images_output/whiteCarLaneSwitch_07edgeslane.jpg 
[valLineImage]: ./test_images_output/whiteCarLaneSwitch_09vallane.jpg 
![alt text][inputImage]

### **2.Tune parameters**
* **Guassian blur**:  The input image is smoother with a gaussian blur kernel of size 11. as with any edge detection algorithm the smooth the gradient the better the edge detection. the gaussian blur kernel has trade off of processing time and edge rentention as kernel size increases
 ![alt text][blurImage]
*  **Canny Edge detection**: The canny edge detection is implemented with a gradient high threshold of 100 and low threshold of 30. this configuration helps the edge detector to classify bright pixels in gradient as edge and retain the not bright pixels connected to the bright ones
![alt text][edgeImage]
*  **ROI**: The region of interest for the lane markings is chosen as trapizodial from 3/5 of height to bottom and width having  taperig to the edges with a margin from edges.
*  **Hough Transform**: the hough transform is intersection threshold, min line length is choose to detect the lane lines and ignore other smaller lines. also the max line gap is choosen to bit around 1:3 ration to connect the lines.
![alt text][edgesLineImage]
* **Categorize Line segments**: The line segments detected from the hough transform is categorized as part of left or right lanes using the slope of line Right >0.5, left <-0.5
* **Average**:The categorized line segments are averaged in slope intercept form, as the slopes and intercepts are similar for line segments of same line  
* **Extrapolate**: once the average line is calculated the line is extrapolated from bottom of image to the region of interest using y=mx+b where m and b are known y is start and end of region of interest
![alt text][valLineImage]

### **3.Extend to Videos**
The same processing is extended to the videos processing every frame individually from the video and saved to video file

### **4.Temporal average**
On some frames on video the edge detection fails to detect the lines due to variations on road gradient and the lane markings are missing on overlay. in order to calculate the line marking on missing frames a temporal averaging can be used to across frames with history of 20 frames.

## **Shortcomings/Improvements**
The method works well on sample images and most time on videos, however there are some shortcomes when running on curve lanes which will need advance line fit algorithms to fit the segments of lines
* detect and categorize the continuous curve line as small line segments
* apply polynomial line fit on line segments part of same line 
* calculate the curvate of the lane  and extrapolate the to the region of interest


