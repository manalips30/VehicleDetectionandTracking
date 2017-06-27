##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Screenshots/Image1
[image2]: ./Screenshots/Image2.png
[image3]: ./Screenshots/IMage3.png
[image4]: ./Screenshots/Image4.png
[image5]: ./Screenshots/Image5.png
[image6]: ./Screenshots/Image6.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like using the code in fourth code cell 


![alt text][image6]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled up with the parameters that gave me the highest SVC test accuracy of 99.35%. The following are the parameters that I finally used

```
colorspace = 'YCrCb' 
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial = 32
histbin = 32
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I concatenated features extracted from the following steps
1. convert the image into YCrCb colorspace
2. extracted histogram
3. Used raw image by space binning to a size of (32, 32)
4. extracted HOG featured from all the channels of YCrCb colorspace image

I then trained a linearSVC using these features and obtained a testing accuracy of 99.35%

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the following windows to search for car. The code is implemented in code cell 9

```
windows = slide_window(img, x_start_stop=[250, None], y_start_stop=[400, 500], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))
windows += slide_window(img, x_start_stop=[250, None], y_start_stop=[400, 500], 
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
windows += slide_window(img, x_start_stop=[250, None], y_start_stop=[450, 578], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
windows += slide_window(img, x_start_stop=[250, None], y_start_stop=[450, None], 
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))
```
Here is an image to give you an ideo of windows used

![alt text][image2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap for every 6 frames and then thresholded that map by 6 to identify vehicle positions.  I then used blob detection in Sci-kit Image (Determinant of a Hessian [`skimage.feature.blob_doh()`](http://scikit-image.org/docs/dev/auto_examples/plot_blob.html) worked best for me) to identify individual blobs in the heatmap and then determined the extent of each blob using [`skimage.morphology.watershed()`](http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html). I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The code is implemented in code cell 11
I have pasted a small part of code for reference
```
if ProcessVideo.FrameCount>6:
        ProcessVideo.FrameCount = 0
        ProcessVideo.heatmap = apply_threshold(ProcessVideo.heatmap, 6)
        ProcessVideo.labels = label(ProcessVideo.heatmap)
        ProcessVideo.heatmap = np.zeros((720,1280))
        
window_img = draw_labeled_bboxes(img, ProcessVideo.labels)
```
###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues: I faced issue with getting rid of false positives. I had to play with the FrameCount and threshold a lot to come up with a robust vehicle detection

Likely Fail: The pipeline will fail if two cars are adjacent to each other. Both the cars will be identified as a single big vehicle

Solution: In order to make it robust we need to take into account that when we are detecting two cars continuously in the video for some time, one of the car cannot suddenly disspappear. So when the algorihm confuses when two cars are adjacent to each other we need to stick with old box coordinates or divide the one big box into two
