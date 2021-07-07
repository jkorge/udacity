# Lane Line Detection
#### James Korge
-------------------------------------------------------------------------------------------
The goal of this project is to develop, in Python, a pipeline which reads in an image and outputs a marked-up version of that image. The markings are meant to indicate the presence of lane lines on the road. The following is a brief report on the pipeline and how it was developed to find lane lines on an image of a road and includes a discussion on potential shortcomings as well as suggestions for further development.

## Pipeline

The pipeline consisted of 5 main steps:

1. Convert the image to grayscale
2. Apply a Gaussian blur
3. Apply a Canny edge-detection
4. Define a region of interest and create a masked copy of the image
5. Generate a sequence of Hough lines and draw them over the image

The first 3 steps were completed using OpenCV’s cvtColor, GaussianBlur, and Canny functions, respectively. The Gaussian blur was done with a kernel size of 5 x 5 and the Canny edge-detection was done with a low_threshold of 25 and a high_threshold of 100.

![Figure 1](images/canny-edges.jpg?raw=true "Sample output of Canny edge-detection")
Figure 1 Sample output of Canny edge-detection

The region of interest was hard-coded (upon inspection of the images and videos in question) to encompass a polygon ranging from approximately the lower two corners of the image to a pair of close-packed vertices approximately 60% down the images’ vertical extent. A masked image was then created by calling on OpenCV’s fillPoly function to create a polygon of white pixels which is combined with the Canny-edge image using OpenCV’s bitwise_and, thereby creating an image of regionally masked Canny edges.

![Figure 2](images/masked-canny-edges.jpg?raw=true "Output of regional masking when applied to Fig 1") 
Figure 2 Output of regional masking when applied to Fig 1

Finally, Hough line endpoints were determined using OpenCV’s HoughLinesP function. Initially, these endpoints were used to draw the Hough lines on a black background; that result was then added to the original input image thereby drawing Hough lines on the original image over the desired lane lines.

![Figure 3](images/hough-lines.jpg?raw=true "Output with Hough lines drawn over the original image")
Figure 3 Output with Hough lines drawn over the original image

The final process of drawing lines over the image was later updated. The updates consisted of calculating the slope and midpoint of each Hough line and using those data points to find an average slope and midpoint. These average values were used to define a pair of lines which would map to the desired lane lines.
The averages were determined by building two arrays: one for slopes and one for midpoints. The lines were separated by the sign of their slope: positive slope for the right lane line, negative slope for the left. The average midpoint was determined by calculating the average x and y coordinates for the slope-grouped Hough lines. Lastly, the final lines were calculated using the y-axis endpoints of the region of interest and the point-slope equation of the lines.

![Figure 4](images/hough-lines-final.jpg?raw=true "Final output with Hough line averages used to approximate both lane lines")
Figure 4 Final output with Hough line averages used to approximate both lane lines

## Potential Shortcomings

The most immediate shortcoming is the hard-coding of many parameters in this solution. The region of interest was defined based on the given images and would not map well to images produced from different setups. For example, when the hood of the car becomes visible (as in the challenge portion of the project) these parameters fail entirely. A similar failure is expected for any turns, steep hills, and heavy traffic.
Additionally, when this pipeline is applied to a video there appears to be transient misalignments between the drawn lines and the lane lines. This is possibly due to markings on the roads, changing traffic conditions, and variations in lighting. Dealing with such transience is critical in developing a stable pipeline.

## Suggested Improvements

A possible way to improve upon this is to dynamically construct the region of interest based on other parameters in the image. For example, if a constant horizontal line is present at the bottom of the image, this should be discounted from consideration as it probably represents the hood of the car.
Another possible improvement would be to average the lines in time as well as space. This would deal with many of the transient affectations which inhibit the drawn lines’ stability.

## Conclusion

The pipeline presented here represents an interesting start to the coursework and draws attention to many of the challenges faced when developing a self-driving car. The most obvious of these is the inconsistency of the car’s surroundings. This phenomenon produces transient errors in this image processing pipeline and will require more dynamic solutions to produce a truly robust lane-mapping algorithm.
