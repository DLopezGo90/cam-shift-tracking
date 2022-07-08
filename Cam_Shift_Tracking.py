import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def stackImages(scale,imgArray):
#function to stack images

    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cap = cv2.VideoCapture(r'C:\Users\dlopez\Desktop\Github\Cam_Shift_Tracking\car_racing.mp4')
#link to the video as argument

ret, frame = cap.read()
#ret is a boolean value (True if there's success taking a frame, False otherwise)

width = int(cap.get(3))
#video width

height = int(cap.get(4))
#video height

out = cv2.VideoWriter(r'C:\Users\dlopez\Desktop\Github\Cam_Shift_Tracking\car_racing_cam_shift.avi', 
                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
#path for the output (we are creating a video)
#codec used for compressing frames
#fps (frames per second) of the video we are creating
#framesize

x, y, w, h = 0, 400, 400, 150
#parameters of initial roi (where the car is at the beginning of the video)
#x and y coordinates of left upper vertex
#w and h width and height of roi

track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
#region of interest 

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#converting roi to hsv space color

mask = cv2.inRange( hsv_roi, np.array((15, 127, 127)), np.array((30, 255, 255)) )
#inside hsv_roi we are only interested in pixels that meet these conditions:
# 15 <= hue <= 30
# 127 <= sat <= 255
# 127 <= val <= 255
#mask is a numpy array that only has black and white pixels 
#pixels in mask are white if they meet the conditions
#black otherwise

result = cv2.bitwise_and(roi, roi, mask = mask)
#result is a 3 channels numpy array 
#the difference with mask is that pixels that are white in mask have their original colors in result

imgStack = stackImages(1,([roi, hsv_roi], [mask,result]))
#using the stackImages function we defined at the beginning

cv2.imshow('Stacked Images', imgStack)
#showing roi, hav_roi, mask and result

cv2.waitKey(0)
#this line is necessary 
#execution of the script won't continue till a keycap is pressed 

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
#[0] because we only care about the hue channel in this case
#mask as argument, this way only pixels that met the conditions are considered 
#180 bins because there are 180 possible values (0-179)
#[0-180] are the possible hue values (not 179 as upper limit because it is not considered) 
#cv2.calcHist gives us the absolute frequency (not normalized values)

print(type(roi_hist))
#roi_hist is a numpy array
#shape (180,1)
#absolute frequency per bin

#flatten = roi_hist.flatten()

#print(flatten.shape)

#plt.bar(np.arange(180), flatten)

#plt.show()

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
#why 0 and 255?
#this way we transform the bigger frequency to 255
#the bins with no frequency continue being 0
#this way when taking a video frame, we convert it to grayscale and the gray values depend on this hue-values correlation

#print(roi_hist)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#10 iterations as maximum when moving the window
#if the algorithm proposes a window movement of 1 pixel distance, stop iteraing

while(True):
#this loop will end after the last frame of the video

    ret, frame = cap.read()
    #ret is a boolean value. True if there's success taking a frame, False otherwise

    if ret == True:
    #if there was success taking a frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #converting the frame to hsv colorspace

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        #converted frame to hsv as input
        #we will work with the 1st channel (hue)
        #roi_hist give us the values to assign to each hue value
        #[0, 180] so it knows how to do the assignation
        #scale = 1?

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        #track_window get the values of the window after converging

        x, y, w, h = track_window
        #we get the values of track_window

        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)
        #img2 is an image created from the processed frame
        #we draw the rectangle that represents the window after converging
        #white color
        #2 as thickness value

        out.write(img2)
        #out means output I guess
        #this will add the frame to the video we are creating

    else:
    #if there are no more frames

        break
        #end this loop

cap.release()

out.release()

