import cv2, numpy as np, os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import cv2, numpy as np, os

img_size = 60 # has to be the same as for training
#chromakey values for the color green
h,s,v,h1,s1,v1 = 48,92,0,64,255,255
#parameter for segmenting image
pad = 60

#capture video
cap = cv2.VideoCapture('/home/stephen/Desktop/shapes/test.MP4')

#finds the largest contour in a list of contours
#returns a single contour
def largest_contour(contours): return max(contours, key=cv2.contourArea)[1]

#finds the center of a contour
#takes a single contour
#returns (x,y) position of the contour
def contour_center(c):
    M = cv2.moments(c)
    try: center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except: center = 0,0
    return center

#takes image and range
#returns parts of image in range
def only_color(img, (h,s,v,h1,s1,v1)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    kernel = np.ones((3,3), np.uint)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return res, mask

#returns the region of interest around the largest countour
#the accounts for objects not being centred in the frame
def bbox(img, c):
    x,y,w,h = cv2.boundingRect(c)
    return img[y-pad:y+h+pad, x-pad:w+x+pad], (x,y)

#import model
from keras.models import load_model
model = load_model('/home/stephen/Desktop/shapes/shapes_model.h5')
dimData = np.prod([img_size, img_size])

#main loop
while True:
    #read image from video, create a copy to draw on
    _, img= cap.read()
    imgc = img.copy()
    height, width, _ = img.shape

    #mask of the green regions in the image    
    _, mask = only_color(img, (h,s,v,h1,s1,v1))

    #find the contours in the image
    _, contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #iterate through the contours, "Keras, what shape is this contour?"
    for c in contours:
        #if the contour is too big or too small, it can be ignored
        area = cv2.contourArea(c)
        #print area
        if area> 3000 and area< 1180000:

            #crop out the green shape
            roi, coords = bbox(img, c)

            #filter out contours that are long and stringy
            if np.prod(roi.shape[:2])>10:

                #get the black and white image of the shape
                roi = cv2.resize(roi, (img_size, img_size))
                _, roi = only_color(roi, (h,s,v,h1,s1,v1))
                roi= 255-roi #Keras likes things black on white
                mask = cv2.resize(roi, (img_size, img_size))
                mask = mask.reshape(dimData)
                mask = mask.astype('float32')
                mask /=255
                #feed image into model
                prediction = model.predict(mask.reshape(1,dimData))[0].tolist()

                #create text --> go from categorical labels to the word for the shape.
                text = ''
                p_val, th = .25, .5
                if max(prediction)> p_val:
                    if prediction[0]>p_val and prediction[0]==max(prediction): text, th =  'triangle', prediction[0]
                    if prediction[1]>p_val and prediction[1]==max(prediction): text, th =  'star', prediction[1]
                    if prediction[2]>p_val and prediction[2]==max(prediction): text, th =  'square', prediction[2]
                    if prediction[3]>p_val and prediction[3]==max(prediction): text, th =  'circle', prediction[3]
                
                #draw the contour
                cv2.drawContours(imgc, c, -1, (0,0,255), 1)

                #draw the text
                org, font, color = (coords[0], coords[1]+int(area/400)), cv2.FONT_HERSHEY_SIMPLEX, (0,0,255)
                cv2.putText(imgc, text, org, font, int(2.2*area/15000), color, int(6*th), cv2.LINE_AA)

                #paste the black and white image onto the source image (picture in picture)
                if text!='': imgc[imgc.shape[0]-200:imgc.shape[0], img.shape[1]-200:img.shape[1]] = cv2.cvtColor(cv2.resize(roi, (200,200)), cv2.COLOR_GRAY2BGR)
                
    cv2.imshow('img', cv2.resize(imgc, (640, 480))) #expect 2 frames per second
    k = cv2.waitKey(100)
    if k == 27: break
    
cap.release()
cv2.destroyAllWindows()

