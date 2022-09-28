import cv2, numpy as np, os

img_size = 200 #size of training images
#chromakey values for green
h,s,v,h1,s1,v1 = 48,92,0,64,255,255
#parameter for segmenting images
pad = 60

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
def only_color(img, (h,s,v,h1,s1,v1), pad):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3,3), np.uint)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

#returns the region of interest around the largest countour
#the accounts for objects not being centred in the frame
def bbox(img, c):
    x,y,w,h = cv2.boundingRect(c)
    return img[y-pad:y+h+pad, x-pad:w+x+pad]

#paths to source video
data_paths = ['/home/stephen/Desktop/shapes/triangle.MP4',
              '/home/stephen/Desktop/shapes/square.MP4',
              '/home/stephen/Desktop/shapes/circle.MP4',
              '/home/stephen/Desktop/shapes/star.MP4']
#paths to folders where training data will be stored
folder_names = ['/home/stephen/Desktop/shapes/triangle/0/',
                '/home/stephen/Desktop/shapes/square/1/',
                '/home/stephen/Desktop/shapes/circle/2/',
                '/home/stephen/Desktop/shapes/star/3/']

for data_path, folder_name in zip(data_paths, folder_names):
    frame_num = 0
    cap = cv2.VideoCapture(data_path)
    while True:
        _, img= cap.read()
        try: height, width, _ = img.shape
        except: break
        #get a mask of the image to remove the green background
        mask = only_color(img, (h,s,v,h1,s1,v1), pad)
        #find the contours in the image
        _, contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sort the contours by area
        contours = sorted(contours, key=cv2.contourArea)
        #if there are any contours, continue
        if len(contours)>0:
            #crop out the largest contour (which should be the shape)
            crop = bbox(mask, contours[-1])
            #if the shape is not too long and thin, continue
            if np.prod(crop.shape)!=0:
                crop = cv2.resize(crop, (img_size, img_size))
                crop = 255-crop
                cv2.imshow('img', crop)
                cv2.imwrite(folder_name+str(frame_num)+'.png', crop)
                frame_num += 1
                cv2.waitKey(1)
        cv2.imshow('img1', cv2.resize(mask, (640,480)))
        if frame_num%250==0: print frame_num,'----------------------------'               
    cap.release()
cv2.destroyAllWindows()


