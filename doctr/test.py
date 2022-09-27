import numpy as np
import cv2
import imutils

import cv2




def do(frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray_frame, (5, 5), 1)
    edge_image_blurred = cv2.Canny(img_blurred, 60, 180)
    cv2.imshow("Edge image", edge_image_blurred)


    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(edge_image_blurred, kernel, iterations=2)
    cv2.imshow('ImgDilated', imgDil)


    key_points = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(key_points)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)



    biggest_contour = np.array([])
    for contour in contours:
       area = cv2.contourArea(contour)
       if area > 10000:
           perimeter = cv2.arcLength(contour, True)
           approximation = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
           if len(approximation) == 4:
               biggest_contour = approximation



    if biggest_contour.size != 0:
       cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 15)




    if biggest_contour.size != 0:
       cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 20)

       pts1 = np.float32(biggest_contour)
       pts2 = np.float32([[0, 0], [0, 512], [512, 512], [512, 0]])
       P = cv2.getPerspectiveTransform(pts1, pts2)
       image_warped = cv2.warpPerspective(frame, P, (512, 512))

       cv2.imshow('Warped image', image_warped)


       cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 20)
       pts1 = np.float32(biggest_contour)
       pts2 = np.float32([[812, 0], [0, 0], [0, 812], [812, 812]])
       P = cv2.getPerspectiveTransform(pts1, pts2)
       image_warped = cv2.warpPerspective(gray_frame, P, (812, 812))
       binary_image = cv2.adaptiveThreshold(image_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71,                               9)
       cv2.imshow('frame', binary_image)



recording = cv2.VideoCapture(2)

while True:

   ret, frame = recording.read() #ret is True or False value depending on whether we successfully got an image from recording
   cv2.imshow('Frame', frame)

   do(frame)

   if cv2.waitKey(1) == ord('x'):
       break

cv2.destroyAllWindows()

