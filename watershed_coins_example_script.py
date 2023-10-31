import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#read image
img = cv.imread('water_coins.png')
assert img is not None, "file could not be read, check with os.path.exists()"

#convert graySclae
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#save grayScaleImage
cv.imwrite('grayscale_water_coins.png', gray)

#apply threshold
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#save thresholded image
cv.imwrite('thresholded_water_coins.png', thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

#save removedNoise image
cv.imwrite('noise_removed_water_coins.png', opening)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

#save background dilated image
cv.imwrite('background_dilated_water_coins.png', sure_bg)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)

# save distance transformed image
cv.imwrite('distance_transformed_water_coins.png', dist_transform)

# apply again treshold
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# save threshold applied image
cv.imwrite('distance_transformed_thresholded_water_coins.png', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# run watershed algorithm with markers
markers = cv.watershed(img,markers)

# save markered image
cv.imwrite('markered_image_water_coins.png', markers)


img[markers == -1] = [255,0,0]

#save object detected image
cv.imwrite('object_detected_water_coins.png', img)