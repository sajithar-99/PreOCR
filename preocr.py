#*****************************************************************************#
# <ALGORITHM>
#1. Import packages and Parse arguments
#2. Load Image specified in Argument
#3. Convert to GreyScale and Invert
#4. THRESH_OTSU for foreground background detection (OR use erosion and THRESH_BINARY)
#5. Find minimum bounding rectangle
#6. Rotate to deskew
#7. Show angle + output and save
#
#*****************************************************************************#


import numpy as np
import argparse
import cv2

#Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())

# load image
image = cv2.imread(args["image"])
#----------------------------------D.E.S.K.E.W---------------------------------
#BGR-->Gray then Invert
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

#THRESH_OTSU - setthing foreground to 255 and background to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#Find minimum bounding rectangle
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

#find the angle

if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#display angle over image
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
#	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#------------------------------C.O.N.T.R.A.S.T----------------------------------

img = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
cl1 = clahe.apply(img)
#------------------------------S.H.A.R.P.E.N------------------------------------

out = bilFilter = cv2.bilateralFilter(cl1,9,20,20)

#------------------------------------------------------------------------------
#output and save

print("[INFO] angle: {:.2f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Output", out)
cv2.imwrite('preocr_res.png',out)
cv2.waitKey(0)
