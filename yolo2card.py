"""
Task: Detect card corners and fix perspective
"""


import cv2
import numpy as np


img = cv2.imread('yolo_detected/detected_object_1.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq_image = cv2.equalizeHist(gray)
blur = cv2.medianBlur(eq_image, 3)
_, thresh = cv2.threshold(blur, 100 , 255, cv2.THRESH_BINARY)


## Get contours
contours,h = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))


## only draw contour that have big areas
imx = img.shape[0]
imy = img.shape[1]
lp_area = (imx * imy) / 10



#################################################################
# Four point perspective transform
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#################################################################

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


#################################################################


## Get only rectangles given exceeding area
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01 * cv2.arcLength(cnt, True), True)


    if len(approx) >= 4 and cv2.contourArea(cnt) > lp_area:
        print("rectangle")

        # find points along contours, the points are, closest to the top left, top right, bottom left, bottom right of the image
        p1 = (0, 0)
        p2 = (img.shape[1], 0)
        p3 = (0, img.shape[0])
        p4 = (img.shape[1], img.shape[0])
        dist1 = float('inf')
        dist2 = float('inf')
        dist3 = float('inf')
        dist4 = float('inf')
        p1_contour = None
        p2_contour = None
        p3_contour = None
        p4_contour = None
        for p in cnt.reshape(-1, 2):
            d1 = np.linalg.norm(p - p1)
            if d1 < dist1:
                dist1 = d1
                p1_contour = p
            d2 = np.linalg.norm(p - p2)
            if d2 < dist2:
                dist2 = d2
                p2_contour = p
            d3 = np.linalg.norm(p - p3)
            if d3 < dist3:
                dist3 = d3
                p3_contour = p
            d4 = np.linalg.norm(p - p4)
            if d4 < dist4:
                dist4 = d4
                p4_contour = p

        # Perform a perspective transform using the points closest to the image corners
        tmp_img = img.copy()
        pts = np.array([p1_contour, p2_contour, p3_contour, p4_contour], dtype="float32")
        warped = four_point_transform(tmp_img, pts)
        cv2.imshow("Warped Perspective", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()