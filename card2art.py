import cv2
import numpy as np
import time

def extract_artwork(image, thresh_val=180, img_size=56):
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eq_image = cv2.equalizeHist(gray_image)

    _, binary_image = cv2.threshold(eq_image, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    eroded_image = cv2.erode(binary_image,kernel,iterations = 1)
    eroded_image = cv2.erode(eroded_image,kernel,iterations = 1)
    eroded_image = cv2.erode(eroded_image,kernel,iterations = 1)

    dilated_image = cv2.dilate(eroded_image,kernel,iterations = 1)
    dilated_image = cv2.dilate(dilated_image,kernel,iterations = 1)
    dilated_image = cv2.dilate(dilated_image,kernel,iterations = 1)

    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    x,y,w,h = cv2.boundingRect(max_contour)
    if image.shape[0] > image.shape[1]:
        if (image.shape[0] - (y + h)) < y:
            art = image[int((h*4)*0.2):y, x:x+w]
        else:
            art = image[y+h:int(((y+h)*4)*0.8), x:x+w]
    else:
        if x < (image.shape[1] // 2):
            art = image[y:y+h, x+w:int((w*4)*0.8)]
        else:
            art = image[y:y+h, int((w*4)*0.2):x]

    art = cv2.resize(art, (img_size, img_size))
    
    return art

image_path = 'yolo_detected/detected_object_5.jpg'

image = cv2.imread(image_path)
total_time = 0
for _ in range(1000):
    start = time.time()
    art = extract_artwork(image)
    end = time.time()
    total_time += (end - start)
print("Average run time: {} seconds".format(total_time / 1000))

cv2.imshow('Artwork', art)

cv2.waitKey(0)
cv2.destroyAllWindows()