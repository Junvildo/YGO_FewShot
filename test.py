import cv2
import numpy as np
import glob
import time

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def extract_artwork(image, thresh_val=180, img_size=56):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_image = cv2.equalizeHist(gray_image)
    _, binary_image = cv2.threshold(eq_image, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.erode(binary_image, kernel, iterations=3)
    processed_image = cv2.dilate(processed_image, kernel, iterations=3)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    if image.shape[0] > image.shape[1]:
        art = image[int((h * 4) * 0.2):y, x:x + w] if (image.shape[0] - (y + h)) < y else image[y + h:int(((y + h) * 4) * 0.8), x:x + w]
    else:
        art = image[y:y + h, x + w:int((w * 4) * 0.8)] if x < (image.shape[1] // 2) else image[y:y + h, int((w * 4) * 0.2):x]
    
    return cv2.resize(art, (img_size, img_size))

def extract_art_from_image(img):
    # Preprocess and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_image = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(eq_image, 3)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Get contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    imx, imy = img.shape[:2]
    lp_area = (imx * imy) / 10

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 4 and cv2.contourArea(cnt) > lp_area:
            p1, p2, p3, p4 = (0, 0), (img.shape[1], 0), (0, img.shape[0]), (img.shape[1], img.shape[0])
            closest_points = [None] * 4
            distances = [float('inf')] * 4
            for p in cnt.reshape(-1, 2):
                for i, corner in enumerate([p1, p2, p3, p4]):
                    d = np.linalg.norm(p - corner)
                    if d < distances[i]:
                        distances[i] = d
                        closest_points[i] = p
            pts = np.array(closest_points, dtype="float32")
            warped = four_point_transform(img.copy(), pts)

            # Extract artwork from warped perspective
            art = extract_artwork(warped)
            return art

    return None

img_paths = sorted(glob.glob("yolo_detected/*.jpg"))
arts = []
time_sum = 0
for _ in range(1000):
    time_start = time.time()
    arts = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        art = extract_art_from_image(img)
        if art is not None:
            arts.append(art)
        else:
            print("No artwork found in the image.")
    
    arts_batch = []
    for i in range(0, len(arts), 8):
        arts_batch.append(np.array(arts[i:i+8]))
    time_end = time.time()
    time_sum += (time_end - time_start)

print("Average time cost: {} seconds".format(time_sum / 1000))
print(len(arts_batch))
print(arts_batch[0].shape)
