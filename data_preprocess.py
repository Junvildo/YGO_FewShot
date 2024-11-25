import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the image by converting to grayscale, equalizing histogram,
    and applying a median blur.
    Args:
    - image: Input image as a numpy array (BGR format).

    Returns:
    - thresh: Thresholded binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_image = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(eq_image, 3)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    return thresh


def find_contours(thresh):
    """
    Find contours in the thresholded image.
    Args:
    - thresh: Binary image after thresholding.

    Returns:
    - contours: List of contours found in the image.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def order_points(pts):
    """
    Order points in the following order: top-left, top-right, bottom-right, bottom-left.
    Args:
    - pts: Array of points to be ordered.

    Returns:
    - rect: Ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Perform a perspective transform on the image using the given points.
    Args:
    - image: Input image as a numpy array.
    - pts: Array of four points for perspective transformation.

    Returns:
    - warped: Warped image after perspective correction.
    """
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_and_correct_perspective(image, min_area_ratio=0.1):
    """
    Detect card-like contours in the image and correct their perspective.
    Args:
    - image: Input image as a numpy array.
    - min_area_ratio: Minimum contour area ratio compared to the image size to consider as a card.
    
    Returns:
    - The warped image with corrected perspective, or None if no valid contour is found.
    """
    thresh = preprocess_image(image)
    contours = find_contours(thresh)
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * min_area_ratio

    max_area = 0
    largest_contour = None

    # Find the largest valid contour
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 4 and cv2.contourArea(cnt) > min_area:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt

    if largest_contour is not None:
        # Find the four corners of the largest contour
        p1 = (0, 0)
        p2 = (image.shape[1], 0)
        p3 = (0, image.shape[0])
        p4 = (image.shape[1], image.shape[0])
        dist1 = dist2 = dist3 = dist4 = float('inf')
        p1_contour = p2_contour = p3_contour = p4_contour = None

        for p in largest_contour.reshape(-1, 2):
            d1, d2, d3, d4 = (np.linalg.norm(p - p1), np.linalg.norm(p - p2),
                              np.linalg.norm(p - p3), np.linalg.norm(p - p4))
            if d1 < dist1: dist1, p1_contour = d1, p
            if d2 < dist2: dist2, p2_contour = d2, p
            if d3 < dist3: dist3, p3_contour = d3, p
            if d4 < dist4: dist4, p4_contour = d4, p

        # Create the points array for perspective transform
        pts = np.array([p1_contour, p2_contour, p3_contour, p4_contour], dtype="float32")
        return four_point_transform(image, pts)

    # Return None if no valid contour is found
    return None



def extract_artwork(image, thresh_val=180, img_size=56):
    """
    Extract the artwork area from the given image by processing the largest contour.
    Args:
    - image: Input image as a numpy array (BGR format).
    - thresh_val: Threshold value for binary segmentation.
    - img_size: Size to which the extracted artwork is resized (img_size x img_size).
    
    Returns:
    - art: Resized artwork area extracted from the image.
    """
    # Convert to grayscale and equalize the histogram
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_image = cv2.equalizeHist(gray_image)

    # Threshold the equalized image
    _, binary_image = cv2.threshold(eq_image, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=3)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Define the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)
    if image.shape[0] > image.shape[1]:  # Portrait orientation
        if (image.shape[0] - (y + h)) < y:
            art = image[int((h * 4) * 0.2):y, x:x + w]
        else:
            art = image[y + h:int(((y + h) * 4) * 0.8), x:x + w]
    else:  # Landscape orientation
        if x < (image.shape[1] // 2):
            art = image[y:y + h, x + w:int((w * 4) * 0.8)]
        else:
            art = image[y:y + h, int((w * 4) * 0.2):x]

    # Resize the extracted artwork
    art = cv2.resize(art, (img_size, img_size))

    return art
