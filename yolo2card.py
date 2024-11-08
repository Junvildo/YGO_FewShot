import cv2
import numpy as np

def auto_detect_card_corners_and_transform(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None

    # Resize for easier processing if needed
    scale_percent = 40  # adjust scale as needed
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(blurred, 200, 1000)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and pick the largest one (assumed to be the card)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_contour = contours[0]

    # Approximate the contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)

    if len(approx) != 4:
        print("Error: Could not detect 4 corners of the card.")
        return None

    # Obtain the four points
    card_corners = np.array([point[0] for point in approx], dtype="float32")

    # Order the corners in a consistent way: top-left, top-right, bottom-right, bottom-left
    s = card_corners.sum(axis=1)
    diff = np.diff(card_corners, axis=1)
    top_left = card_corners[np.argmin(s)]
    bottom_right = card_corners[np.argmax(s)]
    top_right = card_corners[np.argmin(diff)]
    bottom_left = card_corners[np.argmax(diff)]
    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    # Define the dimensions for the bird's eye view
    width, height = 300, 200
    dst_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))

    # Display the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Bird's Eye View", bird_eye_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bird_eye_view

# Usage
image_path = 'yolo_detected/detected_object_5.jpg'
bird_eye_view = auto_detect_card_corners_and_transform(image_path)
