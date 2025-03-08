import cv2
import numpy as np
import json

# Global list to store calibration points (e.g., strike zone corners)
calibration_points = []
# Global variable to store the detected ball (x, y, radius)
detected_ball = None

def detect_ball(image, expected_radius, radius_tolerance=1):
    """
    Auto-detects the ball in an image using Hough Circle detection.

    Parameters:
        image (numpy.ndarray): The input image.
        expected_radius (int): Known ball radius in pixels (for the resized image).
        radius_tolerance (int): Tolerance for the ball's radius.

    Returns:
        tuple or None: Detected ball parameters (x, y, radius) or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=expected_radius - radius_tolerance,
        maxRadius=expected_radius + radius_tolerance
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]  # returns (x, y, radius)
    return None

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to record calibration points.
    
    When the left mouse button is clicked, the (x, y) coordinate is saved,
    and a small green circle is drawn on the image.
    """
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", param['image'])
        print(f"Calibration point added: {(x, y)}")

def main():
    global calibration_points, detected_ball

    # Set a resize factor. Use the same value in both calibration and detection files.
    resize_factor = 1

    # Update with your calibration image path and expected ball radius (in resized pixel units)
    image_path = "/mnt/d/aaravStuff/spring25-26/strikePhone/fullVideoTests/fullVideoTestOneCalibrate.jpg"
    expected_radius = 25  # Adjust this based on the ball's size in the resized image

    # Load the calibration image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image. Check the file path.")
        return

    # Resize the image for calibration and consistency with detection code
    display_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)

    # Auto-detect the ball in the resized image
    ball = detect_ball(display_image, expected_radius)
    if ball is not None:
        x, y, r = ball
        detected_ball = (x, y, r)
        print(f"Detected ball at: ({x}, {y}) with radius: {r}")
        cv2.circle(display_image, (x, y), r, (255, 0, 0), 2)  # Blue circle around the ball
        cv2.circle(display_image, (x, y), 2, (0, 0, 255), 3)  # Red dot at the center
    else:
        print("No ball detected in the calibration image.")

    # Set up the window and mouse callback for calibration points
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback, param={'image': display_image})

    print("Click on the image to select strike zone calibration points (e.g., top-left and bottom-right corners).")
    print("Press 'q' when done.")

    # Loop until the user presses 'q'
    while True:
        cv2.imshow("Calibration", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
