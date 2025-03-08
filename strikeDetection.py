import cv2
import numpy as np

def detect_ball(image, expected_radius, radius_tolerance=7):
    """
    Auto-detects the ball in an image using Hough Circle detection.
    
    Parameters:
        image (numpy.ndarray): The input frame.
        expected_radius (int): Known ball radius in pixels.
        radius_tolerance (int): Allowed variation in ball radius.
        
    Returns:
        tuple or None: Detected ball parameters (x, y, radius) or None if not found.
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    # Use Hough Circle Transform to detect circles
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
        # Return the first detected circle (x, y, radius)
        return circles[0][0]
    return None

def is_inside_strike_zone(ball_center, strike_zone):
    """
    Checks if the ball's center is inside the defined strike zone.
    
    Parameters:
        ball_center (tuple): (x, y) coordinates of the ball's center.
        strike_zone (tuple): A tuple containing the top-left and bottom-right coordinates
                             of the strike zone: ((x1, y1), (x2, y2)).
                             
    Returns:
        bool: True if inside the strike zone, False otherwise.
    """
    (x1, y1), (x2, y2) = strike_zone
    x, y = ball_center
    return x1 <= x <= x2 and y1 <= y <= y2

def main():
    # ----------------------------------------------------------------------
    # Calibration Data (update these with your actual calibration points)
    # These coordinates are from the full resolution calibration.
    strike_zone_top_left = (350, 200)      # Replace with your actual coordinate
    strike_zone_bottom_right = (600, 450)   # Replace with your actual coordinate
    strike_zone = (strike_zone_top_left, strike_zone_bottom_right)
    
    # ----------------------------------------------------------------------
    # Video Source
    video_path = "/mnt/d/aaravStuff/spring25-26/strikePhone/fullVideoTests/fullVideoTestOne.MP4"
    #cap = cv2.VideoCapture(video_path)
    
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # ----------------------------------------------------------------------
    # Known ball parameters (from calibration)
    expected_radius = 20  # Update this value based on your calibration
    radius_tolerance = 7  # Adjust tolerance if needed

    resize_factor = .5  # Resizing factor to speed processing
    
    frame_count = 0  # Frame counter for skipping frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # Resize the frame for faster processing
        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

        # Scale the strike zone coordinates to match the resized frame
        scaled_strike_zone = (
            (int(strike_zone_top_left[0] * resize_factor), int(strike_zone_top_left[1] * resize_factor)),
            (int(strike_zone_bottom_right[0] * resize_factor), int(strike_zone_bottom_right[1] * resize_factor))
        )
        
        # Process detection only on every second frame
        if frame_count % 2 == 0:
            # Detect the ball in the current frame
            ball = detect_ball(frame, expected_radius, radius_tolerance)
            
            # Draw the strike zone rectangle on the frame
            cv2.rectangle(frame, scaled_strike_zone[0], scaled_strike_zone[1], (0, 255, 0), 2)
            
            if ball is not None:
                x, y, r = ball
                # Draw the detected ball (circle and center dot)
                cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                # Check if the ball is inside the strike zone
                if is_inside_strike_zone((x, y), scaled_strike_zone):
                    cv2.putText(frame, "Strike", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Ball", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No ball detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        #else:
            # On skipped frames, you can choose to just draw the strike zone
            #cv2.rectangle(frame, scaled_strike_zone[0], scaled_strike_zone[1], (0, 255, 0), 2)
            #cv2.putText(frame, "Frame Skipped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the processed frame
        cv2.imshow("Strike Zone Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
