import cv2
import numpy as np

def detect_boxes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use edge detection (Canny) with relaxed thresholds
    edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds for easier edge detection
    
    # Apply dilation to strengthen weak edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image for drawing contours
    contour_image = np.zeros_like(frame)
    
    box_count = 0

    for contour in contours:
        # Approximate the contour to simplify its shape
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)  # Relaxed approximation tolerance
        
        # Check if the contour is a rectangle (has 4 sides)
        if len(approx) == 4:
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)
            
            # Lower the size threshold for detected boxes
            if w > 20 and h > 20:  # Detect smaller boxes
                box_count += 1
                # Draw the rectangle around the detected box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Label the rectangle with its position
                cv2.putText(frame, f"Box {box_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw the contour outline on the contour image
        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), 1)
    
    return frame, contour_image, box_count

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame for consistent processing
    frame = cv2.resize(frame, (640, 480))
    
    # Detect boxes and contours in the current frame
    processed_frame, contour_image, box_count = detect_boxes(frame)

    # Display the number of boxes detected
    cv2.putText(processed_frame, f"Boxes Detected: {box_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Real-time display of the processed frame and contour outlines
    cv2.imshow("Box Detection", processed_frame)
    cv2.imshow("Contour Outlines", contour_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
