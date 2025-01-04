import cv2
import numpy as np

# Set the size of the chessboard
CHESSBOARD_SIZE = (8, 5)  # Number of internal corners in rows and columns
SQUARE_SIZE = 21  # Size of the square (in mm or cm) of the chessboard pattern

# Prepare object points (real-world coordinates)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale the object points to the actual size of the squares

# Arrays to store object points and image points from the single image
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Start the camera capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera!")
    exit()

# Capture a single frame
ret, img = cap.read()

if ret:
    # Save the captured frame as an image
    image_path = "captured_chessboard.png"
    cv2.imwrite(image_path, img)
    print(f"Captured image saved as {image_path}")

    # Convert the captured image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow("Chessboard", img)
        cv2.waitKey(500)
    else:
        print("Chessboard corners not found!")

else:
    print("Error: Failed to capture an image.")

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()

# Perform camera calibration with the detected corners
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration results
    np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    # Print the camera matrix and distortion coefficients
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Calibration failed. No valid points detected.")
