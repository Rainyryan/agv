import cv2
import numpy as np

desired_aruco_dictionary = "DICT_6X6_250"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Define camera parameters (example values)
camera_matrix = np.array([[558.36401905, 0.0, 316.16819865], [0.0, 527.55928876, 235.78707571], [0, 0, 1]], dtype=float)
# camera_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float)
dist_coeffs = np.array([[-9.95988725e-02,  2.09943517e+00,  1.16936750e-02,  7.48710455e-03, -7.75414113e+00]])  # Assuming no lens distortion for simplicity

if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(desired_aruco_dictionary))
    sys.exit(0)

# Load the ArUco dictionary
print("[INFO] detecting '{}' markers...".format(desired_aruco_dictionary))
this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
this_aruco_parameters = cv2.aruco.DetectorParameters()

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect ArUco markers in the video frame
    corners, ids, rejected = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters
    )

    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()

        # Loop over the detected ArUco corners
        for marker_corners, marker_id in zip(corners, ids):
            # Estimate pose of the ArUco marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], 0.0526, camera_matrix, dist_coeffs
            )

            # Draw the bounding box and axis
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.0526)

            # Extract position and orientation
            position = tvec[0]
            orientation = rvec[0]

            # Display the position and orientation
            text0 = f"ID: {marker_id}, Pos: {position}"
            text1 = f"Orient: {orientation}"
            cv2.putText(frame, text0, (10, 30+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, text1, (10, 60+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
   
    # Display the resulting frame
    cv2.imshow("frame", frame)

    # If "q" is pressed on the keyboard, exit this loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close down the video stream
cap.release()
cv2.destroyAllWindows()
