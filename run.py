import cv2
import numpy as np

from aruco_detect import ARUCO_DICT
from wifiControlnew import set_motor_state

esp8266_base_url = "http://192.168.0.35"

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])  # Convert to degrees

# Returns angle in degrees
def get_orien(destination, cur_pos):
    dx = destination[0] - cur_pos[0] 
    dy = destination[1] - cur_pos[1]
    dist = np.abs(dx)+np.abs(dy)
    return (np.arctan2(dy, dx)*180/np.pi, dist)

def pure_pursuit(destination, cur_pos, cur_ori):
    orien, dist = get_orien(destination, cur_pos)
    orien = orien - cur_ori
    
    if dist < 0.15:
        return False
    base_speed = 90
    if orien > 90: orien = 90
    if orien < -90: orien = -90
    speedA = base_speed - (100-base_speed)*orien/90
    speedB = base_speed + (100-base_speed)*orien/90
    # print(speedA,speedB)
    set_motor_state(esp8266_base_url, "motorA", speedA, "forward")
    set_motor_state(esp8266_base_url, "motorB", speedB, "forward")
    return True

def turn_to_dest(destination, cur_pos, cur_ori):
    orien, dist = get_orien(destination, cur_pos)
    orien = orien - cur_ori
    if orien > 30:
        set_motor_state(esp8266_base_url, "motorB", 20, "forward")

        return 0
    
    if orien < -30:
        set_motor_state(esp8266_base_url, "motorB", 20, "forward")

        return 0
    return 1
    

if __name__ == "__main__":
    
    desired_aruco_dictionary = "DICT_6X6_250"
    camera_matrix = np.array([[1.17347416e+03, 0.0, 3.16649127e+02], [0.0, 1.17728067e+03, 2.53014584e+02], [0, 0, 1]], dtype=float)
    dist_coeffs = np.array([[-1.23322016e-01,  1.84925848e+01,  1.34875673e-02, -3.41599338e-02,
  -1.91508627e+02]])  # Assuming no lens distortion for simplicity

    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(desired_aruco_dictionary))
        sys.exit(0)

    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(desired_aruco_dictionary))
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters()
    this_aruco_parameters.adaptiveThreshWinSizeMin = 3  # Adjust as needed
    this_aruco_parameters.adaptiveThreshWinSizeMax = 23
    this_aruco_parameters.adaptiveThreshConstant = 10
    this_aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Start the video stream
    cap = cv2.VideoCapture(0)
    turning = True
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (1280, 960))

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
                position = tvec[0][0][:-1]
                rmat, jacobian = cv2.Rodrigues(rvec)
                euler_angles = rotation_matrix_to_euler_angles(rmat)
                orientation = euler_angles[2]
                dest = [-0.28, -0.3]
                
                
                
                if turn_to_dest(dest, position, orientation) == 1:
                    pure_pursuit(dest, position, orientation)
                
                # Display the position and orientation
                text0 = f"ID: {marker_id}, Pos: {position}"
                text1 = f"Orient: {orientation}"
                cv2.putText(frame, text0, (10, 30+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, text1, (10, 60+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            set_motor_state(esp8266_base_url, "motorA", 0, "stop")
            set_motor_state(esp8266_base_url, "motorB", 0, "stop")
                
    
        
        # Display the resulting frame
        cv2.imshow("frame", frame)

        # If "q" is pressed on the keyboard, exit this loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            set_motor_state(esp8266_base_url, "motorA", 0, "stop")
            set_motor_state(esp8266_base_url, "motorB", 0, "stop")
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
