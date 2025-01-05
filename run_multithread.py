import cv2
import numpy as np
import threading
import time
from aruco_detect import ARUCO_DICT
from wifiControlnew import set_motor_state

# Shared state for position and orientation
shared_state = {
    "position": None,
    "orientation": None,
    "dests": [],
    "stop": False,
    "lpoint":[],
    "ori":None
}
state_lock = threading.Lock()

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

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point to the points list
        with state_lock:
            shared_state["dests"].append((x, y))

def get_orien(destination, cur_pos):
    dx = destination[0] - cur_pos[0]
    dy = destination[1] - cur_pos[1]
    dist = np.abs(dx) + np.abs(dy)
    return (np.arctan2(dy, dx) * 180 / np.pi, dist)

def projection_on_line(P1, P2, P3):
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    
    v = P2 - P1

    w = P3 - P1

    proj_w_on_v = np.clip(np.dot(w, v) / np.dot(v, v), 0, 1) * v
    P_proj = P1 + proj_w_on_v
    return P_proj

def get_lookahead_point(prev_p, next_p, cur_point, cur_orien):
    if np.array_equal(prev_p, next_p):
        return prev_p
    ldist = 50#lookahead distance
    
    # v1 = np.array(np.cos(np.deg2rad(cur_orien)), np.sin(np.deg2rad(cur_orien)))
    # v1 = v1 / np.linalg.norm(v1)
    
    lahdp = cur_point+cur_orien*ldist
    
    # print(prev_p, next_p, lahdp)
    return projection_on_line(prev_p, next_p, lahdp)
    

def video_analysis_thread(camera_matrix, dist_coeffs, this_aruco_dictionary, this_aruco_parameters):
    cap = cv2.VideoCapture(0)
    writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640,480))

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", mouse_callback)
    
    while not shared_state["stop"]:
        ret, frame = cap.read()
        
        if not ret:
            continue
        with state_lock:
            points = shared_state["dests"]
        


        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, this_aruco_dictionary, parameters=this_aruco_parameters
        )
        if len(corners) > 0:
            ids = ids.flatten()
            for marker_corners, marker_id in zip(corners, ids):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [marker_corners], 0.0526, camera_matrix, dist_coeffs)
                
                position = np.mean(marker_corners[0], axis=0)
                
                rmat, _ = cv2.Rodrigues(rvec)
                euler_angles = rotation_matrix_to_euler_angles(rmat)
                orientation = euler_angles[2]
                
                
                ori = [marker_corners[0][1][0]-marker_corners[0][0][0], marker_corners[0][1][1]-marker_corners[0][0][1]]
                ori = ori = ori / np.linalg.norm(ori)
                
                
                with state_lock:
                    shared_state["position"] = position
                    shared_state["orientation"] = orientation
                    shared_state['ori'] = ori

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.0526)
                text0 = f"ID: {marker_id}, Pos: {position}"
                text1 = f"Orient: {orientation}"
                cv2.putText(frame, text0, (10, 30+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, text1, (10, 60+60*marker_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Draw lines between the clicked points
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (255, 0, 0), 2)  # Blue line
        # Draw the clicked points
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Red dots            
        else:
            set_motor_state(esp8266_base_url, "motorA", 0, "stop")
            set_motor_state(esp8266_base_url, "motorB", 0, "stop")
            
        with state_lock:
            lpoint = shared_state["lpoint"]
            
        if len(lpoint) > 0:
            # print(type(lpoint), type(position))
            cv2.circle(frame, tuple(map(int, lpoint)), 5, (200, 200, 0), -1)
            cv2.line(frame, tuple(map(int, position)), tuple(map(int, lpoint)), (200, 200, 0), 2)

        writer.write(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            with state_lock:
                shared_state["stop"] = True
            set_motor_state(esp8266_base_url, "motorA", 0, "stop")
            set_motor_state(esp8266_base_url, "motorB", 0, "stop")
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def robot_control_thread():
    i = 0
    previ = 0
    while not shared_state["stop"]:
        with state_lock:
            position = shared_state["position"]
            orientation = shared_state["orientation"]
            dests = shared_state["dests"]
            ori = shared_state['ori']

        if position is not None and orientation is not None and len(dests) != 0:
            lpoint = get_lookahead_point(np.array(dests[previ]), np.array(dests[i]), np.array(position), ori)
            # print(previ, i )
            # print(dests)
            
            
            with state_lock:
                shared_state["lpoint"] = lpoint

            
            _, dist = get_orien(dests[i], position)
            # orien, _ = get_orien(lpoint, position)
            # print(orien)
            # orien -= orientation
            
            vt = lpoint-position
            vt = vt/np.linalg.norm(vt)

            det = np.linalg.det(np.array([ori,vt]))
           

            if dist < 40:
                set_motor_state(esp8266_base_url, "motorA", 0, "stop")
                set_motor_state(esp8266_base_url, "motorB", 0, "stop")
                print("Reached destination! Bitch")
                # time.sleep(0.5)
                previ = i
                i = (i+1)%len(dests)
                print(f"going to the next dest:{i} at {dests[i]}")

            # orien = max(min(orien, 90), -90)
            # speedA = base_speed - (100 - base_speed) * orien / 90
            # speedB = base_speed + (100 - base_speed) * orien / 90

            if det > 0.25:
                set_motor_state(esp8266_base_url, "motorA", 10, "stop")
                set_motor_state(esp8266_base_url, "motorB", 5, "forward")
                continue
            if det < -0.25:
                set_motor_state(esp8266_base_url, "motorA", 5, "forward")
                set_motor_state(esp8266_base_url, "motorB", 10, "stop")
                continue
            set_motor_state(esp8266_base_url, "motorA", 5, "forward")
            set_motor_state(esp8266_base_url, "motorB", 5, "forward")
                        

        time.sleep(0.05)  # Adjust loop frequency as needed

if __name__ == "__main__":
    desired_aruco_dictionary = "DICT_6X6_250"
    camera_matrix = np.array(
        [[1.17347416e+03, 0.0, 3.16649127e+02], [0.0, 1.17728067e+03, 2.53014584e+02], [0, 0, 1]], dtype=float
    )
    dist_coeffs = np.array([[-1.23322016e-01, 1.84925848e+01, 1.34875673e-02, -3.41599338e-02, -1.91508627e+02]])
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters()

    video_thread = threading.Thread(target=video_analysis_thread, args=(camera_matrix, dist_coeffs, this_aruco_dictionary, this_aruco_parameters))
    control_thread = threading.Thread(target=robot_control_thread)

    video_thread.start()
    control_thread.start()

    video_thread.join()
    control_thread.join()
    
