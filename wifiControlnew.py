import requests
import time

def set_motor_state(base_url, motor:str, speed:int, direction:str):
    """
    Sends a request to control the motor state.

    :param base_url: The base URL of the ESP8266 web server (e.g., "http://192.168.1.100")
    :param motor: The motor to control ("motorA" or "motorB")
    :param speed: Speed of the motor (0-255)
    :param direction: Direction of the motor ("forward", "reverse", or "stop")
    """
    calibration_factor = -0.05
    min_speed = 100
    max_speed = 130
    
    if speed > 100 or speed < 0:
        return
    speed = min_speed + (max_speed-min_speed)/100*speed 
    cal_spd = calibration_factor*speed
    
    speed = speed+cal_spd if motor == "motorA" else speed-cal_spd
    
    
    url = f"{base_url}/{motor}"
    init_params = {
        "speed": 255,
        "direction": direction
    }
    params = {
        "speed": speed,
        "direction": direction
    }
    
    try:
        response = requests.get(url, params=init_params)
        if response.status_code == 200:
            # print(f"Successfully updated {motor}: {response.text}")
            pass
        else:
            print(f"Failed to update {motor}: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Error connecting to {base_url}: {e}")
    time.sleep(0.1)
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # print(f"Successfully updated {motor}: {response.text}")
            pass
        else:
            print(f"Failed to update {motor}: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Error connecting to {base_url}: {e}")

if __name__ == "__main__":
    # Set the ESP8266 IP address here
    esp8266_base_url = "http://192.168.0.35"  # Replace with the actual IP address
    
    ### Simplist form of communication
    motor = "motorA"
    url = f"{esp8266_base_url}/{motor}"
    params = {'speed': 255, 'direction': 'stop'}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to update {motor}: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Error connecting to {esp8266_base_url}: {e}")
    ###
        
    # Example usage
    while True:
        print("\nEnter motor control command in the format: 'A 255 1', 'B 100 0', or 'A 0 2'")
        print("- First character (A or B): Motor selection")
        print("- Second value (0-255): Speed")
        print("- Third value (1, 0, or 2): Direction (1 = forward, 0 = reverse, 2 = stop)")
        print("Type 'q' to quit.")

        command = input("Enter command: ").strip().upper()
        if command == "Q":
            print("Exiting...")
            set_motor_state(esp8266_base_url, "motorA", speed, "stop")
            set_motor_state(esp8266_base_url, "motorB", speed, "stop")
            break

        try:
            parts = command.split()
            if len(parts) != 3:
                print("Invalid format. Please follow the format 'A 255 1'.")
                continue

            motor = "motorA" if parts[0] == "A" else "motorB" if parts[0] == "B" else None
            if motor is None:
                print("Invalid motor selection. Use 'A' or 'B'.")
                continue

            speed = int(parts[1])
            if not (0 <= speed <= 255):
                print("Speed must be between 0 and 255.")
                continue

            direction = "forward" if parts[2] == "1" else "reverse" if parts[2] == "0" else "stop" if parts[2] == "2" else None
            if direction is None:
                print("Invalid direction. Use '1' for forward, '0' for reverse, or '2' for stop.")
                continue

            set_motor_state(esp8266_base_url, motor, speed, direction)
        except ValueError:
            print("Invalid input. Ensure the speed is a number and direction is 0, 1, or 2.")
