import socket
import requests
import numpy as np
import cv2
import mediapipe as mp
from numpy import integer


def is_pinching(landmarks, mp_hands, frame_width: int, frame_height: int, threshold=0.2):
    """ Detects if the hand is pinching by checking the distance between thumb and index finger tips """
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_base = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Convert to pixel coordinates
    thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    wrist_x, wrist_y = int(wrist.x * frame_width), int(wrist.y * frame_height)
    index_base_x, index_base_y = int(index_base.x * frame_width), int(index_base.y * frame_height)

    # Calculate Euclidean distance
    pinch_distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
    hand_size = np.sqrt((wrist_x - index_base_x) ** 2 + (wrist_y - index_base_y) ** 2)
    normalized_distance = pinch_distance / hand_size

    print(f"Pinch distance is: {normalized_distance}")

    return normalized_distance < threshold  # Return True if fingers are close enough

def send_wled_udp(data, WLED_IP, WLED_PORT, mode='DRGB'):
    '''
    This method automatically adds the header to and transmits a data packet to the WLED strip

    Args:
            data: a bytearray of data for transmission to the WLED strip
            WLED_IP: IP address of the WLED strip
            WLED_PORT: Port number of the WLED strip for UDP connection
            mode: either DRGB or WARLS, data is assumed to be compliant to this
            '''
    if(mode == 'DRGB'):
        newData = bytearray([2, 2])
    elif(mode == 'WARLS'):
        newData = bytearray([1, 2])
    else:
        raise ValueError('mode must be either DRGB or WARLS, perhaps check capitalization?')

    newData.extend(data)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(newData, (WLED_IP, WLED_PORT))


def get_led_count(ip):
    url = f"http://{ip}/json/info"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("leds", {}).get("count", 0)  # Extract LED count
    else:
        print("Failed to retrieve LED info.")
        return None

def position_to_rgb(centerX, frame_width):
    # Normalize x-coordinate
    norm_x = np.clip(centerX / frame_width, 0, 1)

    # Map normalized x to hue (0-179 in OpenCV)
    hue = int(norm_x * 179)  # OpenCV hue max is 179, not 360

    # Create HSV image with full saturation and value
    hsv = np.uint8([[[hue, 255, 255]]])  # Shape: (1,1,3)

    # Convert HSV to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]

    # Return as RGB tuple
    return int(bgr[2]), int(bgr[1]), int(bgr[0])

def average_of_points(point1, point2, point3, point4):
    """
    Calculates the average of four points.

    Args:
        point1: A tuple or list representing the coordinates of the first point (x, y).
        point2: A tuple or list representing the coordinates of the second point (x, y).
        point3: A tuple or list representing the coordinates of the third point (x, y).
        point4: A tuple or list representing the coordinates of the fourth point (x, y).

    Returns:
        A tuple representing the integer average x and y coordinates of the four points.
    """
    x_coords = [point1[0], point2[0], point3[0], point4[0]]
    y_coords = [point1[1], point2[1], point3[1], point4[1]]

    average_x = sum(x_coords) / len(x_coords)
    average_y = sum(y_coords) / len(y_coords)

    return (int(average_x), int(average_y))
