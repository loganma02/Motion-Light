import cv2
import mediapipe as mp
import socket
import requests

# def is_pinching(landmarks, frame_width, frame_height, threshold=30):
#     """ Detects if the hand is pinching by checking the distance between thumb and index finger tips """
#     thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#
#     # Convert to pixel coordinates
#     thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
#     index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
#
#     # Calculate Euclidean distance
#     distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
#
#     return distance < threshold  # Return True if fingers are close enough

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
    print(newData)
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
