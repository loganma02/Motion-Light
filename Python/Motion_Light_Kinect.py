import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
from openCV_methods import *

from pykinect2024 import PyKinectRuntime
from pykinect2024 import PyKinect2024

FRAME_WIDTH = 1920 # Kinect v2 color feed resolution
FRAME_HEIGHT = 1080
NUM_PINCHES = 2

WLED_IP = '10.0.0.128' # CHANGE THIS TO YOUR LOCAL WLED IP
WLED_PORT = 21324
LED_COUNT = get_led_count(ip=WLED_IP)

# These are the variables that are most likely to be tweaked
FADE_WIDTH = 1 # Change this value to change how wide the fade is along the strips
GAMMA = 0.2  # Change this value to change how intense the fade is when a hand approaches
FADE_COLOR = [255,20,255] # This is the color of the fade
VIDEO_MODE = 'color' # Currently support the color and infrared feeds from kinect

if (VIDEO_MODE == 'color'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)
elif (VIDEO_MODE == 'infrared'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
    FRAME_WIDTH = 512 #Kinect v2 Infrared camera resolution
    FRAME_HEIGHT = 424
else:
    raise ValueError('Check video mode, should be color or infrared.')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

stage = 'pinchStage'
currPinch = 0
frameCount = 0
LEDCords = [[0.0, 0.0], [0.0, 0.0]]
awayCords = [[0.0, 0.0], [0.0, 0.0]]

quad_pts = np.array([LEDCords[0], LEDCords[1], awayCords[0], awayCords[1]], dtype=np.float32)# (x,y)
rect_pts = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
perspective_matrix = cv2.getPerspectiveTransform(quad_pts, rect_pts)

while True:
    if (VIDEO_MODE == 'infrared'):
        frame = kinect.get_last_infrared_frame()
        frame = frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        frame = (frame / 256).astype(np.uint8)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif (VIDEO_MODE == 'color'):
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    results = hands.process(rgb_frame)

    pinch = False
    cx = 0  # These values are the position of the middle finger in the frame
    cy = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Handle pinching
            pinch = is_pinching(hand_landmarks, mp_hands, FRAME_WIDTH, FRAME_HEIGHT, threshold=0.28)

            # Get landmark for the tip of the middle finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] #INDEX_FINGER_TIP

            cx, cy = int(index_finger_tip.x * FRAME_WIDTH), int(index_finger_tip.y * FRAME_HEIGHT)

            # Draw a circle over the index finger tip
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    index = int((currPinch / (NUM_PINCHES - 1)) * (LED_COUNT - 1))

    if (stage == 'pinchStage'):
        send_wled_udp(bytearray([index, 255, 255, 255]), WLED_IP, WLED_PORT, 'WARLS')
        cv2.putText(frame, f"Please pinch and hold directly above the lit LED: {index + 1} of {LED_COUNT}", (20, 400),
                    cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"The current middle finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)

        if (pinch):  # If pinching add to frameCount
            frameCount = frameCount + 1
        else:  # Otherwise reset it
            frameCount = 0

        if (frameCount > 30):  # Once its been 30 frames of pinching we can set the LEDCords value
            if (cx != 0 and cy != 0):
                LEDCords[currPinch] = [cx, cy]
            currPinch = currPinch + 1
            frameCount = 0
            send_wled_udp(bytearray([index, 0, 0, 0]), WLED_IP, WLED_PORT, 'WARLS')  # Blank LED that was just on

        if (currPinch == NUM_PINCHES):
            stage = 'awayPinch'
            currPinch = 0

    elif (stage == 'awayPinch'):
        send_wled_udp(bytearray([index, 255, 255, 255]), WLED_IP, WLED_PORT, 'WARLS')
        cv2.putText(frame,
                    f"Now pinch and hold about 12 inches away from the lit LED: {index + 1} of {LED_COUNT}",
                    (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"The current middle finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)

        if (pinch):  # If pinching add to frameCount
            frameCount = frameCount + 1
        else:  # Otherwise reset it
            frameCount = 0

        if (frameCount > 30):  # Once its been 30 frames of pinching we can set the LEDCords value
            if (cx != 0 and cy != 0):
                awayCords[currPinch] = [cx, cy]
            currPinch = currPinch + 1
            frameCount = 0
            send_wled_udp(bytearray([index, 0, 0, 0]), WLED_IP, WLED_PORT, 'WARLS')  # Blank LED that was just on

        if (currPinch == NUM_PINCHES):
            currPinch = 0
            quad_pts = np.array([LEDCords[0], LEDCords[1], awayCords[0], awayCords[1]], dtype=np.float32)  # (x,y)
            rect_pts = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
            perspective_matrix = cv2.getPerspectiveTransform(quad_pts, rect_pts)
            stage = 'lightStage'

    elif (stage == 'lightStage'):
        cv2.putText(frame, f"The current middle finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, LEDCords[0], LEDCords[1], FADE_COLOR, 3, cv2.LINE_AA) # Interaction bounding box
        cv2.line(frame, LEDCords[1], awayCords[1], FADE_COLOR, 3, cv2.LINE_AA)
        cv2.line(frame, awayCords[1], awayCords[0], FADE_COLOR, 3, cv2.LINE_AA)
        cv2.line(frame, awayCords[0], LEDCords[0], FADE_COLOR, 3, cv2.LINE_AA)


        point = np.array([[cx, cy]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), perspective_matrix)[0][0]

        u, v = transformed[0], transformed[1]  # (0 to 1 range)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        intensity = v ** (1 / GAMMA)  # Adjust brightness perception

        led_data = bytearray(LED_COUNT * 3)  # Initialize all LEDs to off

        for i in range(LED_COUNT):
            distance = abs((i / LED_COUNT) - u) * LED_COUNT
            brightness = intensity * np.exp(- (distance ** 2) / (2 * FADE_WIDTH ** 2))

            # Scale RGB color
            r = int(FADE_COLOR[0] * brightness)
            g = int(FADE_COLOR[1] * brightness)
            b = int(FADE_COLOR[2] * brightness)

            # Set the LED's RGB values
            led_data[i * 3] = r
            led_data[i * 3 + 1] = g
            led_data[i * 3 + 2] = b
        send_wled_udp(led_data, WLED_IP, WLED_PORT, 'DRGB')

    # Display the output
    cv2.imshow('Motion Light', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"These were the LEDCords: {LEDCords}")
        print(f"These were the awayCords: {awayCords}")
        break

cap.release()
cv2.destroyAllWindows()

