import os

from Python.WLED_Discovery import discover_wled_devices

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
from openCV_methods import *
import pandas as pd

from pykinect2024 import PyKinectRuntime
from pykinect2024 import PyKinect2024

FRAME_WIDTH = 1920 # Kinect v2 color feed resolution
FRAME_HEIGHT = 1080

#WLED_IP = '10.0.0.128' # CHANGE THIS TO YOUR LOCAL WLED IP
#WLED_PORT = 21324
#LED_COUNT = get_led_count(ip=WLED_IP)
DEVICES = pd.DataFrame(columns=['name', 'ip', 'numLED'])
print(DEVICES)


# These are the variables that are most likely to be tweaked
FADE_WIDTH = 1 # Change this value to change how wide the fade is along the strips
GAMMA = 0.2  # Change this value to change how intense the fade is when a hand approaches
FADE_COLOR = [255,20,255] # This is the color of the fade
VIDEO_MODE = 'infrared' # Currently support the color and infrared feeds from kinect

#Video mode setup
if (VIDEO_MODE == 'color'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)
elif (VIDEO_MODE == 'infrared'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
    FRAME_WIDTH = 512 #Kinect v2 Infrared camera resolution
    FRAME_HEIGHT = 424
else:
    raise ValueError('Check video mode, should be color or infrared.')

# Initialize MediaPipe Hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# quad_pts = np.array([LEDCords[0], LEDCords[1], awayCords[0], awayCords[1]], dtype=np.float32)# (x,y)
# rect_pts = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
# perspective_matrix = cv2.getPerspectiveTransform(quad_pts, rect_pts)

while True:
    if (VIDEO_MODE == 'infrared'):
        frame = kinect.get_last_infrared_frame()
        frame = frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        frame = (frame / 256).astype(np.uint8)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif (VIDEO_MODE == 'color'):
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


    # Display the output
    cv2.imshow('Motion Light', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"These were the LEDCords: {LEDCords}")
        print(f"These were the awayCords: {awayCords}")
        break

    # Listen for refresh when r is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        DEVICES = discover_wled_devices()

    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(DEVICES.head())

cap.release()
cv2.destroyAllWindows()

