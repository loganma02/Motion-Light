import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
from openCV_methods import *

#TODO: Figure out how I can move this to another file
def is_pinching(landmarks, frame_width, frame_height, threshold=40):
    """ Detects if the hand is pinching by checking the distance between thumb and index finger tips """
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Convert to pixel coordinates
    thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

    # Calculate Euclidean distance
    distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

    print(f"Pinch distance is: {distance}")
    return distance < threshold  # Return True if fingers are close enough

FRAME_WIDTH = 680
FRAME_HEIGHT = 480
NUM_PINCHES = 2

WLED_IP = '10.0.0.34'
WLED_PORT = 8080
#LED_COUNT = get_led_count(ip=WLED_IP)
LED_COUNT = 20

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

stage = 'pinchStage'
currPinch = 0
frameCount = 0
LEDCords = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
awayCords = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    pinch = False
    cx = 0 #These values are the position of the middle finger in the frame
    cy = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Handle pinching
            pinch = is_pinching(hand_landmarks, FRAME_WIDTH, FRAME_HEIGHT)

            # Get landmark for the tip of the middle finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw a circle over the middle finger tip
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            # Optionally draw all landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # TODO: Make this work to combine the pinch methods, those are basically the same anyway

    # def pinchStages(lineText, frameCount, LEDCords, currPinch):
    #     cv2.putText(frame,
    #                 f"{lineText}: {currPinch + 1} of {NUM_PINCHES}",
    #                 (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    #     cv2.putText(frame, f"The current middle finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX,
    #                 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    #
    #     if (pinch): #If pinching add to frameCount
    #         frameCount = frameCount + 1
    #     else: #Otherwise reset it
    #         frameCount = 0
    #
    #     if (frameCount > 30): #Once its been 30 frames of pinching we can set the LEDCords value
    #         if (cx != 0 and cy != 0):
    #             LEDCords[currPinch] = [cx, cy]
    #         currPinch = currPinch + 1
    #         frameCount = 0
    #
    #     if (currPinch == NUM_PINCHES):
    #         stage = 'lightStage'
    #         currPinch = 0

    index = (currPinch / NUM_PINCHES-1) * (LED_COUNT-1)

    if (stage == 'pinchStage'):
        #send_wled_udp(bytearray([255,255,255]), '10.0.0.34', 8050, 'DRGB')
        index = (currPinch // NUM_PINCHES - 1) ** (LED_COUNT - 1)
        print(index)
        # WLED light up num of leds/ currPinch
        cv2.putText(frame, f"Please pinch and hold directly above the lit LED: {currPinch+1} of {NUM_PINCHES}", (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"The current index finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        # The pinch and hold code
        if (pinch): #If pinching add to frameCount
            frameCount = frameCount + 1
        else: #Otherwise reset it
            frameCount = 0

        if (frameCount > 30): #Once its been 30 frames of pinching we can set the LEDCords value
            if (cx != 0 and cy != 0):
                LEDCords[currPinch] = [cx, cy]
            currPinch = currPinch + 1
            frameCount = 0


        if (currPinch == NUM_PINCHES):
            stage = 'awayPinch'
            currPinch = 0

    elif (stage == 'awayPinch'):
        #WLED light up
        cv2.putText(frame, f"Now pinch and hold about 12 inches away from the lit LED: {currPinch + 1} of {NUM_PINCHES}",
                    (20, 400), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"The current index finger coordinate is {cx}, {cy}", (20, 460), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)

        if (pinch): #If pinching add to frameCount
            frameCount = frameCount + 1
        else: #Otherwise reset it
            frameCount = 0

        if (frameCount > 30): #Once its been 30 frames of pinching we can set the LEDCords value
            if (cx != 0 and cy != 0):
                awayCords[currPinch] = [cx, cy]
            currPinch = currPinch + 1
            frameCount = 0

        if (currPinch == NUM_PINCHES):
            stage = 'lightStage'
            currPinch = 0

    #elif (stage == 'lightStage'):


    # Display the output
    cv2.imshow('Hand Tracking', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"These are the LEDCords: {LEDCords}")
        print(f"These are the awayCords: {awayCords}")
        break

cap.release()
cv2.destroyAllWindows()

