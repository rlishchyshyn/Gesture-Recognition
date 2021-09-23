import os
import signal
import sys
import time
from itertools import chain

import cv2
import imutils
import mediapipe as mp
import numpy as np
from imutils.video import VideoStream

from gesture_classifier import GestureClassifier
from utils.hand_bounding_box import get_bounding_box

# init necessary constants and global variables
HEIGHT, WIDTH = 480, 640
MAX_NUM_HANDS = 2
FEATURES_COUNT = 42
ASSETS_PATH = "assets"
ICONS = {}
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
gc = GestureClassifier("dataset/dataset.csv")

# read all assets icons
for imagePath in os.listdir(ASSETS_PATH):
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    icon = cv2.imread(f"{ASSETS_PATH}{os.path.sep}{imagePath}")

    ICONS[label] = icon


# function to handle keyboard interrupt
def signal_handler(sig, frame):
    print("[INFO] You pressed `ctrl + c`! Closing face recognition"
          " door monitor application...")
    sys.exit(0)


# signal trap to handle keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)
print("[INFO] Press `ctrl + c` to exit, or 'q' to quit if you have"
      " the display option on...")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

with mp_hands.Hands(
        min_detection_confidence=.7,
        max_num_hands=MAX_NUM_HANDS,
        min_tracking_confidence=.6) as hands:
    while True:
        # grab the frame from the threaded video stream and flip it
        frame = vs.read()
        image = frame.copy()
        image = imutils.resize(image, width=WIDTH, height=HEIGHT)
        image = cv2.flip(image, 1)

        # print text in the image
        image = cv2.putText(image, 'Current gestures:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # get the hand landmarks and handedness
        results = hands.process(image)

        if results.multi_hand_landmarks:
            features_data = []

            for hand_landmarks in results.multi_hand_landmarks:
                # draw the hand annotations on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # bounding box over the hands
                landmark_list = [[int(landmark.x * WIDTH), int(landmark.y * HEIGHT)] for _, landmark in
                                 enumerate(hand_landmarks.landmark)]

                x, y, w, h = get_bounding_box(landmark_list)
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), 255, 2)

                # normalize landmark coordinates relative to the bounding box
                landmarks = map(lambda landmark: [(landmark[0] - x) / w, (landmark[1] - y) / h], landmark_list)

                # cast it to necessary format
                chained_landmarks = list(chain.from_iterable(landmarks))
                features_data.append(np.array(chained_landmarks))

                # classify gesture
                gestures = gc.predict_gesture(features_data)

                # if exists display an icon
                if len(gestures):
                    for idx, gesture in enumerate(gestures):
                        # get icon gesture
                        icon = ICONS[gesture]

                        # create offset for gesture visualization
                        x_offset = idx * icon.shape[1] + 10
                        y_offset = 50

                        # create a mask and apply it to the area
                        _, mask = cv2.threshold(icon, 1, 255, cv2.THRESH_BINARY_INV)

                        res = cv2.bitwise_and(
                            image[y_offset:y_offset + icon.shape[0], x_offset:x_offset + icon.shape[1]], mask)
                        final = cv2.bitwise_or(icon, res)

                        image[y_offset:y_offset + icon.shape[0], x_offset:x_offset + icon.shape[1]] = final

        # show the output frame
        cv2.imshow("frame", image)

        # if the `q` key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

# do a bit of cleanup
vs.stop()
