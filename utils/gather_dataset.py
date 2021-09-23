import os
import signal
import sys
import time
from itertools import chain

import cv2
import imutils
import mediapipe as mp
from imutils.video import VideoStream

from utils.hand_bounding_box import get_bounding_box

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HEIGHT, WIDTH = 480, 640
DATASET_PATH = "dataset/dataset.csv"


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

# open file if it is empty write header
file = open(DATASET_PATH, "a")
if os.stat(DATASET_PATH).st_size == 0:
    file.write(','.join(
        ["Gesture",
         "Wrist_X", "Wrist_Y",
         "Thumb_cmc_x", "Thumb_cmc_y",
         "Thumb_mcp_x", "Thumb_mcp_y",
         "Thumb_ip_x", "Thumb_ip_y",
         "Thumb_tip_x", "Thumb_tip_y",
         "Index_finger_mcp_x", "Index_finger_mcp_y",
         "Index_finger_pip_x", "Index_finger_pip_y",
         "Index_finger_dip_x", "Index_finger_dip_y",
         "Index_finger_tip_x", "Index_finger_tip_y",
         "Middle_finger_mcp_x", "Middle_finger_mcp_y",
         "Middle_finger_pip_x", "Middle_finger_pip_y",
         "Middle_finger_dip_x", "Middle_finger_dip_y",
         "Middle_finger_tip_x", "Middle_finger_tip_y",
         "Ring_finger_mcp_x", "Ring_finger_mcp_y",
         "Ring_finger_pip_x", "Ring_finger_pip_y",
         "Ring_finger_dip_x", "Ring_finger_dip_y",
         "Ring_finger_tip_x", "Ring_finger_tip_y",
         "Pinky_mcp_x", "Pinky_mcp_y",
         "Pinky_pip_x", "Pinky_pip_y",
         "Pinky_dip_x", "Pinky_dip_y",
         "Pinky_tip_x", "Pinky_tip_y\n"]))

# IMPORTANT! Use only 1 number of hands
with mp_hands.Hands(
        min_detection_confidence=.5,
        max_num_hands=1,
        min_tracking_confidence=.5) as hands:
    while True:
        # grab the frame from the threaded video stream and flip it
        image = vs.read()
        image = imutils.resize(image, width=WIDTH, height=HEIGHT)
        image = cv2.flip(image, 1)

        # get the hand landmarks and handedness
        results = hands.process(image)

        # wait any key press
        key = cv2.waitKey(1) & 0xFF

        # normalize landmark coordinates relative to the bounding box and write it to csv file
        if results.multi_hand_landmarks:
            chained_landmarks = []

            for landmarks in results.multi_hand_landmarks:
                # bounding box over the hands
                landmark_list = [[int(landmark.x * WIDTH), int(landmark.y * HEIGHT)] for _, landmark in
                                 enumerate(landmarks.landmark)]
                x, y, w, h = get_bounding_box(landmark_list)
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), 255, 2)

                # normalize landmark coordinates relative to the bounding box
                landmarks = map(lambda landmark: [(landmark[0] - x) / w, (landmark[1] - y) / h], landmark_list)

                # cast it to necessary format
                chained_landmarks = list(chain.from_iterable(landmarks))

            # backhand-index-pointing-down
            if key == ord("z"):
                res = ",".join(str(x) for x in ["backhand-index-pointing-down"] + chained_landmarks)
                file.write(res + "\n")

            # backhand-index-pointing-left
            if key == ord("x"):
                res = ",".join(str(x) for x in ["backhand-index-pointing-left"] + chained_landmarks)
                file.write(res + "\n")

            # backhand-index-pointing-right
            if key == ord("c"):
                res = ",".join(str(x) for x in ["backhand-index-pointing-right"] + chained_landmarks)
                file.write(res + "\n")

            # call-me-hand
            if key == ord("v"):
                res = ",".join(str(x) for x in ["call-me-hand"] + chained_landmarks)
                file.write(res + "\n")

            # crossed-fingers
            if key == ord("b"):
                res = ",".join(str(x) for x in ["crossed-fingers"] + chained_landmarks)
                file.write(res + "\n")

            # hand-with-fingers-splayed
            if key == ord("n"):
                res = ",".join(str(x) for x in ["hand-with-fingers-splayed"] + chained_landmarks)
                file.write(res + "\n")

            # index-pointing-up
            if key == ord("m"):
                res = ",".join(str(x) for x in ["index-pointing-up"] + chained_landmarks)
                file.write(res + "\n")

            # love-you-gesture
            if key == ord("a"):
                res = ",".join(str(x) for x in ["love-you-gesture"] + chained_landmarks)
                file.write(res + "\n")

            # middle-finger
            if key == ord("s"):
                res = ",".join(str(x) for x in ["middle-finger"] + chained_landmarks)
                file.write(res + "\n")

            # ok-hand
            if key == ord("d"):
                res = ",".join(str(x) for x in ["ok-hand"] + chained_landmarks)
                file.write(res + "\n")

            # oncoming-fist
            if key == ord("f"):
                res = ",".join(str(x) for x in ["oncoming-fist"] + chained_landmarks)
                file.write(res + "\n")

            # pinching-hand
            if key == ord("g"):
                res = ",".join(str(x) for x in ["pinching-hand"] + chained_landmarks)
                file.write(res + "\n")

            # raised-fist
            if key == ord("h"):
                res = ",".join(str(x) for x in ["raised-fist"] + chained_landmarks)
                file.write(res + "\n")

            # raised-hand
            if key == ord("j"):
                res = ",".join(str(x) for x in ["raised-hand"] + chained_landmarks)
                file.write(res + "\n")

            # sign-of-the-horns
            if key == ord("k"):
                res = ",".join(str(x) for x in ["sign-of-the-horns"] + chained_landmarks)
                file.write(res + "\n")

            # thumbs-down
            if key == ord("l"):
                res = ",".join(str(x) for x in ["thumbs-down"] + chained_landmarks)
                file.write(res + "\n")

            # thumbs-up
            if key == ord("w"):
                res = ",".join(str(x) for x in ["thumbs-up"] + chained_landmarks)
                file.write(res + "\n")

            # victory-hand
            if key == ord("e"):
                res = ",".join(str(x) for x in ["victory-hand"] + chained_landmarks)
                file.write(res + "\n")

        # draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # show the output frame
        cv2.imshow("frame", image)

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
vs.stop()
