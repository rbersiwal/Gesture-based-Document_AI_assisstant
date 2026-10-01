import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

st.title("Gesture Recognition")

start_camera = st.button("Start Camera")
stop_camera = st.button("Stop Camera")

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if start_camera:
    st.session_state.camera_running = True

if stop_camera:
    st.session_state.camera_running = False


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def count_fingers(hand):

    fingers = 0

    wrist = hand[0]

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]

    for tip, pip in finger_pairs:

        tip_distance = distance(
            hand[tip],
            wrist
        )

        pip_distance = distance(
            hand[pip],
            wrist
        )

        if tip_distance > pip_distance * 1.10:
            fingers += 1

    return fingers


if st.session_state.camera_running:

    model_path = "models/hand_landmarker.task"

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )

    detector = vision.HandLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()
    gesture_placeholder = st.empty()

    try:

        while st.session_state.camera_running:

            ret, frame = cap.read()

            if not ret:

                st.error("Camera could not be opened.")
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = detector.detect(mp_image)

            gesture = "NO HAND"

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                finger_count = count_fingers(hand)

                if finger_count == 0:

                    gesture = "CLEAR"

                elif finger_count == 1:

                    gesture = "SEARCH"

                elif finger_count == 2:

                    gesture = "NEXT"

                elif finger_count == 4:

                    gesture = "STOP"

                else:

                    gesture = f"{finger_count} FINGERS"

                for landmark in hand:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            gesture_placeholder.subheader(
                f"Gesture: {gesture}"
            )

            frame_placeholder.image(
                frame,
                channels="BGR"
            )

            time.sleep(0.03)

    finally:

        cap.release()
        detector.close()

        frame_placeholder.empty()
        gesture_placeholder.empty()

        st.session_state.camera_running = False