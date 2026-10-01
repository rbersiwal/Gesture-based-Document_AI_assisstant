import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        command = "No Gesture"

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            tips = [8, 12, 16, 20]
            joints = [6, 10, 14, 18]

            fingers = 0

            for tip, joint in zip(tips, joints):
                if hand[tip].y < hand[joint].y:
                    fingers += 1

            if hand[4].x < hand[3].x:
                fingers += 1

            if fingers == 1:
                command = "SEARCH"

            elif fingers == 2:
                command = "NEXT"

            elif fingers == 5:
                command = "STOP"

            elif fingers == 0:
                command = "CLEAR"

        cv2.putText(
            frame,
            command,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.imshow("Gesture Command Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()