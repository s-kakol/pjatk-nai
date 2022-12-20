import math
import cv2
import mediapipe as mp


def get_angle(vertice1, vertice2, vertice3):
    x1, y1, _ = vertice1
    x2, y2, _ = vertice2
    x3, y3, _ = vertice3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return angle


face_cascade = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    raise IOError('Failed to load cascade classifier xml file')

color_red_kill = (0, 0, 255)
color_green_live = (0, 255, 0)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
camera = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while camera.isOpened():
        _, frame = camera.read()
        headshot_mode = True
        color = (0, 255, 0)

        if not _:
            continue

        landmarks = []

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_processed = pose.process(frame)

        height, width, _ = frame.shape

        for landmark in mp_processed.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

        left_elbow = get_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        left_shoulder = get_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

        right_elbow = get_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        right_shoulder = get_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

        # print(f"LEFT ELBOW: {left_elbow}")
        # print(f"RIGHT ELBOW: {right_elbow}")
        # print(f"LEFT SHOULDER: {left_shoulder}")
        # print(f"RIGHT SHOULDER: {right_shoulder}")

        if (70 < left_elbow < 125 and 240 < right_elbow < 300) or (
                175 < left_shoulder < 260 and 115 < right_shoulder < 190):
            headshot_mode = False

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detection = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_detection:
            if headshot_mode:
                color = (0, 0, 255)
                cv2.line(frame, (x + w // 2, y + h), (x + w // 2, y), color, 2)
                cv2.line(frame, (x + w, y + h // 2), (x, y + h // 2), color, 2)
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, color, 4)

        cv2.imshow('Headshot', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
