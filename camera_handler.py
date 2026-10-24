import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import calculate_distance, smooth_value
from volume_control import set_system_volume


class CameraHandler:
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(self, model_path="models/hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.prev_volume = 0.5
        self.alpha = 0.2
        self.timestamp_ms = 0

    def get_gesture_volume(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.hands.detect_for_video(image, self.timestamp_ms)
        self.timestamp_ms += 1
        current_volume = self.prev_volume

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                height, width = frame.shape[:2]
                points = [
                    (int(landmark.x * width), int(landmark.y * height))
                    for landmark in hand_landmarks
                ]
                for start, end in self.HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)

                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                distance = calculate_distance(
                    (thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y)
                )

                min_distance = 0.04
                max_distance = 0.25
                distance = max(min_distance, min(max_distance, distance))
                raw_volume = (distance - min_distance) / (
                    max_distance - min_distance
                )
                current_volume = smooth_value(
                    raw_volume, self.prev_volume, self.alpha
                )
                self.prev_volume = current_volume

                bar_width = int(current_volume * 200)
                cv2.rectangle(
                    frame, (20, 200), (20 + bar_width, 220), (0, 255, 0), -1
                )
                cv2.putText(
                    frame,
                    f"Volume: {int(current_volume * 100)}%",
                    (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                set_system_volume(current_volume)

        return frame, current_volume

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.resize(frame, (640, 480))
                processed_frame, _ = self.get_gesture_volume(frame)
                cv2.imshow("Gesture Volume Control", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
