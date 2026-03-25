import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# ================= DETECTION RANGES =================
YAW_MIN          = 18
YAW_MAX          = 34
PITCH_DOWN_LIMIT = 10
PITCH_UP_LIMIT   = 24

# FIXED: right iris was escaping at 0.65 — tightened to 0.60
# These limits apply to the AVERAGE of both eye ratios
IRIS_LEFT_LIMIT  = 0.44
IRIS_RIGHT_LIMIT = 0.60

SMOOTHING = 6
# ====================================================


def compute_iris_ratios(face_landmarks):
    """
    Computes iris position ratios for both eyes independently.
    Formula: (iris_center - outer_corner) / eye_width
      ~0.5  = eyes looking straight ahead
      LOW   = looking toward outer side (away from nose)
      HIGH  = looking toward inner side (toward nose)
    """
    LEFT_IRIS  = [474, 475, 476, 477]   # person's left  eye
    RIGHT_IRIS = [469, 470, 471, 472]   # person's right eye

    # Left eye: outer corner=33, inner corner=133
    l_outer = face_landmarks.landmark[33].x
    l_inner = face_landmarks.landmark[133].x
    l_width = abs(l_inner - l_outer) + 1e-6

    # Right eye: inner corner=362, outer corner=263
    r_inner = face_landmarks.landmark[362].x
    r_outer = face_landmarks.landmark[263].x
    r_width = abs(r_outer - r_inner) + 1e-6

    l_center = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
    r_center = np.mean([face_landmarks.landmark[i].x for i in RIGHT_IRIS])

    left_ratio  = float(np.clip((l_center - l_outer) / l_width,  0.0, 1.0))
    right_ratio = float(np.clip((r_center - r_inner) / r_width,  0.0, 1.0))
    avg_ratio   = (left_ratio + right_ratio) / 2.0

    return left_ratio, right_ratio, avg_ratio


def is_iris_cheating(left_ratio, right_ratio, avg_ratio):

    # Average out of range
    if avg_ratio < IRIS_LEFT_LIMIT or avg_ratio > IRIS_RIGHT_LIMIT:
        return True
    # Either individual eye strongly out of range
    margin = 0.06
    if (left_ratio  < IRIS_LEFT_LIMIT  - margin or
        left_ratio  > IRIS_RIGHT_LIMIT + margin or
        right_ratio < IRIS_LEFT_LIMIT  - margin or
        right_ratio > IRIS_RIGHT_LIMIT + margin):
        return True
    return False


class CameraDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.iris_history = []

    def process_frame(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        cheating_head = False
        cheating_iris = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # HEADPOSE
            nose       = face_landmarks.landmark[1]
            chin       = face_landmarks.landmark[152]
            left_face  = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]

            pitch = (chin.y - nose.y) * 100
            yaw   = (right_face.x - left_face.x) * 100

            if yaw < YAW_MIN or yaw > YAW_MAX:
                cheating_head = True
            if pitch <= PITCH_DOWN_LIMIT or pitch >= PITCH_UP_LIMIT:
                cheating_head = True

            # IRIS
            left_ratio, right_ratio, avg_ratio = compute_iris_ratios(face_landmarks)

            self.iris_history.append(avg_ratio)
            if len(self.iris_history) > SMOOTHING:
                self.iris_history.pop(0)
            smoothed = float(np.mean(self.iris_history))

            # Recompute smoothed left/right for individual check
            if is_iris_cheating(left_ratio, right_ratio, smoothed):
                cheating_iris = True

        return cheating_head, cheating_iris