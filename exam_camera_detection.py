
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from report_logger import save_report, append_session_data, get_dataset_stats

head_cheat_count = 0
iris_cheat_count = 0
tab_switch_count = 0

#=DETECTION RANGES =
YAW_MIN          = 18
YAW_MAX          = 34
PITCH_DOWN_LIMIT = 10
PITCH_UP_LIMIT   = 24

IRIS_LEFT_LIMIT  = 0.09
IRIS_RIGHT_LIMIT = 0.31

HEAD_TIME         = 2
IRIS_TIME         = 3
ALERT_TIME        = 2
SMOOTHING         = 6
FACE_MISSING_TIME = 3
SETUP_HOLD_TIME   = 3

# AUTO_COLLECT  : saves 1 frame every SAMPLE_EVERY_N frames automatically
# Manual C/N    : press C or N any time to force-label the current frame
AUTO_COLLECT   = True
SAMPLE_EVERY_N = 3        # collect 1 in every 3 frames (avoids duplicates)

session_rows  = []        # all rows collected this session → saved on ESC
frame_counter = 0
manual_label  = None      # set to "cheating" or "normal" when C/N pressed

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

head_timer          = None
iris_timer          = None
face_missing_timer  = None
alert_active        = False
alert_start         = None
iris_history        = []

phase               = "SETUP"
setup_hold_start    = None
confirm_flash_start = None

print("\n" + "="*48)
get_dataset_stats()
print("="*48)
print("KEYS:  N=Normal  C=Cheating  T=TabSwitch  ESC=Quit")
print("="*48 + "\n")

def compute_iris(face_landmarks):
    LEFT_IRIS  = [474, 475, 476, 477]
    face_left  = face_landmarks.landmark[234].x
    face_right = face_landmarks.landmark[454].x
    face_cx    = (face_left + face_right) / 2.0
    face_width = abs(face_right - face_left) + 1e-6
    iris_x     = np.mean([face_landmarks.landmark[i].x for i in LEFT_IRIS])
    return float((iris_x - face_cx) / face_width)

def draw_iris_bar(frame, x, y, width, height,
                  value, v_min, v_max,
                  safe_min, safe_max, is_cheating):

    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)

    def to_px(v):
        ratio = (float(v) - v_min) / (v_max - v_min + 1e-6)
        return int(x + np.clip(ratio, 0.0, 1.0) * width)

    safe_start_px = to_px(safe_min)
    safe_end_px   = to_px(safe_max)

    cv2.rectangle(frame, (x + 1, y + 2),
                  (safe_start_px, y + height - 2), (30, 30, 180), -1)
    cv2.rectangle(frame, (safe_start_px, y + 2),
                  (safe_end_px, y + height - 2), (0, 180, 0), -1)
    cv2.rectangle(frame, (safe_end_px, y + 2),
                  (x + width - 1, y + height - 2), (30, 30, 180), -1)

    cv2.putText(frame, str(safe_min),
                (safe_start_px - 10, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)
    cv2.putText(frame, str(safe_max),
                (safe_end_px - 10, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)
    cv2.putText(frame, str(v_min),
                (x, y + height + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    cv2.putText(frame, str(v_max),
                (x + width - 28, y + height + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    dot_px  = to_px(value)
    dot_col = (0, 0, 220) if is_cheating else (0, 220, 220)
    dot_r   = height // 2 + 3
    cv2.circle(frame, (dot_px, y + height // 2), dot_r, dot_col, -1)
    cv2.circle(frame, (dot_px, y + height // 2), dot_r, (255, 255, 255), 1)

    cv2.putText(frame, "Iris Bar",
                (x - 75, y + height - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)


def draw_setup_overlay(frame, yaw, pitch, iris_val,
                       in_range, hold_progress, countdown_left):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, "SETUP: Position Yourself Correctly",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)

    yaw_ok   = YAW_MIN          <= yaw   <= YAW_MAX
    pitch_ok = PITCH_DOWN_LIMIT <= pitch <= PITCH_UP_LIMIT

    yaw_c   = (0, 220, 0) if yaw_ok   else (0, 60, 255)
    pitch_c = (0, 220, 0) if pitch_ok else (0, 60, 255)

    if yaw_ok:           yaw_m = "Face straight (OK)"
    elif yaw < YAW_MIN:  yaw_m = "Turn LEFT  <--"
    else:                yaw_m = "Turn RIGHT -->"

    if pitch_ok:                    pitch_m = "Head level (OK)"
    elif pitch < PITCH_DOWN_LIMIT:  pitch_m = "Tilt DOWN  v"
    else:                           pitch_m = "Tilt UP  ^"

    cv2.putText(frame, "Head Yaw:   " + str(int(yaw))   + "  " + yaw_m,
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yaw_c,   2)
    cv2.putText(frame, "Head Pitch: " + str(int(pitch)) + "  " + pitch_m,
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_c, 2)
    cv2.putText(frame, "Iris:       " + "{:.2f}".format(iris_val) + "  (info only)",
                (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if in_range:
        st = "Hold still... " + str(round(countdown_left, 1)) + "s"
        sc = (0, 220, 0)
    else:
        st = "Adjust Yaw and Pitch until both GREEN"
        sc = (0, 60, 255)

    cv2.putText(frame, st, (30, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2)

    bx, by, bw, bh = 30, 255, w - 60, 22
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    if in_range and hold_progress > 0:
        filled = int(bw * min(hold_progress, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), (0, 220, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)

    cv2.putText(frame, "Sit 50-70 cm away | Face camera directly",
                (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_confirm_flash(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),
                  (frame.shape[1], frame.shape[0]), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "Calibration Complete",
                (30, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)


with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as face_mesh:

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            continue

        frame_counter += 1

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        cheating_head = False
        cheating_iris = False
        face_vis      = 0

        cur_yaw      = (YAW_MIN + YAW_MAX) / 2.0
        cur_pitch    = (PITCH_DOWN_LIMIT + PITCH_UP_LIMIT) / 2.0
        cur_iris     = (IRIS_LEFT_LIMIT + IRIS_RIGHT_LIMIT) / 2.0
        face_detected = False

        if results.multi_face_landmarks:
            face_missing_timer = None
            face_detected      = True
            face_vis           = 1

            face_lm    = results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, face_lm,
                mp_face_mesh.FACEMESH_CONTOURS,
                None,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

            nose       = face_lm.landmark[1]
            chin       = face_lm.landmark[152]
            left_face  = face_lm.landmark[234]
            right_face = face_lm.landmark[454]

            cur_pitch = (chin.y - nose.y) * 100
            cur_yaw   = (right_face.x - left_face.x) * 100

            raw_iris = compute_iris(face_lm)
            iris_history.append(raw_iris)
            if len(iris_history) > SMOOTHING:
                iris_history.pop(0)
            cur_iris = float(np.mean(iris_history))

        if phase == "SETUP":
            yaw_ok   = YAW_MIN          <= cur_yaw   <= YAW_MAX
            pitch_ok = PITCH_DOWN_LIMIT <= cur_pitch <= PITCH_UP_LIMIT
            in_range = face_detected and yaw_ok and pitch_ok

            if in_range:
                if setup_hold_start is None:
                    setup_hold_start = time.time()
                elapsed        = time.time() - setup_hold_start
                hold_progress  = elapsed / SETUP_HOLD_TIME
                countdown_left = max(0.0, SETUP_HOLD_TIME - elapsed)
            else:
                setup_hold_start = None
                elapsed        = 0.0
                hold_progress  = 0.0
                countdown_left = float(SETUP_HOLD_TIME)

            draw_setup_overlay(frame, cur_yaw, cur_pitch, cur_iris,
                               in_range, hold_progress, countdown_left)

            if in_range and elapsed >= SETUP_HOLD_TIME:
                phase = "CONFIRM"
                confirm_flash_start = time.time()

        elif phase == "CONFIRM":
            draw_confirm_flash(frame)
            if time.time() - confirm_flash_start > 1.5:
                phase = "DETECTION"
                head_timer = iris_timer = face_missing_timer = None
                alert_active = False
                iris_history.clear()
                print("Calibration Complete — monitoring started")

        elif phase == "DETECTION":

            if results.multi_face_landmarks:

                if len(results.multi_face_landmarks) > 1:
                    cv2.putText(frame, "MULTIPLE PERSONS DETECTED",
                                (30, 170), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 3)
                    cheating_head = True

                # Head check
                center_yaw = (YAW_MIN + YAW_MAX) / 2.0
                if abs(cur_yaw - center_yaw) > 6:
                    cheating_head = True
                if cur_pitch > PITCH_UP_LIMIT or cur_pitch < PITCH_DOWN_LIMIT:
                    cheating_head = True

                # Iris check
                if YAW_MIN <= cur_yaw <= YAW_MAX:
                    if cur_iris < IRIS_LEFT_LIMIT or cur_iris > IRIS_RIGHT_LIMIT:
                        cheating_iris = True

                # HUD
                cv2.putText(frame, "Yaw:   " + "{:.2f}".format(cur_yaw),
                            (30, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Pitch: " + "{:.2f}".format(cur_pitch),
                            (30, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Iris:  " + "{:.2f}".format(cur_iris),
                            (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # IRIS BAR
                fw    = frame.shape[1]
                bar_x = 90
                bar_w = fw - bar_x - 20
                bar_h = 16

                draw_iris_bar(
                    frame,
                    x=bar_x, y=128, width=bar_w, height=bar_h,
                    value=cur_iris,
                    v_min=0.0,  v_max=0.45,
                    safe_min=IRIS_LEFT_LIMIT,
                    safe_max=IRIS_RIGHT_LIMIT,
                    is_cheating=cheating_iris
                )

            else:
                if face_missing_timer is None:
                    face_missing_timer = time.time()
                elif time.time() - face_missing_timer > FACE_MISSING_TIME:
                    cheating_head = True

            # Head timer
            if cheating_head and not alert_active:
                if head_timer is None:
                    head_timer = time.time()
                elif time.time() - head_timer > HEAD_TIME:
                    alert_active      = True
                    alert_start       = time.time()
                    head_cheat_count += 1
            else:
                head_timer = None

            # Iris timer
            if cheating_iris and not cheating_head and not alert_active:
                if iris_timer is None:
                    iris_timer = time.time()
                elif time.time() - iris_timer > IRIS_TIME:
                    alert_active      = True
                    alert_start       = time.time()
                    iris_cheat_count += 1
            else:
                iris_timer = None

            # Alert
            if alert_active:
                cv2.putText(frame, "CHEATING DETECTED",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3)
                if time.time() - alert_start > ALERT_TIME:
                    alert_active = False

            # Counts
            cv2.putText(frame, "Head Movements: " + str(head_cheat_count),
                        (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Iris Movements: " + str(iris_cheat_count),
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Tab Switches:   " + str(tab_switch_count),
                        (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "[ EXAM MONITORING ACTIVE ]",
                        (20, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # ── AUTO COLLECT ──────────────────────────────────────
            # Saves 1 frame every SAMPLE_EVERY_N frames automatically
            # If you pressed C or N manually, that overrides the auto-label
            if face_detected and AUTO_COLLECT and frame_counter % SAMPLE_EVERY_N == 0:
                any_cheat  = cheating_head or cheating_iris
                auto_label = "cheating" if any_cheat else "normal"
                # Manual key overrides auto-label
                final_label = manual_label if manual_label else auto_label
                session_rows.append({
                    "yaw":          round(cur_yaw,   3),
                    "pitch":        round(cur_pitch, 3),
                    "iris":         round(cur_iris,  4),
                    "face_visible": face_vis,
                    "tab_switch":   0,
                    "label":        final_label
                })
                manual_label = None   # reset after one use

            # REC indicator on screen
            if AUTO_COLLECT and face_detected:
                rec_label = manual_label if manual_label else (
                    "cheating" if (cheating_head or cheating_iris) else "normal")
                rec_color = (0, 0, 255) if rec_label == "cheating" else (0, 200, 0)
                cv2.putText(frame,
                            "[REC:" + rec_label + "]  N=Normal  C=Cheat  T=Tab",
                            (20, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, rec_color, 1)

        # ── Show frame ────────────────────────────────────────────
        cv2.imshow("Exam Cheating Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ─────────────────────────────────────────
        if phase == "DETECTION":
            if key == ord('n') or key == ord('N'):
                manual_label = "normal"
                print("[Manual] Labelled: NORMAL")

            elif key == ord('c') or key == ord('C'):
                manual_label = "cheating"
                print("[Manual] Labelled: CHEATING")

            elif key == ord('t') or key == ord('T'):
                # Tab switch event — add directly to session_rows
                tab_switch_count += 1
                alert_active  = True
                alert_start   = time.time()
                session_rows.append({
                    "yaw":          round(cur_yaw,   3),
                    "pitch":        round(cur_pitch, 3),
                    "iris":         round(cur_iris,  4),
                    "face_visible": face_vis,
                    "tab_switch":   1,
                    "label":        "cheating"
                })
                print(f"[Tab] Tab switch logged! Total tabs: {tab_switch_count}")

        if key == 27:   # ESC — save and quit
            break


cap.release()
cv2.destroyAllWindows()

print("\n" + "="*48)
print("  Session ended — saving...")
print("="*48)

# 1. Save session summary to exam_report.csv
save_report(head_cheat_count, iris_cheat_count, tab_switch_count)

# 2. Append this session's frames to training_data.csv
append_session_data(session_rows)

# 3. Show updated dataset totals
get_dataset_stats()

print("="*48)
print(f"  Head : {head_cheat_count}")
print(f"  Iris : {iris_cheat_count}")
print(f"  Tabs : {tab_switch_count}")
print(f"  Rows collected this session: {len(session_rows)}")
print("="*48 + "\n")
print("Run  python train_model.py  to retrain the model with new data.")