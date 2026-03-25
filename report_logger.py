
import csv
import os
from datetime import datetime

TRAINING_CSV = "training_data.csv"
REPORT_CSV   = "exam_report.csv"
FIELDNAMES   = ["yaw", "pitch", "iris", "face_visible", "tab_switch", "label"]


def save_report(head: int, iris: int, tab_switches: int = 0):
    first = not os.path.exists(REPORT_CSV) or os.path.getsize(REPORT_CSV) == 0
    with open(REPORT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["Timestamp", "Head Movements", "Iris Movements", "Tab Switches"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            head, iris, tab_switches
        ])
    print(f"[Report] Saved → Head:{head}  Iris:{iris}  Tabs:{tab_switches}")


def append_session_data(session_rows: list):

    if not session_rows:
        print("[Data]  Nothing to save this session.")
        return
    first = not os.path.exists(TRAINING_CSV) or os.path.getsize(TRAINING_CSV) == 0
    with open(TRAINING_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if first:
            w.writeheader()
        w.writerows(session_rows)
    n = sum(1 for r in session_rows if r["label"] == "normal")
    c = sum(1 for r in session_rows if r["label"] == "cheating")
    print(f"[Data]  {len(session_rows)} rows saved → Normal:{n}  Cheating:{c}")


def get_dataset_stats():
    if not os.path.exists(TRAINING_CSV) or os.path.getsize(TRAINING_CSV) == 0:
        print("[Data]  training_data.csv not found yet — will be created after first session.")
        return
    with open(TRAINING_CSV, "r") as f:
        rows = list(csv.DictReader(f))
    n = sum(1 for r in rows if r.get("label") == "normal")
    c = sum(1 for r in rows if r.get("label") == "cheating")
    print(f"[Data]  Dataset: {len(rows)} total  |  Normal:{n}  Cheating:{c}")
