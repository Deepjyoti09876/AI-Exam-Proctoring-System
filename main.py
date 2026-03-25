import subprocess
import sys
import os

if __name__ == "__main__":
    # Get path to exam_camera_detection.py in same folder
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "exam_camera_detection.py")

    if not os.path.exists(script):
        print("[ERROR] exam_camera_detection.py not found in the same folder.")
        sys.exit(1)

    print("Starting AI Proctor System...")
    subprocess.run([sys.executable, script])