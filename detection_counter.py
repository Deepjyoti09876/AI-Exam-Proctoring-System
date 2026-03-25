"""
detection_counter.py
Debounced counter — only increments ONCE per sustained violation window.
"""
import time


class DetectionCounter:
    HEAD_TIME = 2.0
    IRIS_TIME = 3.0

    def __init__(self):
        self.head_count       = 0
        self.iris_count       = 0
        self.tab_switch_count = 0
        self._head_timer      = None
        self._iris_timer      = None
        self._head_alerted    = False
        self._iris_alerted    = False

    def update(self, cheating_head: bool, cheating_iris: bool):
        now = time.time()
        if cheating_head:
            if self._head_timer is None:
                self._head_timer = now; self._head_alerted = False
            elif not self._head_alerted and (now - self._head_timer) >= self.HEAD_TIME:
                self.head_count += 1; self._head_alerted = True
        else:
            self._head_timer = None; self._head_alerted = False

        if cheating_iris and not cheating_head:
            if self._iris_timer is None:
                self._iris_timer = now; self._iris_alerted = False
            elif not self._iris_alerted and (now - self._iris_timer) >= self.IRIS_TIME:
                self.iris_count += 1; self._iris_alerted = True
        else:
            self._iris_timer = None; self._iris_alerted = False

    def increment_tab_switch(self):
        self.tab_switch_count += 1

    def get_report(self):
        return {
            "Head Movements": self.head_count,
            "Iris Movements": self.iris_count,
            "Tab Switches":   self.tab_switch_count
        }
