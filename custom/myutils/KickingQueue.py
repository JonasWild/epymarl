import queue

import numpy as np


class HistoryQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.history = []

    def put(self, item, block=True, timeout=None):
        if 0 < 100000 == self.qsize():
            self.get()

        super().put(item, block, timeout)

    def reset(self):
        cur = []
        while not self.empty():
            cur.append(self.get())
        self.history.append(np.sum(cur))

    def reset_history(self):
        self.history.clear()


class ConditionalResetHistoryQueue(HistoryQueue):
    def __init__(self, reset_condition):
        self.reset_condition = reset_condition
        super().__init__()

    def put(self, item, block=True, timeout=None):
        if not self.empty() and self.reset_condition(self.queue[-1], item):
            self.reset()
            return

        super().put(item, block, timeout)
