import queue


class KickingQueue(queue.Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize)

    def put(self, item, block=True, timeout=None):
        if 0 < self.maxsize == self.qsize():
            self.get()

        super().put(item, block, timeout)

    def reset(self):
        while not self.empty():
            self.get()


class ConditionalResetKickingQueue(KickingQueue):
    def __init__(self, maxsize, reset_condition):
        self.reset_condition = reset_condition
        super().__init__(maxsize)

    def put(self, item, block=True, timeout=None):
        if not self.empty() and self.reset_condition(self.queue[-1], item):
            self.reset()
        super().put(item, block, timeout)
