class HeatmapQueueValue:
    def __init__(self, heatmap_value, queue_value):
        self.heatmap_value = heatmap_value
        self.queue_value = queue_value

    def __str__(self):
        return f"heatmap: {self.heatmap_value}, queue: {self.queue_value}"
