import numpy as np
class HeatSC2map:
    width = 0
    height = 0

    @staticmethod
    def init_map_size(map_width, map_height):
        HeatSC2map.width = map_width
        HeatSC2map.height = map_height

    def __init__(self, name):
        self.data = np.full((HeatSC2map.width, HeatSC2map.height), np.nan)
        self.name = name

    def add_value(self, row, col, value):
        if np.isnan(self.data[row, col]):
            self.data[row, col] = value
        elif not np.isnan(value):
            self.data[row, col] += value

    def get_value(self, row, col):
        return self.data[row, col]
