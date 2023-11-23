import numpy as np


class MyHeatmap:
    def __init__(self, name, width=1, height=1, x_label="time steps", y_label="index"):
        self.name = name
        self.width = width
        self.height = height
        self.data = [[np.nan for _ in range(height)] for _ in range(width)]
        self.y_label = y_label
        self.x_label = x_label

    def add_value(self, row, col, value):
        if row < 0 or col < 0:
            return

        if row >= self.width:
            # Resize the data to accommodate the new row
            for _ in range(row - self.width + 1):
                self.data.append([np.nan for _ in range(self.height)])
            self.width = row + 1

        if col >= self.height:
            # Resize each row to accommodate the new column
            for row_data in self.data:
                row_data.extend([np.nan for _ in range(col - self.height + 1)])
            self.height = col + 1

        if np.isnan(self.data[row][col]):
            self.data[row][col] = value
        elif not np.isnan(value):
            self.data[row][col] += value

    def get_value(self, row, col):
        return self.data[row][col]

    def reset(self):
        self.data = [[np.nan for _ in range(self.height)] for _ in range(self.width)]

    def preprocess_data(self):
        # return np.fliplr(self.data)
        return np.transpose(self.data)

    def __str__(self):
        return str(np.array(self.data))


class HeatSC2map(MyHeatmap):
    width = 0
    height = 0
    bg_image_file = None

    @staticmethod
    def init_map_size(map_width, map_height, bg_image_file: str = None):
        HeatSC2map.width = map_width
        HeatSC2map.height = map_height
        HeatSC2map.bg_image_file = bg_image_file

    def preprocess_data(self):
        # return np.flipud(self.data)
        return np.transpose(self.data)

    def __init__(self, name):
        super().__init__(name, HeatSC2map.width, HeatSC2map.height, x_label="map x", y_label="map y")
