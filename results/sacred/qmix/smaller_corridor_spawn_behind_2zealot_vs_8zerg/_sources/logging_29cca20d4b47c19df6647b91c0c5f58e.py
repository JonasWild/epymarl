import io
from collections import defaultdict

import PIL

import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
from torchvision.transforms import ToTensor

from myutils.HeatSC2map import MyHeatmap, HeatSC2map


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        self.heatmap_steps = 0

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value, log_images
        configure(directory_name)
        self.tb_logger = log_value
        self.tb_image_logger = log_images
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_heatmaps(self, heatmaps: [MyHeatmap]):
        if self.use_tb and heatmaps is not None:
            for heatmap in heatmaps:

                # Reverse the y-axis data
                heatmap_data = heatmap.preprocess_data()

                # Create the heatmap using matplotlib
                fig = plt.figure(figsize=(len(heatmap_data), len(heatmap_data[0])))

                if np.all(np.isnan(heatmap_data)):
                    vmin = 0
                    vmax = 0
                    print(f"All values are NaN in {heatmap.name}")
                elif np.all(heatmap_data >= 0):
                    vmin = 0
                    vmax = np.nanmax(np.abs(heatmap_data))
                    cmap = plt.cm.get_cmap('YlOrRd')
                elif np.all(heatmap_data <= 0):
                    vmax = 0
                    vmin = np.nanmin(heatmap_data)
                    cmap = plt.cm.get_cmap('Blues_r')
                else:
                    vmax = np.nanmax(np.abs(heatmap_data))
                    vmin = np.nanmin(np.abs(heatmap_data))
                    cmap = plt.cm.get_cmap('bwr')

                # Define the x and y axis labels
                # x_labels = [f'X{i}' for i in range(len(heatmap_data))]
                # y_labels = [f'Y{i}' for i in range(len(heatmap_data))]

                # Set the x and y axis labels
                # plt.xticks(range(num_cols), x_labels)
                # plt.yticks(range(num_rows), y_labels)

                plt.xlabel(heatmap.x_label)
                plt.ylabel(heatmap.y_label)

                # Add a title to the plot with the name of the metric
                plt.title(heatmap.name)

                plt.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
                plt.colorbar()

                if isinstance(heatmap, HeatSC2map):
                    background_image = plt.imread(f"./assets/{HeatSC2map.bg_image_file}")
                    height, width, _ = background_image.shape
                    plt.imshow(background_image, extent=[0, width, 0, height], alpha=0.5)

                # Convert figure to a NumPy array
                canvas = fig.canvas
                canvas.draw()
                width, height = canvas.get_width_height()
                figure_data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                figure_data = figure_data.reshape((height, width, 3))

                plt.close(fig)

                self.tb_image_logger(heatmap.name, [figure_data], self.heatmap_steps)
                print(f"logging {heatmap.name} at step {self.heatmap_steps} from vmin={vmin} to vmax={vmax}")

            self.heatmap_steps += 1

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    return logger


