import os
from collections import defaultdict

from matplotlib import ticker, gridspec

import logging
import numpy as np
import matplotlib.pyplot as plt

from custom.myutils.HeatSC2map import MyHeatmap, HeatSC2map
from runners import EpisodeRunner


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        self.heatmap_steps = 0
        self.heatmap_dict = {}

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

    def log_heatmap_stat(self, name, col_values: []):
        if name not in self.heatmap_dict:
            self.heatmap_dict[name] = MyHeatmap(name, height=len(col_values))

        row = self.heatmap_dict[name].width
        for col, value in enumerate(col_values):
            self.heatmap_dict[name].add_value(row, col, value)

    def log_heatmaps(self, heatmaps: [MyHeatmap]):
        heatmaps = heatmaps + list(self.heatmap_dict.values())
        if self.use_tb and heatmaps is not None:
            for heatmap in heatmaps:

                # Reverse the y-axis data
                heatmap_data = heatmap.preprocess_data()

                # Calculate the aspect ratio of the heatmap data
                heatmap_aspect = heatmap_data.shape[1] / heatmap_data.shape[0]

                # Set the width of the colorbar (in inches)
                colorbar_width = 0.5

                # Set the padding for labels around the heatmap and colorbar (in inches)
                label_padding = 0.5

                default_size = 6.0

                # Calculate the width of the heatmap based on the number of columns and the colorbar width and labels padding
                heatmap_width = default_size - colorbar_width - 2 * label_padding

                # Calculate the figsize based on the heatmap and colorbar widths and the aspect ratio
                fig_width = max(heatmap_width + colorbar_width + 2 * label_padding, default_size)
                fig_height = max(fig_width / heatmap_aspect, 8.0)

                # Create a figure with the calculated figsize
                fig = plt.figure(figsize=(fig_width, fig_height))

                # Calculate the relative width ratio for the heatmap and the colorbar
                heatmap_width_ratio = heatmap_width / fig_width
                colorbar_width_ratio = colorbar_width / fig_width

                # Create a GridSpec with the desired aspect ratio and one column
                gs = gridspec.GridSpec(1, 2, width_ratios=[heatmap_width_ratio, colorbar_width_ratio])

                if np.all(np.logical_or(np.isnan(heatmap_data), heatmap_data >= 0)):
                    vmin = 0
                    vmax = np.nanmax(heatmap_data)
                    cmap = plt.cm.get_cmap('YlOrRd')
                elif np.all(np.logical_or(np.isnan(heatmap_data), heatmap_data <= 0)):
                    vmax = 0
                    vmin = np.nanmin(heatmap_data)
                    cmap = plt.cm.get_cmap('Blues_r')
                else:
                    vmax = np.nanmax(heatmap_data)
                    vmin = np.nanmin(heatmap_data)
                    cmap = plt.cm.get_cmap('bwr')

                # Define the x and y axis labels
                # x_labels = [f'X{i}' for i in range(len(heatmap_data))]
                # y_labels = [f'Y{i}' for i in range(len(heatmap_data))]

                # Set the x and y axis labels
                # plt.xticks(range(num_cols), x_labels)
                # plt.yticks(range(num_rows), y_labels)

                if not isinstance(heatmap, HeatSC2map):
                    ax = plt.gca()
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*EpisodeRunner.heatmap_size:.0f}'))

                plt.xlabel(heatmap.x_label, fontsize=14, labelpad=label_padding)
                plt.ylabel(heatmap.y_label, fontsize=14, labelpad=label_padding)

                # Add a title to the plot with the name of the metric
                plt.title(heatmap.name, fontsize=16)

                ax_heatmap = plt.subplot(gs[0])
                heatmap_plot = ax_heatmap.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

                ax_colorbar = plt.subplot(gs[1])
                cbar = plt.colorbar(heatmap_plot, cax=ax_colorbar)
                # cbar.set_label('Colorbar', fontsize=14) TODO

                plt.tight_layout()

                if isinstance(heatmap, HeatSC2map):
                    path = os.path.join(os.path.abspath(__file__), "..", "..", "..", "assets", HeatSC2map.bg_image_file)
                    background_image = plt.imread(path)
                    heatmap_height, heatmap_width = heatmap_data.shape

                    ax_heatmap = plt.subplot(gs[0])
                    bg_plot = ax_heatmap.imshow(background_image, extent=[0, heatmap_width, 0, heatmap_height], alpha=0.3)

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



