# Reverse the y-axis data
import numpy as np
from matplotlib import pyplot as plt, gridspec

from myutils.HeatSC2map import HeatSC2map

heatmap_data = np.random.rand(14, 100)

# Calculate the aspect ratio of the heatmap data
heatmap_aspect = heatmap_data.shape[1] / heatmap_data.shape[0]

# Set the width of the colorbar (in inches)
colorbar_width = 0.5

# Set the padding for labels around the heatmap and colorbar (in inches)
label_padding = 0.5

default_size = 8.0

# Calculate the width of the heatmap based on the number of columns and the colorbar width and labels padding
fig_width = min((default_size * heatmap_aspect) + colorbar_width + 2 * label_padding, 16)

fig = plt.figure(figsize=(fig_width, default_size))

# Calculate the relative width ratio for the heatmap and the colorbar
colorbar_width_ratio = colorbar_width / fig_width
heatmap_width_ratio = 1 - colorbar_width_ratio

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

#if not isinstance(heatmap, HeatSC2map):
#    ax = plt.gca()
#    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * EpisodeRunner.heatmap_size:.0f}'))

#plt.xlabel(heatmap.x_label, fontsize=14, labelpad=label_padding)
#plt.ylabel(heatmap.y_label, fontsize=14, labelpad=label_padding)

# Add a title to the plot with the name of the metric
#plt.title(heatmap.name, fontsize=16)

ax_heatmap = plt.subplot(gs[0])
heatmap_plot = ax_heatmap.imshow(heatmap_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

ax_colorbar = plt.subplot(gs[1])
cbar = plt.colorbar(heatmap_plot, cax=ax_colorbar)
# cbar.set_label('Colorbar', fontsize=14) TODO

plt.tight_layout()
plt.show()
