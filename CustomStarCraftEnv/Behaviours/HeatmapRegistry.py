from myutils.HeatSC2map import HeatSC2map


class HeatmapRegistry:
    def __init__(self, name, n_agents):
        self.name = name
        self.n_agents = n_agents
        self.heatmaps = [HeatSC2map(f"{name}_{i}") for i in range(n_agents)]

    def add_data(self, index, row, col, value):
        self.heatmaps[index].add_value(row, col, value)

    def reset_heatmaps(self):
        for heatmap in self.heatmaps:
            heatmap.reset()

    def get_single_agent_heatmap(self, index):
        return self.heatmaps[index]

    def get_summed_agent_heatmap(self):
        summed_heatmap = HeatSC2map(self.name + "_summed")
        for x, row in enumerate(summed_heatmap.data):
            for y, _ in enumerate(row):
                for h in self.heatmaps:
                    summed_heatmap.add_value(x, y, h.get_value(x, y))
        return summed_heatmap
