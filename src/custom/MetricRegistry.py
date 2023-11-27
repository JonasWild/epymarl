from custom.CustomStarCraftEnv.Metrics.DistanceToEnemiesPerEpisode import DistanceToEnemiesPerEpisode
from custom.CustomStarCraftEnv.Metrics.StepsTillAllAttacked import StepsTillAllAttacked
from custom.CustomStarCraftEnv.Metrics.StepsTillFirstAttacked import StepsTillFirstAttacked


class MetricRegistry:
    def __init__(self, config, n_agents, metrics_mapping):
        self.config = config
        self.n_agents = n_agents
        self.metrics = []
        self.episodes = 0
        self.metrics_mapping = metrics_mapping

    def initialize(self):
        # Check if self.config is not None
        if self.config:
            # Check if 'metrics' exists and is a dictionary
            metrics = self.config.get('metrics')
            if metrics and isinstance(metrics, dict):
                # Register all behaviors based on the config
                for metric_name, metric_config in metrics.items():
                    metric_class = self.metrics_mapping.get(metric_name)
                    if metric_class:
                        metric = metric_class(metric_config, self.n_agents)
                        self.metrics.append(metric)
            else:
                print("Metrics not found or not a dictionary")
        else:
            print("Config is None")

    def add_metrics_data(self, env, actions, obs):
        for index, metric in enumerate(self.metrics):
            metric.add_data(env, actions, obs)

    def evaluate_episode(self, env):
        self.episodes += 1
        for metric in self.metrics:
            metric.evaluate_episode(env)

    def get_metric_results(self):
        results = {}
        for metric in self.metrics:
            result = metric.get_and_reset_total()
            results[metric.__class__.__name__] = result / self.episodes
        self.episodes = 0
        return results

