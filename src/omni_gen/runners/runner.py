import abc

from omni_gen.configs import DataConfig, ModelConfig, OptimizerConfig, RunnerConfig


class Runner(abc.ABC):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        runner_config: RunnerConfig,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.runner_config = runner_config

    @abc.abstractmethod
    def __call__(self):
        ...
