import importlib
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser, ActionConfigFile

from .configs import DataConfig, ModelConfig, OptimizerConfig, RunnerConfig
from .runners import Runner
if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands


def add_run_subcommand(subcommands: "_ActionSubCommands"):
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )

    parser.add_argument(
        "--seed",
        type=bool | int,
        default=True,
        help=(
            "Random seed. "
            "If True, a random seed will be generated. "
            "If False, no random seed will be used. "
            "If an integer, that integer will be used as the random seed."
        ),
    )

    parser.add_dataclass_arguments(
        DataConfig,
        nested_key="data",
    )

    parser.add_dataclass_arguments(
        ModelConfig,
        nested_key="model",
    )

    parser.add_dataclass_arguments(
        OptimizerConfig,
        nested_key="optimizer",
    )

    parser.add_dataclass_arguments(
        RunnerConfig,
        nested_key="runner",
    )

    subcommands.add_subcommand(
        "run",
        parser,
        help="Run the pipeline",
    )


def main():
    parser = ArgumentParser(
        description="OmniGen CLI",
        env_prefix="OMNI_GEN",
    )

    subcommands = parser.add_subcommands(required=True)
    add_run_subcommand(subcommands)

    config = parser.parse_args()

    subcommand = config["subcommand"]
    if subcommand == "run":
        config = config[subcommand]

        data_config = DataConfig(**config["data"].as_dict())
        model_config = ModelConfig(**config["model"].as_dict())
        optimizer_config = OptimizerConfig(**config["optimizer"].as_dict())
        runner_config = RunnerConfig(**config["runner"].as_dict())

        # Instantiate the runner
        class_module, class_name = runner_config.runner_cls_path.rsplit(".", 1)
        module = importlib.import_module(class_module)
        runner_cls = getattr(module, class_name)
        assert issubclass(runner_cls, Runner), f"{runner_cls} is not a subclass of Runner"

        runner = runner_cls(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            runner_config=runner_config,
        )

        # Run the pipeline
        runner()
    else:
        raise ValueError(f"Invalid subcommand: {subcommand}")


if __name__ == "__main__":
    main()
