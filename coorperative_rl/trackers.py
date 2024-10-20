import warnings
from abc import ABC, abstractmethod
from typing import Self

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking.fluent import ActiveRun

from coorperative_rl.utils import flatten_2D_list


class BaseTracker(ABC):
    @abstractmethod
    def __enter__(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int) -> None:
        raise NotImplementedError


class MLFlowTracker(BaseTracker):
    """
    NOTE: mlflow has its own global list to keep track of runs,
    so we don't need to call something on a run directly
    but just call mlflow.something (e.g. mlflow.log_metric)
    """

    def __init__(self) -> None:
        self.run: ActiveRun

    def __enter__(self) -> Self:
        self.run = (
            mlflow.start_run().__enter__()
        )  # doesn't really do anything because ActiveRun.__enter__ just returns self
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        self.run.__exit__(exc_type, exc_val, exc_tb)

    def log_metric(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key, value, step)


class MatplotlibPlotTraker(BaseTracker):
    """
    We need this because we can't use third-party library in our assignment... (don't use this in real life)
    this just holds records in memory

    bascially, have a big figure and for each ax, draw visualization of metric
    """

    MAX_NUM_PLOTS = 9  # assume we have 9 metrics to plot and no more

    def __init__(self) -> None:
        self.max_num_plots = 9
        self.data: dict[
            str, tuple[plt.Axes, list[int], list[float]]
        ] = {}  # [ax, step, values]

    def __enter__(self) -> Self:
        self.fig, axs = plt.subplots(
            MatplotlibPlotTraker.MAX_NUM_PLOTS // 3,
            MatplotlibPlotTraker.MAX_NUM_PLOTS // 3,
        )
        self.axs = flatten_2D_list(axs)
        self.axs_unused = self.axs.copy()  # should point to the same axes
        self.axs_unused.reverse()  # so that we can pop from the end

        plt.show(
            block=False
        )  # make this specific figure non-blocking (so that we can update it)
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        plt.close(self.fig)
        del self.data

    def log_metric(self, key: str, value: float, step: int) -> None:
        if key not in self.data and len(self.data) < MatplotlibPlotTraker.MAX_NUM_PLOTS:
            self.data[key] = (self.axs_unused.pop(), [], [])
        elif key not in self.data:
            warnings.warn(
                f"Can't plot more than {MatplotlibPlotTraker.MAX_NUM_PLOTS} metrics"
            )
            return

        self.data[key][1].append(step)
        self.data[key][2].append(value)
        ax = self.data[key][0]

        ax.clear()
        ax.plot(self.data[key][1], self.data[key][2])
        ax.set_title(f"{key} over time")

        self.fig.canvas.flush_events()
        plt.pause(0.001)  # small pause to allow GUI events to be processed


class NullTracker(BaseTracker):
    "A tracker, when tracking is set to False"

    def __init__(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int) -> None:
        pass
