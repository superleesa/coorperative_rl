import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import jsonlines
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking.fluent import ActiveRun

from coorperative_rl.utils import flatten_2D_list
from coorperative_rl.types import SARS, serialize_sars


class BaseTracker(ABC):
    def can_log_sars(self) -> bool:
        return False
    
    @abstractmethod
    def __enter__(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int) -> None:
        raise NotImplementedError
    
    def log_sars(self, sars_list: list[SARS], step: int) -> None:
        if self.can_log_sars():
            raise NotImplementedError
        raise ValueError("This tracker does not support logging SARS")


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

    def __init__(self) -> None:
        self.max_rows, self.max_cols = 3, 2
        self.data: dict[
            str, tuple[plt.Axes, list[int], list[float]]
        ] = {}  # [ax, step, values]

    def __enter__(self) -> Self:
        self.fig, axs = plt.subplots(
            self.max_rows, self.max_cols
        )
        self.axs = flatten_2D_list(axs)
        self.axs_unused = self.axs.copy()  # should point to the same axes
        self.axs_unused.reverse()  # so that we can pop from the end
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        plt.close(self.fig)
        del self.data

    def log_metric(self, key: str, value: float, step: int) -> None:
        if key not in self.data and len(self.data) < self.max_cols*self.max_rows:
            self.data[key] = (self.axs_unused.pop(), [], [])
        elif key not in self.data:
            warnings.warn(
                f"Can't plot more than {self.max_cols*self.max_rows} metrics"
            )
            return

        self.data[key][1].append(step)
        self.data[key][2].append(value)
        ax = self.data[key][0]

        ax.clear()
        ax.plot(self.data[key][1], self.data[key][2])
        ax.set_title(f"{key}", fontsize=9)
        
        self.fig.suptitle("Validation Metrics", fontsize=14)

        self.fig.subplots_adjust(wspace=0.3, hspace=0.6)
        self.fig.canvas.draw_idle()
        plt.pause(0.002)


class JSONLinesTracker(BaseTracker):
    JSONLinesDataPath = "jsonlines_logs"
    def __init__(self) -> None:
        # createa a log file
        path = Path(self.JSONLinesDataPath)
        path.mkdir(parents=True, exist_ok=True)
        log_files = [file.name for file in path.glob("*.jsonl") if file.name.startswith("log")]
        if log_files:
            self.file_path = f"{self.JSONLinesDataPath}/log_{len(log_files)}.jsonl"
        else:
            self.file_path = f"{self.JSONLinesDataPath}/log_0.jsonl"
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        return
    
    def log_metric(self, key: str, value: float, step: int) -> None:
        with jsonlines.open(self.file_path, mode='a') as writer:
            writer.write({"metric": key, "value": value, "step": step})
    
    def can_log_sars(self) -> bool:
        return True

    def log_sars(self, sars_list: list[SARS], step: int) -> None:
        # stringified_sars = base64.b64encode(pickle.dumps(sars)).decode('utf8')
        with jsonlines.open(self.file_path, mode='a') as writer:
            writer.write({"metric": "sars", "value": [serialize_sars(sars) for sars in sars_list], "step": step})


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


def select_tracker(tracking: bool, tracker_type: str = "mlflow") -> BaseTracker:
    if not tracking:
        return NullTracker()
    if tracker_type == "mlflow":
        return MLFlowTracker()
    elif tracker_type == "matplotlib":
        return MatplotlibPlotTraker()
    elif tracker_type == "jsonlines":
        return JSONLinesTracker()
    raise ValueError(f"Unknown tracker type: {tracker_type}")
