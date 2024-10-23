from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from coorperative_rl.envs import Environment


class MapVisualizer:
    def __init__(
        self,
        env: Environment,
        plot_pause_seconds: float | int = 0.7,
        visualize: bool = True,
    ) -> None:
        self.visualize = visualize

        self.env = env

        self.fig, self.ax = None, None

        self.plot_pause_seconds = (
            plot_pause_seconds  # Pause to allow visualization of the movement
        )

    def update(self) -> None:
        if not self.visualize:
            return

        # create figure and axis lazily to avoid showing up on the screen even if visualize is False
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.ax.clear()

        self.ax.set_xlim(0, self.env.grid_size)
        self.ax.set_ylim(0, self.env.grid_size)
        self.ax.set_xticks(np.arange(0, self.env.grid_size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.env.grid_size + 1, 1))
        self.ax.grid(True)

        # draw agents + agent legends
        agent_legends = []
        for agent_state in self.env.state.agent_states.values():
            agent_icon = agent_state.type.name.lstrip("TYPE_") + str(agent_state.id)
            agent_color = "red" if agent_state.has_full_key else "black"

            # agent location
            self.ax.text(
                agent_state.location[0] + 0.5,
                agent_state.location[1] + 0.5,
                agent_icon,
                ha="center",
                va="center",
                fontsize=16,
                color=agent_color,
            )
            # agent legend
            agent_legend = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=agent_color,
                markersize=8,
                label=f"Agent ({agent_icon})",
            )
            agent_legends.append(agent_legend)

        # draw goal + goal legend
        self.ax.text(
            self.env.state.goal_location[0] + 0.5,
            self.env.state.goal_location[1] + 0.5,
            "G",
            ha="center",
            va="center",
            fontsize=16,
            color="blue",
        )
        goal_legend = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Goal (G)",
        )

        self.ax.legend(
            handles=agent_legends + [goal_legend],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.subplots_adjust(right=0.75, left=0.1)
        self.fig.canvas.draw_idle()
        plt.pause(self.plot_pause_seconds)

    def close(self) -> None:
        try:
            plt.close(self.fig)
        except Exception:
            # env might have closed this fig already
            pass
