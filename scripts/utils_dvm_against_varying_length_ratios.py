import numpy as np
import matplotlib.pyplot as plt
from musculotendon_ocp import (
    RigidbodyModelWithMuscles,
)


def plot_matplotlib(results, ratios, dts, model, colors, reference_index):
    for ratio in ratios:
        for dt in dts:
            data = results[str(ratio)][str(dt)] if str(ratio) in results and str(dt) in results[str(ratio)] else None
            if data is None:
                continue

            t = np.array(data["t"])
            muscles_fiber_velocity = np.array(data["muscles_fiber_velocity"])
            muscles_force = np.array(data["muscles_force"])
            equilibrated_t_indices = np.array(data["equilibrated_t_indices"])

            plt.figure(f"Muscle fiber velocity and force for a ratio of {ratio} at dt = {dt}")

            # Plot muscle velocities
            plt.subplot(3, 1, 1)
            for m in range(len(model.muscles)):
                plt.plot(
                    t,
                    muscles_fiber_velocity[:, m],
                    label=model.muscles[m].label,
                    color=colors[m],
                    marker="o",
                )
                if equilibrated_t_indices[m] is not None:
                    plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
            plt.title(f"Muscle fiber velocity")
            plt.xlabel("Time (s)")
            plt.ylabel("Muscle fiber velocity (m/s)")
            plt.grid(visible=True)
            plt.legend()

            # Plot muscle forces
            plt.subplot(3, 1, 2)
            for m in range(len(model.muscles)):
                plt.plot(t, muscles_force[:, m], label=model.muscles[m].label, color=colors[m], marker="o")
                if equilibrated_t_indices[m] is not None:
                    plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
            plt.title(f"Muscle force")
            plt.xlabel("Time (s)")
            plt.ylabel("Muscle force (N)")
            plt.grid(visible=True)
            plt.legend()

            # Plot the integrated impulse difference
            plt.subplot(3, 1, 3)
            for m in range(len(model.muscles)):
                cum_diff_force = np.cumsum(muscles_force[:, m] - muscles_force[:, reference_index])
                impulse = np.zeros_like(muscles_force[:, m])
                impulse[1:] = (cum_diff_force[1:] + cum_diff_force[:-1]) * (t[1:] - t[:-1]) / 2
                plt.plot(t, impulse, label=model.muscles[m].label, color=colors[m], marker="o")
                if equilibrated_t_indices[m] is not None:
                    plt.axvline(x=t[equilibrated_t_indices[m]], color=colors[m], linestyle="--")
            plt.title(f"Integrated impulse difference")
            plt.xlabel("Time (s)")
            plt.ylabel("Integrated impulse\ndifference (N*s)")
            plt.grid(visible=True)
            plt.legend()

            plt.tight_layout()
    plt.show()


def plot_plotly(results, ratios, dts, model, colors, reference_index):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Convert matplotlib single-letter colors to plotly full color names
    mpl_to_plotly_colors = {
        "b": "blue",
        "g": "green",
        "r": "red",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
    }

    plotly_colors = [mpl_to_plotly_colors.get(color, color) for color in colors]
    colors = plotly_colors

    for ratio in ratios:
        for dt in dts:
            data = results[str(ratio)][str(dt)] if str(ratio) in results and str(dt) in results[str(ratio)] else None
            if data is None:
                continue

            t = np.array(data["t"])
            muscles_fiber_length = np.array(data["muscles_fiber_length"])
            tendon_length = np.array(data["tendon_length"])
            muscle_tendon_length = np.array(data["muscle_tendon_lengths"])
            muscles_fiber_velocity = np.array(data["muscles_fiber_velocity"])
            muscles_force = np.array(data["muscles_force"])
            equilibrated_t_indices = np.array(data["equilibrated_t_indices"])

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Muscle length and tendon length",
                    "Muscle force",
                    "Muscle fiber velocity",
                    "Integrated impulse difference",
                ),
            )

            for m in range(len(model.muscles)):
                # Plot muscle length and tendon length
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=muscles_fiber_length[:, m],
                        name=model.muscles[m].label + "mucle fiber",
                        mode="lines+markers",
                        marker=dict(color=colors[m]),
                        showlegend=True,
                        # legendgroup=model.muscles[m].label,
                        # legendgrouptitle_text=model.muscles[m].label,
                        fill="tozeroy",
                        # light red
                        fillcolor="rgba(255, 0, 0, 0.1)",
                    ),
                    row=1,
                    col=1,
                )
                # fill between muscle length and tendon length
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=tendon_length[:, m] + muscles_fiber_length[:, m],
                        name=model.muscles[m].label + " tendon",
                        mode="lines+markers",
                        marker=dict(color=colors[m], opacity=0.2),
                        line=dict(dash="dot"),
                        opacity=0.2,
                        showlegend=True,
                        legendgroup=model.muscles[m].label,
                        fill="tonexty",
                        # light brown
                        fillcolor="rgba(255, 165, 0, 0.1)",
                    ),
                    row=1,
                    col=1,
                )
                # total length
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=np.tile(muscle_tendon_length[m], (len(t), 1)).squeeze(),
                        name=model.muscles[m].label + " total",
                        mode="lines",
                        marker=dict(color="black"),
                        line=dict(color="black", dash="solid"),
                        showlegend=True,
                        # legendgroup=model.muscles[m].label,
                        # legendgrouptitle_text=model.muscles[m].label,
                    ),
                    row=1,
                    col=1,
                )

                # Plot muscle velocities
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=muscles_fiber_velocity[:, m],
                        name=model.muscles[m].label,
                        mode="lines+markers",
                        marker=dict(color=colors[m]),
                        showlegend=True,
                        legendgroup=model.muscles[m].label,
                        legendgrouptitle_text=model.muscles[m].label,
                    ),
                    row=2,
                    col=1,
                )
                if equilibrated_t_indices[m] is not None:
                    fig.add_vline(
                        x=t[equilibrated_t_indices[m]],
                        line_dash="dash",
                        line_color=colors[m],
                        row=2,
                        col=1,
                    )

                # Plot muscle forces
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=muscles_force[:, m],
                        name=model.muscles[m].label,
                        mode="lines+markers",
                        marker=dict(color=colors[m]),
                        showlegend=False,
                        legendgroup=model.muscles[m].label,
                    ),
                    row=1,
                    col=2,
                )
                if equilibrated_t_indices[m] is not None:
                    fig.add_vline(
                        x=t[equilibrated_t_indices[m]],
                        line_dash="dash",
                        line_color=colors[m],
                        row=1,
                        col=2,
                    )

                # Plot integrated impulse difference
                cum_diff_force = np.cumsum(muscles_force[:, m] - muscles_force[:, reference_index])
                impulse = np.zeros_like(muscles_force[:, m])
                impulse[1:] = (cum_diff_force[1:] + cum_diff_force[:-1]) * (t[1:] - t[:-1]) / 2
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=impulse,
                        name=model.muscles[m].label,
                        mode="lines+markers",
                        marker=dict(color=colors[m]),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )
                if equilibrated_t_indices[m] is not None:
                    fig.add_vline(
                        x=t[equilibrated_t_indices[m]],
                        line_dash="dash",
                        line_color=colors[m],
                        row=2,
                        col=2,
                    )

            fig.update_layout(
                title=f"Muscle fiber velocity and force for a ratio of {ratio} at dt = {dt}",
                height=900,
            )
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=2)
            fig.update_xaxes(title_text="Time (s)", row=1, col=2)
            fig.update_yaxes(title_text="Muscle fiber velocity (m/s)", row=2, col=1)
            fig.update_yaxes(title_text="Muscle force (N)", row=1, col=2)
            fig.update_yaxes(title_text="Integrated impulse difference (N*s)", row=2, col=2)

            fig.show()
