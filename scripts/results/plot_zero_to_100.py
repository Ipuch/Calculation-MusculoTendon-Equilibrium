import matplotlib.pyplot as plt
import numpy as np

force_defect_data = {
    0.1: {0.0001: 0.0034, 0.0005: 0.0035, 0.0010: None, 0.0050: None, 0.0100: None},
    0.2: {0.0001: 0.0006, 0.0005: None, 0.0010: None, 0.0050: None, 0.0100: None},
    0.5: {0.0001: 0.0100, 0.0005: 0.0100, 0.0010: 0.0100, 0.0050: None, 0.0100: None},
    0.8: {0.0001: 0.0123, 0.0005: 0.0125, 0.0010: 0.0130, 0.0050: None, 0.0100: None},
    1.0: {0.0001: 0.0139, 0.0005: 0.0140, 0.0010: 0.0140, 0.0050: None, 0.0100: None},
    1.5: {0.0001: 0.0158, 0.0005: 0.0160, 0.0010: 0.0160, 0.0050: 0.0250, 0.0100: None},
    2.0: {0.0001: 0.0168, 0.0005: 0.0170, 0.0010: 0.0170, 0.0050: 0.0200, 0.0100: None},
    5.0: {0.0001: 0.0178, 0.0005: 0.0180, 0.0010: 0.0180, 0.0050: 0.0200, 0.0100: None},
    10.0: {0.0001: 0.0164, 0.0005: 0.0165, 0.0010: 0.0170, 0.0050: 0.0200, 0.0100: None},
}

velocity_defect_data = {
    0.1: {0.0001: 0.0034, 0.0005: 0.0035, 0.0010: None, 0.0050: None, 0.0100: 0.0100},
    0.2: {0.0001: 0.0006, 0.0005: None, 0.0010: 0.0010, 0.0050: 0.0050, 0.0100: 0.0100},
    0.5: {0.0001: 0.0100, 0.0005: 0.0100, 0.0010: 0.0100, 0.0050: None, 0.0100: None},
    0.8: {0.0001: 0.0123, 0.0005: 0.0125, 0.0010: 0.0130, 0.0050: None, 0.0100: None},
    1.0: {0.0001: 0.0139, 0.0005: 0.0140, 0.0010: 0.0140, 0.0050: None, 0.0100: None},
    1.5: {0.0001: 0.0158, 0.0005: 0.0160, 0.0010: 0.0160, 0.0050: 0.0250, 0.0100: None},
    2.0: {0.0001: 0.0168, 0.0005: 0.0170, 0.0010: 0.0170, 0.0050: 0.0200, 0.0100: None},
    5.0: {0.0001: 0.0178, 0.0005: 0.0180, 0.0010: 0.0180, 0.0050: 0.0200, 0.0100: None},
    10.0: {0.0001: 0.0164, 0.0005: 0.0165, 0.0010: 0.0170, 0.0050: 0.0200, 0.0100: None},
}

linear_approx_data = {
    0.1: {0.0001: 0.0035, 0.0005: 0.0040, 0.0010: None, 0.0050: None, 0.0100: None},
    0.2: {0.0001: 0.0006, 0.0005: None, 0.0010: None, 0.0050: None, 0.0100: None},
    0.5: {0.0001: 0.0101, 0.0005: 0.0105, 0.0010: 0.0110, 0.0050: None, 0.0100: None},
    0.8: {0.0001: 0.0124, 0.0005: 0.0130, 0.0010: 0.0130, 0.0050: None, 0.0100: None},
    1.0: {0.0001: 0.0139, 0.0005: 0.0145, 0.0010: 0.0150, 0.0050: None, 0.0100: None},
    1.5: {0.0001: 0.0159, 0.0005: 0.0165, 0.0010: 0.0170, 0.0050: None, 0.0100: None},
    2.0: {0.0001: 0.0169, 0.0005: 0.0175, 0.0010: 0.0180, 0.0050: 0.0300, 0.0100: None},
    5.0: {0.0001: 0.0178, 0.0005: 0.0185, 0.0010: 0.0190, 0.0050: 0.0250, 0.0100: None},
    10.0: {0.0001: 0.0165, 0.0005: 0.0170, 0.0010: 0.0170, 0.0050: 0.0250, 0.0100: None},
}

quadratic_data = {
    0.1: {0.0001: 0.0035, 0.0005: 0.0040, 0.0010: None, 0.0050: None, 0.0100: None},
    0.2: {0.0001: 0.0006, 0.0005: None, 0.0010: None, 0.0050: None, 0.0100: None},
    0.5: {0.0001: 0.0101, 0.0005: 0.0105, 0.0010: 0.0110, 0.0050: None, 0.0100: None},
    0.8: {0.0001: 0.0124, 0.0005: 0.0130, 0.0010: 0.0140, 0.0050: None, 0.0100: None},
    1.0: {0.0001: 0.0140, 0.0005: 0.0145, 0.0010: 0.0150, 0.0050: 0.0300, 0.0100: None},
    1.5: {0.0001: 0.0159, 0.0005: 0.0165, 0.0010: 0.0170, 0.0050: 0.0250, 0.0100: None},
    2.0: {0.0001: 0.0169, 0.0005: 0.0175, 0.0010: 0.0180, 0.0050: 0.0250, 0.0100: None},
    5.0: {0.0001: 0.0178, 0.0005: 0.0185, 0.0010: 0.0190, 0.0050: 0.0250, 0.0100: None},
    10.0: {0.0001: 0.0165, 0.0005: 0.0170, 0.0010: 0.0170, 0.0050: 0.0200, 0.0100: None},
}


# Bundle them all together so we can loop
all_data = {
    "Force defect": force_defect_data,
    "Velocity defect": velocity_defect_data,
    "Linear approximation": linear_approx_data,
    "Quadratic approximation": quadratic_data,
}

# 2) Define the time steps as a sorted list (just for consistent plotting)
time_steps = [0.0001, 0.0005, 0.0010, 0.0050, 0.0100]

# 3) Create subplots: one per muscle type
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (muscle_type, data_dict) in zip(axes, all_data.items()):
    # data_dict is something like: { ratio: {dt: equilibrium_time, ...}, ratio2: {...} }

    # For a given muscle type, let’s parse out ratio values and times for each dt
    # We want lines: each line is for a single dt
    for dt in time_steps:
        # Gather (ratio, time) pairs for this dt
        ratio_vals = []
        time_vals = []
        for ratio, dt_dict in data_dict.items():
            t_eq = dt_dict.get(dt, None)
            if t_eq is not None:  # skip missing
                ratio_vals.append(ratio)
                time_vals.append(t_eq * 1000)

        # Plot if we have at least one valid time
        if ratio_vals:
            ax.plot(ratio_vals, time_vals, marker="o", label=f"dt={dt * 1000}")

    ax.set_title(muscle_type)
    # log scale on x
    ax.set_xscale("log")
    # grid
    ax.grid(True)
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel("Time to equilibrium (ms)")
    if ax.get_subplotspec().colspan.start == 1 and ax.get_subplotspec().rowspan.start == 0:
        ax.legend(loc="upper center", ncol=3)
    if ax.get_subplotspec().rowspan.start == 1:
        ax.set_xlabel("Slack length / Optimal length")

plt.tight_layout()
#  save as svg
plt.savefig("equilibrium_time.svg")
plt.show()
