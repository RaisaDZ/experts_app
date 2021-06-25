import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import SessionState
from anomaly_delays.helper_functions import calc_cum_avg_loss, read_nab
from anomaly_delays.main_functions import share_delays

# state = SessionState.get(position=0)

# The slider needs to come after the button, to make sure the first increment
# works correctly. So we create a placeholder for it here first, and fill it in
# later.
# widget = st.empty()

# if st.button('Increment position'):
#    state.position += 1

# state.position = widget.slider('Position', 0, 100, state.position)

ANOMALOUS_DICT = {0: "No", 1: "Yes"}

st.title("Visualisation of the algorithms' predictions, losses, and weights")

folders = [
    s
    for s in os.listdir("NAB/results/numenta")
    if s.startswith("real") or s.startswith("artificial")
]

folder_option = st.selectbox("folder name", folders)

for m, folder_name in enumerate(folders):
    files = [
        i.replace("numenta", "")
        for i in os.listdir(
            os.path.join("NAB/results/numenta", f"{folder_option}")
        )
    ]

file_option = st.selectbox("file name", files)

experts = [
    "knncad",
    "numentaTM",
    "twitterADVec",
    "skyline",
    "earthgeckoSkyline",
    "numenta",
    "bayesChangePt",
    "null",
    "expose",
    "relativeEntropy",
    "htmjava",
    "randomCutForest",
    "random",
    "contextOSE",
    "windowedGaussian",
]

experts_option = st.multiselect(
    "Select at least two experts:",
    experts,
    default=["skyline", "randomCutForest"],
)
share_option = st.selectbox("Select share type:", ["Fixed", "Variable"])
alpha_option = st.slider(
    r"alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.05
)
delay_option = st.slider("delay", min_value=1, max_value=100, value=1, step=1)

if share_option == "Fixed":
    loss_type = "Log-loss"
else:
    loss_type = "Square loss"

dt = read_nab(experts_option, folder_option, file_option)
score_experts = np.array(dt.filter(regex="^score", axis=1))
target = dt["label"].values

score_AA, loss_AA, loss_experts, weights_experts = share_delays(
    target,
    score_experts,
    share_type=share_option,
    alpha=alpha_option,
    delays=delay_option,
)

state = SessionState.get(position=2)
# The slider needs to come after the button, to make sure the first increment
# works correctly. So we create a placeholder for it here first, and fill it in
# later.

if st.button("Increment time") and state.position < len(score_AA) - 1:
    state.position += 1

widget = st.empty()


# time_0, time = st.slider('time', min_value=0, max_value=len(score_AA)-1, value=[0, len(score_AA)-1])
# time = st.slider('time', min_value=0, max_value=len(score_AA)-1, time.position)
# time.position = st.slider('Position', 0, 100, time.position)
# time_0, time = st.slider('time', min_value=0, max_value=len(score_AA)-1, value=[0, len(score_AA)-1])
# form = st.form(key='my-form')
# submit = form.form_submit_button('Increase time step')

# Experts predictions

if state.position >= len(score_AA) - 1:
    state.position = len(score_AA) - 1
state.position = widget.slider("Time", 0, len(score_AA) - 1, state.position)
time = state.position

st.header(
    "Fixed-share and Variable-share mix predictions of different models called experts"
)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
for i, _ in enumerate(experts_option):
    axs.plot(
        list(np.arange(0, time + 1)),
        score_experts.T[i][: time + 1],
        linewidth=6,
        label=f"{experts_option[i]}",
    )
axs.plot(
    list(np.arange(0, time + 1)),
    target[: time + 1],
    label="anomalies",
    linewidth=6,
)
axs.legend(loc="upper right", bbox_to_anchor=(1.65, 1), fontsize=26)
axs.set_xlabel("Time", fontsize=36)
axs.set_ylabel("Probability", fontsize=36)
axs.set_ylim([-0.05, 1.05])
axs.xaxis.set_tick_params(labelsize=26)
axs.yaxis.set_tick_params(labelsize=26)
plt.rcParams.update({"font.size": 36})
fig.suptitle("Experts predictions", fontsize=30)
st.write(fig)

st.subheader(f"Experts predictions at current step _{time}_:")
for i, _ in enumerate(experts_option):
    st.subheader(
        f"{experts_option[i]}: _{round(score_experts.T[i][time], 4)}_"
    )
st.subheader(f"Is the point anomalous? _{ANOMALOUS_DICT[target[time]]}_")


# Experts' losses
st.header(
    "The methods are based on real-time assessment of algorithms’ performance (losses)"
)
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
for i, _ in enumerate(experts_option):
    axs.plot(
        list(np.arange(0, time + 1)),
        loss_experts.T[i][: time + 1],
        linewidth=6,
        label=f"{experts_option[i]}",
    )
axs.legend(loc="upper right", bbox_to_anchor=(1.65, 1), fontsize=26)
axs.set_xlabel("Time", fontsize=36)
axs.set_ylabel("Experts losses", fontsize=36)
axs.set_ylim(
    [
        0.9 * min(loss_experts[: time + 1].min(axis=1)) - 0.1,
        1.1 * max(loss_experts[: time + 1].max(axis=1)) + 0.1,
    ]
)
axs.xaxis.set_tick_params(labelsize=26)
axs.yaxis.set_tick_params(labelsize=26)
plt.rcParams.update({"font.size": 36})
fig.suptitle(f"{loss_type}", fontsize=26)
st.write(fig)
st.subheader(f"Experts losses at current step _{time}_:")
for i, _ in enumerate(experts_option):
    st.subheader(f"{experts_option[i]}: _{round(loss_experts.T[i][time], 4)}_")


# Experts weights
st.header("Higher experts' losses result in lower weights")

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
for i, _ in enumerate(experts_option):
    plt.plot(
        list(np.arange(0, time + 1)),
        weights_experts.T[i][: time + 1],
        linewidth=6,
        label=f"{experts_option[i]}",
    )
axs.legend(loc="upper right", bbox_to_anchor=(1.65, 1), fontsize=26)
axs.set_xlabel("Time", fontsize=36)
axs.set_ylabel("Experts weights", fontsize=36)
axs.set_ylim(
    [
        0.9 * min(weights_experts[: time + 1].min(axis=1)),
        1.1 * max(weights_experts[: time + 1].max(axis=1)),
    ]
)
axs.xaxis.set_tick_params(labelsize=26)
axs.yaxis.set_tick_params(labelsize=26)
plt.rcParams.update({"font.size": 36})
fig.suptitle(
    fr"{share_option}-share, $\alpha = ${alpha_option}, delay = {delay_option}",
    fontsize=26,
)
st.write(fig)
st.subheader(f"Experts weights at current step _{time}_:")
for i, _ in enumerate(experts_option):
    st.subheader(
        f"{experts_option[i]}: _{round(weights_experts.T[i][time], 4)}_"
    )


# Algorithm's predictions
st.header(
    "The resulting algorithm aggregates experts' predictions based on their current weights"
)

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.plot(
    list(np.arange(0, time + 1)),
    score_AA[: time + 1],
    label="predictions",
    linewidth=6,
)
axs.plot(
    list(np.arange(0, time + 1)),
    target[: time + 1],
    label="anomalies",
    linewidth=6,
)
axs.legend(loc="upper right", bbox_to_anchor=(1.65, 1), fontsize=26)
axs.set_xlabel("Time", fontsize=36)
axs.set_ylabel("Probability", fontsize=36)
axs.set_ylim([-0.05, 1.05])
axs.xaxis.set_tick_params(labelsize=26)
axs.yaxis.set_tick_params(labelsize=26)
plt.rcParams.update({"font.size": 36})
fig.suptitle(
    fr"{share_option}-share, $\alpha = ${alpha_option}, delay = {delay_option}",
    fontsize=26,
)
st.write(fig)
st.subheader(
    f"Prediction of {share_option}-share at current step {time} is _{round(score_AA[time], 4)}_"
)
st.subheader(f"Is the point anomalous? _{ANOMALOUS_DICT[target[time]]}_")
