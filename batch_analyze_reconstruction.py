#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:19:44 2021

@author: kirchnerj@MPIBR
adjusted by JaeAnn Dwulet
"""
# %%
import numpy, os, matplotlib, random
import pylab as plt
import matplotlib.cm as cm
import seaborn as sns

from tools.biased_weights import biased_weights

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# defining things for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14.0
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
N = 50
simName = "monte_carlo500000tp"
#simName = "monte_carlov2"
allMats = []
allprms = []
for foldername in numpy.sort(os.listdir("simulations/" + simName + "/")):
    if foldername != '.DS_Store':
        for filename in os.listdir("simulations/" + simName + "/" + foldername):
            if filename.endswith(".pickle"):# and (numpy.random.rand() < 0.05):
                print(
                    os.path.join(
                        os.path.join("simulations/" + simName + "/", foldername), filename
                    )
                )
                time_vec_list, W_vec_v1, W_vec_s1, prms, mats, vec_v1, vec_s1, vec_rl = (
                    numpy.load(
                        os.path.join(
                            os.path.join("simulations/" + simName + "/", foldername),
                            filename,
                        ),
                        allow_pickle=True,
                    )
                )
                allMats.append(mats)
                allprms.append(prms)


# %%
def get_events(prms, mats, max_ms=60000):
    # Initializing the activities
    v1, s1, rl = (
        np.zeros([prms["N_v1"], 1]),
        np.zeros([prms["N_s1"], 1]),
        np.zeros([prms["N_rl"], 1]),
    )

    # generating shared and independent events
    total_ms = prms["total_ms"]

    p1 = np.random.rand(total_ms, 1) <= (1.0 / prms["L_p_v1"]) * (
        1 - prms["spatio_temp_corr"]
    )
    p2 = np.random.rand(total_ms, 1) <= (1.0 / prms["L_p_s1"]) * (
        1 - prms["spatio_temp_corr"]
    )
    p3 = np.random.rand(total_ms, 1) <= (2 / (prms["L_p_v1"] + prms["L_p_v1"])) * (
        prms["spatio_temp_corr"]
    )
    pall = p1 + p2 + p3

    # initialize duration counters and accumulators
    L_dur_v1_counter = round(np.random.normal(prms["L_dur_v1"], prms["L_dur_v1"] * 0.1))
    L_dur_s1_counter = round(np.random.normal(prms["L_dur_s1"], prms["L_dur_s1"] * 0.1))

    # main loop over time
    for tt in range(0, int(total_ms)):
        # generate local event
        if pall[tt] >= 1:
            # correlated events
            corrFlag = p3[tt] == 1
            v1flag = p1[tt] == 1

            # L-events in v1 and s1
            L_start_v1 = np.random.choice(prms["N_v1"], 1)[0]
            L_start_s1 = np.random.choice(prms["N_s1"], 1)[0]
            if corrFlag:
                L_start_s1 = L_start_v1

            L_len_v1 = random.randrange(
                int(prms["N_v1"] * prms["L_range_v1"][0]),
                int(prms["N_v1"] * prms["L_range_v1"][1] + 1),
            )
            L_len_s1 = random.randrange(
                int(prms["N_s1"] * prms["L_range_s1"][0]),
                int(prms["N_s1"] * prms["L_range_s1"][1] + 1),
            )

            eventID_v1 = np.mod(range(L_start_v1, L_start_v1 + L_len_v1), prms["N_v1"])
            eventID_s1 = np.mod(range(L_start_s1, L_start_s1 + L_len_s1), prms["N_s1"])

            if corrFlag:
                eventID_s1 = eventID_v1
                v1 = np.zeros([prms["N_v1"], 1])
                s1 = np.zeros([prms["N_s1"], 1])
                v1[eventID_v1, 0] = prms["L_amp_v1"]
                s1[eventID_s1, 0] = prms["L_amp_s1"]
                L_dur_v1_counter = round(
                    np.random.normal(prms["L_dur_v1"], prms["L_dur_v1"] * 0.1)
                )
                L_dur_s1_counter = L_dur_v1_counter
            elif v1flag:
                v1 = np.zeros([prms["N_v1"], 1])
                v1[eventID_v1, 0] = prms["L_amp_v1"]
                L_dur_v1_counter = round(
                    np.random.normal(prms["L_dur_v1"], prms["L_dur_v1"] * 0.1)
                )
            else:
                s1 = np.zeros([prms["N_s1"], 1])
                s1[eventID_s1, 0] = prms["L_amp_s1"]
                L_dur_s1_counter = round(
                    np.random.normal(prms["L_dur_s1"], prms["L_dur_s1"] * 0.1)
                )

        # updating the output
        rl = rl + (prms["dt"] / prms["tau_out"]) * (
            -rl + np.dot(mats["W_v1"], v1) + np.dot(mats["W_s1"], s1)
        )

        # this is where the old bug was
        L_dur_v1_counter -= 1
        L_dur_s1_counter -= 1

        if tt > max_ms:
            return
        if L_dur_v1_counter == 0:
            v1 = np.zeros([prms["N_v1"], 1])
        if L_dur_s1_counter == 0:
            s1 = np.zeros([prms["N_s1"], 1])
        if (v1.sum() == 0) and (s1.sum() == 0):
            continue
        if rl.sum() == 0:
            continue
        yield v1, s1, rl


# %%
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

all_v1_s1_r_2 = []
all_stats = []
max_mats = 1000
num_events = 500

N = 50
V1template = biased_weights(N, [0, 0], 1, 4)
S1template = biased_weights(N, [0, 0], 1, 4)

for mats, prms in zip(tqdm(allMats),allprms):
    acc_v1 = np.array([])
    acc_s1 = np.array([])
    acc_rl = np.array([])
    for v1, s1, rl in get_events(prms, mats):
        rl = torch.sigmoid(torch.from_numpy(rl) - torch.mean(torch.from_numpy(rl)))

        # Reshape rl, v1, and s1 to be 2D arrays for sklearn
        rl_np = rl.numpy().reshape(-1, 1)
        v1_np = v1.reshape(-1, 1)
        s1_np = s1.reshape(-1, 1)

        acc_v1 = np.hstack((acc_v1, v1_np)) if acc_v1.size else v1_np
        acc_s1 = np.hstack((acc_s1, s1_np)) if acc_s1.size else s1_np
        acc_rl = np.hstack((acc_rl, rl_np)) if acc_rl.size else rl_np

        # deduplicate acc_rl, and remove corresponding rows from acc_v1 and acc_s1
        acc_rl, idx = np.unique(acc_rl, axis=1, return_index=True)
        acc_v1 = acc_v1[:, idx]
        acc_s1 = acc_s1[:, idx]

        if acc_v1.shape[1] >= num_events:
            break
    if acc_v1.shape[1] < num_events:
        continue

    origin = np.vstack((acc_v1, acc_s1))

    # split data into training and test sets
    origin_train = origin[:, : int(num_events * 0.8)].T
    #origin_test = origin[:, int(num_events * 0.8) :].T

    rl_train = acc_rl[:, : int(num_events * 0.8)].T
    #rl_test = acc_rl[:, int(num_events * 0.8) :].T

    # Fit linear regression model to predict origin from rl
    model = LinearRegression()
    model.fit(rl_train, origin_train)

    # Predict on the test set
    origin_pred = model.predict(rl_train)

    v1_test = origin_train[:, :N]
    v1_pred = origin_pred[:, :N]

    s1_test = origin_train[:, N:]
    s1_pred = origin_pred[:, N:]

    v1_s1_r_2 = []

    for test, pred in [(v1_test, v1_pred), (s1_test, s1_pred)]:
        # Compute the residual sum of squares
        rss = np.sum((pred - test) ** 2)

        # Compute the total sum of squares
        tss = np.sum((test - np.mean(test)) ** 2)

        # Compute R^2 manually
        r_squared_manual = 1 - (rss / tss)

        v1_s1_r_2.append(r_squared_manual)

    cMat = np.corrcoef(mats["W_s1"].ravel(), mats["W_v1"].ravel())
    cMatV1 = np.corrcoef(mats["W_v1"].ravel(), V1template.ravel())
    cMatS1 = np.corrcoef(mats["W_s1"].ravel(), S1template.ravel())
    bimodal = ((mats["W_s1"] > 0.5).sum(axis=1) > 0) * (
        (mats["W_v1"] > 0.5).sum(axis=1) > 0
    )
    v1_connected = (mats["W_v1"] > 0.5).sum(axis=1) > 0
    s1_connected = (mats["W_s1"] > 0.5).sum(axis=1) > 0
    all_stats.append(
        {
            "corr_thres": prms["corr_thres"],
            "bias_s1": prms["bias_s1"],
            "spatio_temp_corr": prms["spatio_temp_corr"],
            "cMat": cMat[0, 1],
            "cMatV1": cMatV1[0, 1],
            "cMatS1": cMatS1[0, 1],
            "bimodal": bimodal.mean(),
            "v1_connected": v1_connected.mean(),
            "s1_connected": s1_connected.mean(),
            "v1_s1_r_2": v1_s1_r_2,
            "v1_r2": v1_s1_r_2[0],
            "s1_r2": v1_s1_r_2[1],
            
        }
    )

    all_v1_s1_r_2.append(v1_s1_r_2)
    if len(all_v1_s1_r_2) >= max_mats:
        break

# Plotting after the loop
# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

x_combined = [stat["v1_connected"] for stat in all_stats] + [stat["s1_connected"] for stat in all_stats]
y_combined = [stat["v1_r2"] for stat in all_stats] + [stat["s1_r2"] for stat in all_stats]
hue_combined = [stat["cMatV1"] for stat in all_stats] + [stat["cMatS1"] for stat in all_stats]

# Plot using seaborn for the first subplot
scatter1 = sns.scatterplot(
    x=x_combined,
    y=y_combined,
    hue=hue_combined,
    palette="OrRd",
    legend=False,
    ax=axs[0],
    s=100,
    edgecolor="grey"  # Outline color
)

scatter1.set_xlabel("Connected (%)", fontsize=20, fontname="Arial", labelpad=-15)
scatter1.set_ylabel("V1/S1 recon. ($R^2$)", fontsize=20, fontname="Arial", labelpad=-10)
scatter1.set_xlim([0, 1])
scatter1.set_ylim([0, 1])
scatter1.set_xticks([0, 1])
scatter1.set_yticks([0, 1])
scatter1.tick_params(axis='both', which='major', labelsize=20)

# Adding horizontal color bar inside the first subplot
norm1 = plt.Normalize(0, 1)
sm1 = plt.cm.ScalarMappable(cmap="OrRd", norm=norm1)
sm1.set_array([])
#axins1 = inset_axes(axs[0], width="50%", height="10%", loc='lower right', borderpad=0)
#colorbar1 = fig.colorbar(sm1, cax=axins1, orientation="horizontal")
#colorbar1.set_label('V1/S1 topography', fontsize=12, fontname="Arial")
##colorbar1.ax.tick_params(labelsize=10)

axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Plot using seaborn for the second subplot
v1_r2, s1_r2 = zip(*all_v1_s1_r_2)
bimodal_values = [stat["bimodal"] * 100 for stat in all_stats]  # Convert to percentage
scatter2 = sns.scatterplot(x=v1_r2, y=s1_r2, hue=bimodal_values, palette="PRGn", ax=axs[1], s=100, legend=False, edgecolor="grey")
scatter2.set_xlabel("V1 recon. ($R^2$)", fontsize=20, fontname="Arial", labelpad=-15)
scatter2.set_ylabel("S1 recon. ($R^2$)", fontsize=20, fontname="Arial",  labelpad=-10)
scatter2.set_xlim([0, 1])
scatter2.set_ylim([0, 1])
scatter2.set_xticks([0, 1])
scatter2.set_yticks([0, 1])
scatter2.tick_params(axis='both', which='major', labelsize=20)

# Adding horizontal color bar inside the second subplot
norm2 = plt.Normalize(0, 100)
sm2 = plt.cm.ScalarMappable(cmap="PRGn", norm=norm2)
sm2.set_array([])
#axins2 = inset_axes(axs[1], width="50%", height="5%", loc='lower center', bbox_to_anchor=(0.5, -0.25, 1, 1), bbox_transform=axs[1].transAxes, borderpad=0)
#colorbar2 = fig.colorbar(sm2, cax=axins2, orientation="horizontal")
#colorbar2.set_label('% bimodal', fontsize=12, fontname="Arial")
#colorbar2.ax.tick_params(labelsize=10)

axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
# plt.tight_layout()
# plt.show()

# Save all the data from the plots into a CSV file
data1 = {
    'x': [stat["v1_connected"] for stat in all_stats] + [stat["s1_connected"] for stat in all_stats],
    'y': [stat["v1_r2"] for stat in all_stats] + [stat["s1_r2"] for stat in all_stats],
    'hue': [stat["cMatV1"] for stat in all_stats] + [stat["cMatS1"] for stat in all_stats]
}

df1 = pd.DataFrame(data1)

# Save the DataFrame to a CSV file
csv_file_path = 'scatter_rsquared_connected_topo.csv'
df1.to_csv(csv_file_path, index=False)

print(f'Data saved to {csv_file_path}')

data2 = {
    'x': v1_r2,
    'y': s1_r2,
    'hue': bimodal_values
}

df2 = pd.DataFrame(data2)

# Save the DataFrame to a CSV file
csv_file_path2 = 'scatter_s1r2_v1r2_bimodal.csv'
df2.to_csv(csv_file_path2, index=False)

print(f'Data saved to {csv_file_path2}')

# %%
#fig, ax = plt.subplots(figsize=(5, 5))
#ax.scatter(*zip(*all_v1_s1_r_2))
#ax.set_xlabel("V1 R^2")
#ax.set_ylabel("S1 R^2")
#ax.set_xlim([0, 1])
#ax.set_ylim([0, 1])
#plt.show()

# %%

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

x_combined = [stat["v1_connected"] for stat in all_stats] + [stat["s1_connected"] for stat in all_stats]
y_combined = [stat["v1_r2"] for stat in all_stats] + [stat["s1_r2"] for stat in all_stats]
corr_values = [stat["spatio_temp_corr"] for stat in all_stats]

# Plot using seaborn for the first subplot
scatter1 = sns.scatterplot(
    x=x_combined,
    y=y_combined,
    hue=corr_values+corr_values,
    palette="bwr",
    legend=False,
    ax=axs[0],
    s=100,
    edgecolor="grey"  # Outline color
)

scatter1.set_xlabel("Connected (%)", fontsize=20, fontname="Arial")
scatter1.set_ylabel("V1/S1 reconstruction ($R^2$)", fontsize=20, fontname="Arial")
scatter1.set_xlim([0, 1])
scatter1.set_ylim([0, 1])
norm1 = plt.Normalize(0, 1)
sm1 = plt.cm.ScalarMappable(cmap="bwr", norm=norm1)
sm1.set_array([])
colorbar1 = fig.colorbar(sm1, ax=axs[0])
colorbar1.set_label('Correlation', fontsize=20, fontname="Arial")

colorbar1.ax.yaxis.label.set_rotation(270)
colorbar1.ax.yaxis.label.set_verticalalignment('center')
colorbar1.ax.tick_params(labelsize=14)


# Plot using seaborn for the second subplot
v1_r2, s1_r2 = zip(*all_v1_s1_r_2)
scatter2 = sns.scatterplot(x=v1_r2, y=s1_r2, hue=corr_values, palette="bwr", ax=axs[1],s=100, legend=False, edgecolor="grey")
scatter2.set_xlabel("V1 reconstruction ($R^2$)", fontsize=20, fontname="Arial")
scatter2.set_ylabel("S1 reconstruction ($R^2$)", fontsize=20, fontname="Arial")
scatter2.set_xlim([0, 1])
scatter2.set_ylim([0, 1])
norm2 = plt.Normalize(0, 1)
sm2 = plt.cm.ScalarMappable(cmap="bwr", norm=norm2)
sm2.set_array([])
colorbar2 = fig.colorbar(sm2, ax=axs[1])
colorbar2.set_label('Correlation', fontsize=20, fontname="Arial")

colorbar2.ax.yaxis.label.set_rotation(270)
colorbar2.ax.yaxis.label.set_verticalalignment('center')

colorbar2.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

# %%
