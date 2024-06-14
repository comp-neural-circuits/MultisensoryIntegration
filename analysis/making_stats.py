import numpy as np
from importlib import reload
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# defining things for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font',  size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes',  titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes',  labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend',  fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure',  titlesize=BIGGER_SIZE)  # fontsize of the figure title


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]

def making_stats(theta, correlation, path, W_v1, W_s1, w_thres_activation, N_v1):
    W_active_v1 = W_v1 > w_thres_activation
    W_active_s1 = W_s1 > w_thres_activation
    rf_vec_v1 = np.sum(W_active_v1, axis=1)
    rf_vec_s1 = np.sum(W_active_s1, axis=1)
    finite_rf_v1 = rf_vec_v1[np.where(rf_vec_v1>0)]
    finite_rf_s1 = rf_vec_s1[np.where(rf_vec_s1>0)]
    rf_size_v1 = np.mean(finite_rf_v1)
    rf_size_s1 = np.mean(finite_rf_s1)
    v1_active = list(np.where(rf_vec_v1>0))
    s1_active = list(np.where(rf_vec_s1>0))
    v1_inactive = list(np.where(rf_vec_v1==0))
    s1_inactive = list(np.where(rf_vec_s1==0))
    bimodal_neurons = common_elements(v1_active[0], s1_active[0])
    unimodal_v1 = common_elements(v1_active[0], s1_inactive[0])
    unimodal_s1 = common_elements(s1_active[0], v1_inactive[0])
    proportions = [100*len(unimodal_v1)/N_v1, 100*len(unimodal_s1)/N_v1, 100*len(bimodal_neurons)/N_v1]
    bar_names = ('Unimodal V1', 'Unimodal S1', 'Bimodal')
    y_pos = np.arange(len(bar_names))
    plt.bar(y_pos, proportions, color=('tab:red','tab:blue','tab:green'))
    plt.xticks(y_pos, bar_names)
    plt.ylim([0,100])
    plt.ylabel('Fraction of weights (%)')
    plt.savefig(path / 'proportions.pdf')
    plt.close()
    plt.subplots(1,2, figsize=(15,5))
    plt.subplot(1,2,1)
    plt.hist(rf_vec_v1[rf_vec_v1>0], bins=range(0,49 + 2, 2), color='tab:red', edgecolor='black', linewidth=1.0)
    plt.axvline(rf_size_v1, color='k', linestyle='dashed', linewidth=2)
    plt.xlabel('Receptive field size (V1)')
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    plt.hist(rf_vec_s1[rf_vec_s1>0], bins=range(0,49 + 2, 2), color='tab:blue', edgecolor='black', linewidth=1.0)
    plt.axvline(rf_size_s1, color='k', linestyle='dashed', linewidth=2)
    plt.xlabel('Receptive field size (S1)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(path / 'hist_rfs.pdf')
    plt.close()
    output_data = [theta, correlation, rf_size_v1, rf_size_s1, proportions[0], proportions[1], proportions[2]]
    np.savetxt(path / 'stats.txt', output_data, fmt="%.4f", delimiter=',')
