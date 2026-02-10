# %% [markdown]
# # Running - Study Variation of dFoF
# vdFoF in rest and active

# %% [markdown]
### Load packages and define constants:
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

sys.path += ['../']
from utils_.General_overview_episodes import compute_high_arousal_cond
import utils_.my_math as my_math

import scipy.stats as stats_func
import seaborn as sns

running_speed_threshold = 0.5  #cm/s
pre_stim = 1

#%%
def get_variation_dFoF(episodes, cond=None, pre_stim=pre_stim):  #TO CHANGE - automatize boundaries
    
    time_epi = int(episodes.time_duration[0])
    time_epi_p = time_epi * 1000
    ini_p = pre_stim * 1000
    inter_duration_p = int(time_epi_p/5)

    ini_val1 = ini_p - inter_duration_p 
    ini_val2 = ini_p

    final_val1 = ini_p + time_epi_p - inter_duration_p
    final_val2 = ini_p + time_epi_p

    if cond is not None: 
        episodes = episodes.dFoF[np.asarray(cond), :, :]
        ini_val   = episodes[:, :, ini_val1: ini_val2].mean(axis=0).mean(axis=1)
        final_val = episodes[:, :, final_val1: final_val2].mean(axis=0).mean(axis=1)

    else:
        ini_val   = episodes.dFoF[:, :, ini_val1: ini_val2].mean(axis=0).mean(axis=1)
        final_val = episodes.dFoF[:, :, final_val1:final_val2].mean(axis=0).mean(axis=1)
    
    diff = final_val - ini_val

    return diff

def get_vals_plot(episodes, cond):

    traces_act = episodes.dFoF[cond]
    traces_rest = episodes.dFoF[~cond]

    diffs_act = get_variation_dFoF(episodes, cond=cond, pre_stim=pre_stim)
    diffs_rest = get_variation_dFoF(episodes, cond=~cond, pre_stim=pre_stim)

    return traces_act, traces_rest, diffs_act, diffs_rest

def get_stats(all_diffs_act, all_diffs_rest):
    
    t_stats, p_val = stats_func.ttest_ind(all_diffs_act, all_diffs_rest, nan_policy='omit')
    significance = 'ns'                  
    if p_val==np.nan or p_val>0.05:
        significance = 'ns'  # Default is "not significant"
    elif p_val < 0.001:
        significance = '***'
    elif p_val < 0.01:
        significance = '**'
    elif p_val < 0.05:
        significance = '*'

    return t_stats, p_val, significance

def plot_boxplot(boxplot_dict):
    
    fig, AX = plt.subplots(1, 1, figsize=boxplot_dict["figsize"])

    x = np.arange(len(boxplot_dict["data"]))
    bp = AX.boxplot(boxplot_dict["data"], 
                    positions=x, 
                    sym='', 
                    patch_artist=True,
                    widths=0.6 )
    AX.set_xticks(x)
    AX.set_xticklabels(boxplot_dict["labels"])
    AX.set_xlabel("ROIs considered")
    AX.set_ylabel(boxplot_dict["y_label"])
    

    for patch, color in zip(bp['boxes'], boxplot_dict["colors"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    stats = my_math.calc_stats(boxplot_dict["title"], *boxplot_dict["data"], debug=False)
    my_math.plot_stats(AX, n_groups=len(boxplot_dict["data"]), stats=stats, y_pos1=0)

    return 0

def plot_dFoF(diffs_act, diffs_rest, protocol, filename, metric):

    cols = 2  # Number of columns per row
    rows = 1  # Compute the required number of rows
    fig, AX = plt.subplots(rows, cols, figsize=(7, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    #############################################################################################################
    #############################################################################################################
    #barplot all rois
    if metric =="roi":
        AX[0].set_title(f"File {filename}\n Protocol {protocol},\n average ROIs ( #{len(diffs_act)})")
    elif metric =="sessions":
        AX[0].set_title(f"File {filename}\n Protocol {protocol},\n average sessions ( #{len(diffs_act)})")
    
    bar_width = 0.4
    x = np.arange(2)  
    means = [np.nanmean(diffs_act), np.nanmean(diffs_rest)]
    AX[0].bar(x, means, width=bar_width, color=['orangered', 'grey'], edgecolor='black')
    
    x_act = np.full_like(diffs_act, x[0])
    x_rest = np.full_like(diffs_act, x[1])

    if metric == 'roi': 
        jitter_strength = 0.2  # Adjust for more/less jitter
        x_act_jitter = x_act + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_act))
        x_rest_jitter = x_rest + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_rest))

        AX[0].scatter(x_act_jitter, diffs_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
        AX[0].scatter(x_rest_jitter, diffs_rest, color='black', zorder=4, label="Resting", alpha=0.7)
    
    elif metric == "sessions":
        AX[0].scatter(x_act, diffs_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
        AX[0].scatter(x_rest, diffs_rest, color='black', zorder=4, label="Resting", alpha=0.7)
        for i in range(len(diffs_act)):
            color = 'coral' if diffs_rest[i] < diffs_act[i] else 'darkcyan'  # Blue for decrease, Red for increase
            AX[0].plot([x_act[i], x_rest[i]], [diffs_act[i], diffs_rest[i]], color=color, alpha=0.7, lw=1.5)


    AX[0].set_xticks(x, ['Active', 'Resting'])
    AX[0].set_xlabel("Behavioral state")
    AX[0].set_ylabel("Variation of dFoF")
    #AX[1].set_title(f"all ROIs for 1 file \n n ROIs = {len(diffs_act)}")

    t_stats, p_val, significance = get_stats(diffs_act, diffs_rest)
    AX[0].plot([x[0], x[1]], [np.max([means[0], means[1]]) + 5] * 2, color='black', lw=0.8)  # Line above bars
    AX[0].plot([x[0], x[0]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[0].plot([x[1], x[1]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[0].text(np.mean(x), np.max([means[0], means[1]]) + 5.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[1].set_ylim([-0.5,9])
    
    # Annotate each bar with its mean value
    for i in range(2):
        AX[0].text(i, np.max(means) + 3, f"mean {means[i]:.3f}", ha='center', fontsize=6)
    
    print("ALL ROIs for 1 file")
    print("number of ROIs :", len(diffs_act))
    print(f"active mean : {means[0]:.3f}, resting mean : {means[1]:.3f}")
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")

    
    #############################################################################################################
    #violing all rois for 1 file
    
    d = {'active': diffs_act, 'resting': diffs_rest}
    df = pd.DataFrame(data=d)
    df_melted = df.melt(var_name="Behavioral state", value_name="Variation of dFoF")
    sns.violinplot(data=df_melted, 
                   x="Behavioral state",
                   hue="Behavioral state", 
                   y="Variation of dFoF", 
                   inner="quart", 
                   palette={"active": "orangered", "resting": "grey"}, 
                   ax=AX[1], 
                   legend=False)
    
    t_stats, p_val, significance = get_stats(diffs_act, diffs_rest)
    AX[1].plot([x[0]+0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 3] * 2, color='black', lw=0.8)  # Line above bars
    AX[1].plot([x[0]+0.1, x[0]+0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[1].plot([x[1]-0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[1].text(np.mean(x), np.max([means[0], means[1]]) + 3.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[2].set_title(f"all ROIs for 1 file \n n ROIs = {len(diffs_act)}")
    #AX[2].set_ylim([-0.5,9])
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")
    
    # Calculate mean values for labels
    means_ = df_melted.groupby("Behavioral state")["Variation of dFoF"].mean()
    
    # Annotate each bar with its mean value
    for i, mean in enumerate(means_):
        AX[1].text(i, np.max(means_)-0.1, f"mean {mean:.3f}", ha='center', fontsize=6)

    min_plot = np.min(diffs_act)
    max_plot = np.max(diffs_act)
    AX[0].set_ylim([-4,10])
    AX[1].set_ylim([-4,10])
    
    return 0

def plot_trace_vdFoF(traces_act, traces_rest, aligned=False):

    fig, ax = plt.subplots(1,1, figsize=(3, 3))

    mean_act = np.nanmean(traces_act, axis=(0))
    sem_act  = np.nanstd(traces_act, axis=(0)) / np.sqrt(np.sum(~np.isnan(traces_act), axis=(0)))

    mean_rest = np.nanmean(traces_rest, axis=(0))
    sem_rest  = np.nanstd(traces_rest, axis=(0)) / np.sqrt(np.sum(~np.isnan(traces_rest), axis=(0)))

    if aligned: 
        ini_val1 = 600 #fix
        ini_val2 = 1000 #fix

        bsl_act = mean_act[ini_val1 : ini_val2].mean(axis=0)
        print(bsl_act)
        mean_act = mean_act - bsl_act

        bsl_rest = mean_rest[ini_val1 : ini_val2].mean(axis=0)
        print(bsl_rest)
        mean_rest = mean_rest - bsl_rest


    # Plot traces +- SEM
    ax.plot(episodes.t, mean_act, color="greenyellow", label="Active")
    ax.plot(episodes.t, mean_rest, color="darkolivegreen", label="Rest")
    ax.fill_between(episodes.t, mean_act - sem_act, mean_act + sem_act,
                    color="greenyellow", alpha=0.2)
    ax.fill_between(episodes.t, mean_rest - sem_rest, mean_rest + sem_rest,
                    color="darkolivegreen", alpha=0.2)

    # Formatting
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dFoF")
    ax.set_xticks(np.arange(0, episodes.t.max() + 1, 1))
    ax.fill_between(np.array([0, episodes.time_duration[0]]), y1=1, y2=-1,
                    color="grey", alpha=0.25)
    
    ax.fill_between(np.array([-episodes.time_duration[0]/5, 0]), y1=0.8, y2=-0.8,
                    color="orange", alpha=0.25)
    ax.fill_between(np.array([episodes.time_duration[0] - episodes.time_duration[0]/5,
                              episodes.time_duration[0]]), y1=0.8, y2=-0.8,
                    color="orange", alpha=0.25)
    
    ax.axhline(y=0, linewidth=0.5, linestyle='--')

    return 0

def calc_diff_baseline( trace_act, trace_rest):

    time_epi = int(episodes.time_duration[0])
    time_epi_p = time_epi * 1000
    ini_p = pre_stim * 1000
    inter_duration_p = int(time_epi_p/5)

    ini_val1 = ini_p - inter_duration_p 
    ini_val2 = ini_p

    ini_val_act  = np.nanmean(trace_act[ini_val1: ini_val2])  
    ini_val_rest = np.nanmean(trace_rest[ini_val1: ini_val2])

    diff_baseline = np.nan 

    if ~np.isnan(ini_val_act) and ~np.isnan(ini_val_rest): #why are there nans!!!
        diff_baseline = ini_val_act - ini_val_rest
        
    return diff_baseline

def calc_responsiveness(ep, nROIs):
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):
        t0 = max([0, ep.time_duration[0]-1.5])
        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=0.05,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=list(ep.varied_parameters.keys()))
        

        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])
  
    resp_cond = np.array(session_summary['significant'])                  
    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])
 
    print(f"{sum(resp_cond)} significant ROI ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) out of {len(session_summary['significant'])} ROIs")
 
    return resp_cond, pos_cond, neg_cond

def get_diffs_baseline(traces_act_s, traces_rest_s):
    temp_act = [np.nanmean(traces_act_s[i], axis=0 ) for i in range(len(traces_act_s))]
    temp_act = [arr for arr in temp_act if not np.isnan(arr).any()]
    flattened_act = [row for arr in temp_act for row in arr]
    
    temp_rest = [np.nanmean(traces_rest_s[i], axis=0 ) for i in range(len(traces_rest_s))]
    temp_rest = [arr for arr in temp_rest if not np.isnan(arr).any()]
    flattened_rest = [row for arr in temp_rest for row in arr]

    time_epi = int(episodes.time_duration[0])
    time_epi_p = time_epi * 1000
    ini_p = pre_stim * 1000
    inter_duration_p = int(time_epi_p/5)

    ini_val1 = ini_p - inter_duration_p 
    ini_val2 = ini_p

    diffs_baseline = []

    ini_val_act  = [np.nanmean(flattened_act[i][ini_val1: ini_val2])  for i in range(len(flattened_act))]
    ini_val_rest = [np.nanmean(flattened_rest[i][ini_val1: ini_val2]) for i in range(len(flattened_rest))]

    for i in range(len(flattened_act)):
        if ~np.isnan(ini_val_act[i]) and ~np.isnan(ini_val_rest[i]): #why are there nans!!!
            temp = ini_val_act[i] - ini_val_rest[i]
            diffs_baseline.append(temp)

    return diffs_baseline
###################################################################################################################
#%% Load Data

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]


dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

#%% ######################################################################################
######################################## TEST 1 FILE #####################################
##########################################################################################
##################################
#protocol = "Natural-Images-4-repeats"
protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
#protocol = "static-patch"
#protocol = "looming-stim"

index = 1
filename = SESSIONS['files'][index]
filename_ = os.path.basename(filename)
data = Data(filename,
            verbose=False)
data.build_dFoF(**dFoF_options, verbose=False)
data.build_running_speed()

episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")

traces_act, traces_rest, diffs_act, diffs_rest = get_vals_plot(episodes, cond= HMcond)

trace_act = [row for arr in traces_act for row in arr]
trace_rest = [row for arr in traces_rest for row in arr]

d_bsl = calc_diff_baseline(np.mean(trace_act, axis=0), np.mean(trace_rest, axis=0))
d_dFoF_act = np.mean(get_variation_dFoF(episodes, cond=HMcond, pre_stim=pre_stim))
d_dFoF_rest = np.mean(get_variation_dFoF(episodes, cond=~HMcond, pre_stim=pre_stim))

plot_trace_vdFoF(trace_act, trace_rest)
plot_dFoF(diffs_act, diffs_rest, protocol=protocol, filename=filename_, metric = "roi")
print("Variation baseline (act - rest): ", d_bsl)
print("Variation dFoF act : ", d_dFoF_act)
print("Variation dFoF rest : ", d_dFoF_rest)

# aligned traces
plot_trace_vdFoF(trace_act, trace_rest, aligned=True)


#%% ######################################################################################
################################# TEST ALL FILES (ROI combined) ##########################
##########################################################################################

#%% not dividing by behavior
#protocol = "Natural-Images-4-repeats"
#protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
protocol = "static-patch"
#protocol = "looming-stim"

traces_ = []
diffs_ = []
for i in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][i]
    filename_ = os.path.basename(filename)
    data = Data(filename,
                verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()

    episodes = EpisodeData(data, 
                        quantities=['dFoF', 'Pupil', 'Running-Speed'],
                        protocol_name=protocol,
                        prestim_duration=pre_stim, 
                        verbose=False)
    print(data.nROIs)

    diffs = get_variation_dFoF(episodes, cond=None, pre_stim=pre_stim)
    diffs_.append(diffs)

flattened = [row for arr in diffs_ for row in arr]
############################
cols = 2  # Number of columns per row
rows = 1  # Compute the required number of rows
fig, AX = plt.subplots(rows, cols, figsize=(7, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.6)
#############################################################################################################
#barplot all rois
bar_width = 0.4
x = np.arange(1)  
means = [np.nanmean(flattened)]
AX[0].bar(x, means, width=bar_width, color=['grey'], edgecolor='black')

x = np.full_like(flattened, x[0])
jitter_strength = 0.2  # Adjust for more/less jitter
x_jitter = x + np.random.uniform(-jitter_strength, jitter_strength, size=len(x))

AX[0].scatter(x_jitter, flattened, color='grey', zorder=4, label="Active", alpha=0.7)
#AX[1].set_xticks(x, ['Protocol'])
AX[0].set_xlabel("All")
AX[0].set_ylabel("Variation of dFoF")
#AX[1].set_title(f"all ROIs for 1 file \n n ROIs = {len(diffs_act)}")

# Annotate each bar with its mean value
for i in range(1):
    AX[0].text(i, np.max(means) + 3, f"mean {means[i]:.3f}", ha='center', fontsize=6)

print("ALL ROIs for 1 file")
print("number of ROIs :", len(flattened))
print(f" mean : {means[0]:.3f}")

#violing all rois for 1 file
d = {' ': flattened}
df = pd.DataFrame(data=d)
df_melted = df.melt(var_name="All", value_name="Variation of dFoF")
sns.violinplot(data=df_melted, 
                x="All",
                hue="All", 
                y="Variation of dFoF", 
                inner="quart", 
                palette={" ": "grey"}, 
                ax=AX[1], 
                legend=False)

# Calculate mean values for labels
means_ = df_melted.groupby("All")["Variation of dFoF"].mean()

# Annotate each bar with its mean value
for i, mean in enumerate(means_):
    AX[1].text(i, np.max(means_)+2, f"mean {mean:.3f}", ha='center', fontsize=6)

AX[0].set_ylim([-4,10])
AX[1].set_ylim([-4,10])

#%% ################################################################
###################### REST AND ACTIVE #############################
####################################################################
protocol = "Natural-Images-4-repeats"
#protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
#protocol = "static-patch"
#protocol = "looming-stim"

traces_act_s, traces_rest_s, diffs_act_s, diffs_rest_s = [], [], [], []
traces_act_resp_s, traces_rest_resp_s, traces_act_pos_s, traces_rest_pos_s, traces_act_neg_s, traces_rest_neg_s = [], [], [], [], [], []
diffs_act_resp_s, diffs_rest_resp_s = [], []
diffs_act_pos_s, diffs_rest_pos_s = [], []
diffs_act_neg_s, diffs_rest_neg_s = [], []

for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    filename_ = os.path.basename(filename)
    data = Data(filename,
                verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()

    episodes_ = EpisodeData(data, 
                        quantities=['dFoF'],
                        protocol_name=protocol,
                        prestim_duration=pre_stim, 
                        verbose=False)

    episodes = EpisodeData(data, 
                        quantities=['dFoF', 'Pupil', 'Running-Speed'],
                        protocol_name=protocol,
                        prestim_duration=pre_stim, 
                        verbose=False)

    HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")
    traces_act, traces_rest, diffs_act, diffs_rest = get_vals_plot(episodes, cond= HMcond)
    
    

    resp_cond, pos_cond, neg_cond = calc_responsiveness(episodes_, data.nROIs)

    traces_act_resp  = np.array([traces_act[i][resp_cond] for i in range(len(traces_act))]) 
    traces_act_pos  = np.array([traces_act[i][pos_cond] for i in range(len(traces_act))]) 
    traces_act_neg  = np.array([traces_act[i][neg_cond] for i in range(len(traces_act))]) 

    traces_rest_resp  = np.array([traces_rest[i][resp_cond] for i in range(len(traces_rest))]) 
    traces_rest_pos  = np.array([traces_rest[i][pos_cond] for i in range(len(traces_rest))]) 
    traces_rest_neg  = np.array([traces_rest[i][neg_cond] for i in range(len(traces_rest))]) 

    
    diffs_act_resp = diffs_act[resp_cond] 
    diffs_act_pos = diffs_act[pos_cond] 
    diffs_act_neg = diffs_act[neg_cond] 
    diffs_rest_resp = diffs_rest[resp_cond]
    diffs_rest_pos = diffs_rest[pos_cond]
    diffs_rest_neg = diffs_rest[neg_cond]
    
    #fill lists
    diffs_act_s.append(diffs_act)
    diffs_rest_s.append(diffs_rest)
    diffs_act_resp_s.append(diffs_act_resp)
    diffs_rest_resp_s.append(diffs_rest_resp)
    diffs_act_pos_s.append(diffs_act_pos)
    diffs_rest_pos_s.append(diffs_rest_pos)
    diffs_act_neg_s.append(diffs_act_neg)
    diffs_rest_neg_s.append(diffs_rest_neg)

    traces_act_s.append(traces_act)
    traces_rest_s.append(traces_rest)
    traces_act_resp_s.append(traces_act_resp)
    traces_rest_resp_s.append(traces_rest_resp)
    traces_act_pos_s.append(traces_act_pos)
    traces_rest_pos_s.append(traces_rest_pos)
    traces_act_neg_s.append(traces_act_neg)
    traces_rest_neg_s.append(traces_rest_neg)



#%% ALL CELLS 
####################################################################
####################################################################
####################################################################
temp_act = [np.nanmean(traces_act_s[i], axis=0 ) for i in range(len(traces_act_s))]
flattened_act = [row for arr in temp_act for row in arr]

temp_rest = [np.nanmean(traces_rest_s[i], axis=0 ) for i in range(len(traces_rest_s))]
flattened_rest = [row for arr in temp_rest for row in arr]

#combine ROIs
diffs_act_all = np.concatenate(diffs_act_s) 
diffs_rest_all = np.concatenate(diffs_rest_s)  

#why are there nans?
diffs_act_all_ = [x for x in diffs_act_all if not np.isnan(x)]
diffs_rest_all_ = [x for x in diffs_rest_all if not np.isnan(x)]

plot_trace_vdFoF(flattened_act, flattened_rest)
plot_trace_vdFoF(flattened_act, flattened_rest, aligned=True)
plot_dFoF(diffs_act_all, diffs_rest_all, protocol=protocol, filename="ALL recordings", metric="roi")

boxplot_dict = {"title": "Amplitude peak Act vs Rest",
                "data" : [diffs_act_all_, diffs_rest_all_],
                "labels" : ["Active", "Rest"], 
                "colors" : ["greenyellow","darkolivegreen"], 
                "y_label": "Amplitude peak dFoF", 
                "figsize": (1,3)}

plot_boxplot(boxplot_dict)

#%% ALL RESPONSIVE CELLS
temp_act_resp = [np.nanmean(traces_act_resp_s[i], axis=0 ) for i in range(len(traces_act_resp_s))]
temp_act_resp = [arr for arr in temp_act_resp if not np.isnan(arr).any()]
flattened_act_resp = [row for arr in temp_act_resp for row in arr]

temp_rest_resp = [np.nanmean(traces_rest_resp_s[i], axis=0 ) for i in range(len(traces_rest_resp_s))]
temp_rest_resp = [arr for arr in temp_rest_resp if not np.isnan(arr).any()]
flattened_rest_resp = [row for arr in temp_rest_resp for row in arr]

diffs_act_all_resp = np.concatenate(diffs_act_resp_s) 
diffs_rest_all_resp = np.concatenate(diffs_rest_resp_s)  

#why are there nans?
diffs_act_all_resp_ = [x for x in diffs_act_all_resp if not np.isnan(x)]
diffs_rest_all_resp_ = [x for x in diffs_rest_all_resp if not np.isnan(x)]

plot_trace_vdFoF(flattened_act_resp, flattened_rest_resp)
plot_trace_vdFoF(flattened_act_resp, flattened_rest_resp, aligned=True)
plot_dFoF(diffs_act_all_resp, diffs_rest_all_resp, protocol=protocol, filename="ALL recordings", metric="roi")

boxplot_dict = {"title": "Difference baselines Act vs Rest",
                "data" : [diffs_act_all_resp_, diffs_rest_all_resp_],
                "labels" : ["Active", "Rest"], 
                "colors" : ["orangered","maroon"]}

plot_boxplot(boxplot_dict)

#%% POSITIVE CELLS
temp_act_pos = [np.nanmean(traces_act_pos_s[i], axis=0 ) for i in range(len(traces_act_pos_s))]
temp_act_pos = [arr for arr in temp_act_pos if not np.isnan(arr).any()]
flattened_act_pos = [row for arr in temp_act_pos for row in arr]

temp_rest_pos = [np.nanmean(traces_rest_pos_s[i], axis=0 ) for i in range(len(traces_rest_pos_s))]
temp_rest_pos = [arr for arr in temp_rest_pos if not np.isnan(arr).any()]
flattened_rest_pos = [row for arr in temp_rest_pos for row in arr]

diffs_act_all_pos = np.concatenate(diffs_act_pos_s) 
diffs_rest_all_pos = np.concatenate(diffs_rest_pos_s)  

#why are there nans?
diffs_act_all_pos_ = [x for x in diffs_act_all_pos if not np.isnan(x)]
diffs_rest_all_pos_ = [x for x in diffs_rest_all_pos if not np.isnan(x)]

plot_trace_vdFoF(flattened_act_pos, flattened_rest_pos)
plot_trace_vdFoF(flattened_act_pos, flattened_rest_pos, aligned=True)
plot_dFoF(diffs_act_all_pos, diffs_rest_all_pos, protocol=protocol, filename="ALL recordings", metric="roi")

boxplot_dict = {"title": "Amplitude peak Act vs Rest",
                "data" : [diffs_act_all_pos_, diffs_rest_all_pos_],
                "labels" : ["Active", "Rest"], 
                "colors" : ["greenyellow","darkolivegreen"], 
                "y_label": "Amplitude peak dFoF", 
                "figsize": (1,3)}

plot_boxplot(boxplot_dict)

#%% NEGATIVE CELLS
temp_act_neg = [np.nanmean(traces_act_neg_s[i], axis=0 ) for i in range(len(traces_act_neg_s))]
temp_act_neg = [arr for arr in temp_act_neg if not np.isnan(arr).any()]
flattened_act_neg = [row for arr in temp_act_neg for row in arr]

temp_rest_neg = [np.nanmean(traces_rest_neg_s[i], axis=0 ) for i in range(len(traces_rest_neg_s))]
temp_rest_neg = [arr for arr in temp_rest_neg if not np.isnan(arr).any()]
flattened_rest_neg = [row for arr in temp_rest_neg for row in arr]

diffs_act_all_neg = np.concatenate(diffs_act_neg_s) 
diffs_rest_all_neg = np.concatenate(diffs_rest_neg_s)  

#why are there nans?
diffs_act_all_neg_ = [x for x in diffs_act_all_neg if not np.isnan(x)]
diffs_rest_all_neg_ = [x for x in diffs_rest_all_neg if not np.isnan(x)]

plot_trace_vdFoF(flattened_act_neg, flattened_rest_neg)
plot_trace_vdFoF(flattened_act_neg, flattened_rest_neg, aligned=True)
plot_dFoF(diffs_act_all_neg, diffs_rest_all_neg, protocol=protocol, filename="ALL recordings", metric="roi")

boxplot_dict = {"title": "Difference baselines Act vs Rest",
                "data" : [diffs_act_all_neg_, diffs_rest_all_neg_],
                "labels" : ["Active", "Rest"], 
                "colors" : ["greenyellow","darkolivegreen"], 
                "y_label": "Amplitude peak dFoF", 
                "figsize": (1,3)}
plot_boxplot(boxplot_dict)


#%% VARIATION BASELINE ACT vs BASELINE REST for the different groups of ROIs
d_bsl_all = get_diffs_baseline(traces_act_s, traces_rest_s)
d_bsl_resp = get_diffs_baseline(traces_act_resp_s, traces_rest_resp_s)
d_bsl_pos = get_diffs_baseline(traces_act_pos_s, traces_rest_pos_s)
d_bsl_neg = get_diffs_baseline(traces_act_neg_s, traces_rest_neg_s)

boxplot_dict = {"title": "Difference baselines",
                "data" : [d_bsl_all, d_bsl_resp, d_bsl_pos, d_bsl_neg],
                "labels" : ["All", "Responsive", "Positive", "Negative"], 
                "colors" : ["grey", "yellow", "#b40426", "#3b4cc0"], 
                "y_label": "baseline act − baseline rest", 
                "figsize": (4,3)}

plot_boxplot(boxplot_dict)

boxplot_dict = {"title": "Difference baselines",
                "data" : [d_bsl_all, d_bsl_pos, d_bsl_neg],
                "labels" : ["All", "Positive", "Negative"], 
                "colors" : ["grey", "#b40426", "#3b4cc0"], 
                "y_label": "baseline act − baseline rest", 
                "figsize": (3,3)}

plot_boxplot(boxplot_dict)

