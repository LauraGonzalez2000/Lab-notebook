# %% [markdown]
# # Responsiveness dynamics

#%%
# PACKAGES
import sys, os
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from pathlib import Path

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics

pt.set_style("manuscript")
#%%
# FUNCTIONS
def compute_high_arousal_cond(episodes, 
                              pre_stim = 0,
                              pupil_threshold = 0.29, 
                              running_speed_threshold = 0.1, 
                              metric = None):
    """
    Calculates wether the episodes are aroused/active or calm/resting.

    Args:
        episodes (array of Episode): (Episode#, ROI#, dFoF_values (0.5ms sampling rate)).
        pupil_threshold (float) : The threshold to discriminate calm state and aroused state
        running_speed_threshold (float): The threshold to discriminate resting state and active state.
        metric (string) : metric used to split calm/rest and aroused/active states. ("pupil" or "locomotion")

    Returns:
        np.array : HMcond is True when active/aroused and false when resting/calm
    """
    cond = []
    
    if metric=="pupil":
        '''
        if pupil_threshold is not None:
            cond = (episodes.pupil_diameter.mean(axis=1)>pupil_threshold)
        else:
            print("pupil_threshold not given")
        '''
        if pupil_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.pupil_diameter[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > pupil_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("pupil_threshold not given")
            


    if metric=="locomotion":
        
        if running_speed_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.running_speed[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > running_speed_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("running_speed_threshold not given")

    return cond

def generate_Resp_ROI_dict(data_s, protocols=[''], metric = "category", state='all', subprotocols=False):

    #initialize
    nROIS = sum(data.nROIs for data in data_s)

    if protocols == ['']:
        protocols = [p for p in data_s[0].protocols if (p != 'grey-20min')]
    #Resp_ROI_dict = {f"ROI_{i}": dict.fromkeys(protocols, None) for i in range(nROIS)}
    Resp_ROI_dict = {f"ROI_{i}": {p: [] for p in protocols} for i in range(nROIS)}
    
    #fill
    nROI_id = 0
    for data in data_s:
        print("\n\n data : ", data, "\n\n")
        if protocols == ['']:
            protocols = [p for p in data.protocols if (p != 'grey-20min')]

        for p in protocols: 

            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF', 'running_speed'])
            
            if state == 'all':
                cond_b = ep.find_episode_cond()
            elif state == "active":
                cond_b = compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.1, metric="locomotion")
            elif state == "rest":
                cond_b = ~compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.1, metric="locomotion")

            varied_params = [k for k in ep.varied_parameters.keys() if k != 'repeat']
            #varied_params = [ep.varied_parameters.keys()]
       
            param_values = []
            cond_p = ep.find_episode_cond()

            if len(varied_params) > 0 : 
                param_values = ep.varied_parameters[varied_params[0]]
                for param in varied_params:
                    cond_p = [ep.find_episode_cond(key=param,value=param_v) for param_v in param_values]
                
            if subprotocols==True: 

                if len(varied_params) > 0 : 
                    cond = [cond_p_i & cond_b for cond_p_i in cond_p]
                else: 
                    cond = [cond_p & cond_b]
            else: 
                cond=[cond_b]

            for cond_i in cond: 
                for roi_n in range(data.nROIs):

                    t0 = max([0, ep.time_duration[0]-1.5])
                    
                    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                            interval_post=[t0, t0+1.5],                                   
                                            test='ttest', 
                                            sign='both')
                    if p== "looming-stim":
                        t0 = max([0, ep.time_duration[0]-0.5])
                        stat_test_props = dict(interval_pre=[-0.5,0],                                   
                                                interval_post=[t0, t0+0.5],                                   
                                                test='ttest', 
                                                sign='both')
                
                    roi_summary_data = pre_post_statistics(ep,
                                                    episode_cond = cond_i, #ep.find_episode_cond(),
                                                    response_args = dict(roiIndex=roi_n),
                                                    response_significance_threshold=0.05,
                                                    stat_test_props=stat_test_props,
                                                    repetition_keys=list(ep.varied_parameters.keys()), 
                                                    nMin_episodes=2)  #is that ok??
                    
                    raw_value = roi_summary_data["value"]
                    
                    if raw_value: 
                        if isinstance(raw_value, (list, np.ndarray)):
                            value = float(np.array(raw_value).squeeze())
                        else:
                            value = float(raw_value)

                        #value = roi_summary_data['value']

                        if bool(roi_summary_data['significant'])==False:
                            category = 'NS'
                        else: 
                            if roi_summary_data['value']>0:
                                category = "Positive"
                            else: 
                                category = "Negative"

                        if metric == "category" : 
                            Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(category)

                        elif metric == "value" : 
                            #Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p] = value
                            #print(Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p])
                            Resp_ROI_dict[f"ROI_{nROI_id + roi_n}"][p].append(value)

        nROI_id += data.nROIs

    return Resp_ROI_dict

def nonlinear_cmap(cmap, vmin, vmax, exp=0.5, N=256):
    """
    Smooth nonlinear remapping of a colormap with:
    - asymmetric vmin/vmax
    - midpoint fixed at value = 0
    - power-law control of color saturation
    """
    i = np.linspace(0, 1, N)
    mid = (0 - vmin) / (vmax - vmin)  # where value=0 lies in the data range

    i_nl = np.empty_like(i)

    left = i <= mid
    right = i > mid

    # left side (vmin -> 0)
    i_nl[left] = (0.5 * (i[left] / mid) ** exp)

    # right side (0 -> vmax)
    i_nl[right] = (0.5 + 0.5 * ((i[right] - mid) / (1 - mid)) ** exp)

    colors = cmap(i_nl)
    return mcolors.ListedColormap(colors)

def plot_heatmap(stim_similarity, separators=True):
    # Plot heatmap
    fig, AX = pt.figure(figsize=(10,10),ax_scale=(2, 3.5))
    vmin = -1 #np.min(stim_similarity.values)
    vmax = 1 #np.max(stim_similarity.values)
    AX.imshow(stim_similarity, 
            aspect='auto', 
            cmap= pt.plt.cm.PiYG, 
            vmin =vmin, 
            vmax=vmax)

    cmap_PiGY_nl = nonlinear_cmap(pt.plt.cm.PiYG, 
                                  vmin=vmin, vmax=vmax, 
                                  exp = 0.7, N=256)

    pt.bar_legend(AX,
                colorbar_inset=dict(rect=[1.1,.1,.04,.8]),
                colormap = cmap_PiGY_nl, 
                           # pt.binary, 
                           # pt.plt.cm.plasma 
                           # pt.plt.cm.coolwarm
                bar_legend_args={"fontsize":10},
                bounds=[vmin, vmax],
                ticks = [vmin, 0, vmax],
                # bar_legend_args={'size':2}, 
                label='Cross-correlation \nsimilarity')
                # no_ticks=True)

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=range(len(df.columns)), 
                yticks_labels=df.columns,
                xticks=range(len(df.columns)), 
                xticks_labels=df.columns,
                xticks_rotation=90,
                fontsize=5)
    if separators: 
        for i in [-0.5, 1.5, 5.5, 7.5, 8.5, 13.5, 17.5]:
            print(i)
            AX.axvline(x=i, color='black', linewidth=0.5)
            AX.axhline(y=i, color='black', linewidth=0.5)

def plot_raster(Resp_ROI_dict):
    cmap_graywarm = mcolors.LinearSegmentedColormap.from_list("graywarm",
                                                          ["#3b4cc0",  # blue (negative)
                                                           "#bdbbbb",  # mid gray (zero)
                                                           "#b40426"],   # red (positive)
                                                          N=256)

    vmin = -1
    vmax = 3

    # Convert to matrix
    df = pd.DataFrame.from_dict(Resp_ROI_dict).T

    expanded_cols = []

    for col in df.columns:
        expanded = df[col].apply(pd.Series)
        expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
        expanded_cols.append(expanded)

    df = pd.concat(expanded_cols, axis=1)

    # ROI response vs STIM 
    fig, AX = pt.figure(figsize=(5,10), 
                        ax_scale=(3, 10)) 
            
    cmap_graywarm_nl = nonlinear_cmap(cmap_graywarm, 
                                        vmin=vmin, 
                                        vmax=vmax, 
                                        exp = 0.7, 
                                        N=256)

    AX.imshow(df.values, 
            aspect='auto', 
            cmap= cmap_graywarm_nl, 
            vmin = vmin,
            vmax = vmax, 
            interpolation='nearest')

    pt.bar_legend(AX, 
                colorbar_inset=dict(rect=[1.1,.1,.04,.8], 
                                    facecolor=None),
                colormap = cmap_graywarm_nl, 
                            #pt.binary, 
                            # #pt.plt.cm.plasma 
                            # #pt.plt.cm.coolwarm
                bar_legend_args={'fontsize':1},
                label='Amplitude response post-pre',
                X=np.arange(vmin, vmax+0.5, 0.5),
                bounds=[vmin, vmax],
                ticks = None,
                ticks_labels=None,
                no_ticks=False,
                orientation='vertical')

    pt.set_plot(AX, 
                spines = ['bottom', 'left'],
                yticks=[0, len(df.index)],
                ylabel='ROI',
                xticks=range(len(df.columns)), 
                xticks_labels=df.columns,
                xticks_rotation=90,
                fontsize=8)

#%% 

datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for i in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][i], verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

#protocols = ["static-patch", 
#            "drifting-gratings", 
#            "Natural-Images-4-repeats"]

protocols = ["moving-dots",
                "random-dots",
                "static-patch",
                "looming-stim", 
                "Natural-Images-4-repeats", 
                "drifting-gratings"]

#%% GENERATE CONTINOUS DATA DICT - WITH SUBPROTOCOLS
Resp_ROI_dict = generate_Resp_ROI_dict(data_s, 
                                        protocols=protocols, 
                                        metric="value", 
                                        state="all", 
                                        subprotocols=True)

#%% RASTER PLOT - PEAK AMPLITUDE
plot_raster(Resp_ROI_dict)

#%% HEATMAP - STIM VS STIM SIMILARITY 
df = pd.DataFrame.from_dict(Resp_ROI_dict).T

expanded_cols = []

for col in df.columns:
    expanded = df[col].apply(pd.Series)
    expanded.columns = [f"{col}-{i+1}" for i in expanded.columns]
    expanded_cols.append(expanded)

df = pd.concat(expanded_cols, axis=1)

stim_similarity = np.ones((len(df.columns), len(df.columns)))*np.nan

for i in range(stim_similarity.shape[0]):
    for j in range(stim_similarity.shape[0]):
        stim_similarity[i,j] = df[df.columns[i]].corr(df[df.columns[j]])

plot_heatmap(stim_similarity)
#%%