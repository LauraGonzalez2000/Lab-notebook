# %% [markdown]
# # Running overview 
# plot dFoF in running and resting states - for the different protocols and subprotocols

# %% [markdown]
### Load packages and define constants:

#%%
import sys, os

import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData


import matplotlib.pyplot as plt

from utils_.General_overview_episodes import compute_high_arousal_cond

import random
import itertools

running_speed_threshold = 0.5  #cm/s

#%%
def plot_locomotion(data,  
                    protocol,
                    running_speed_threshold=0.5,
                    pupil_threshold=2.9, 
                    pre_stim = 0):

    episodes = EpisodeData(data, 
                           quantities=['dFoF', 'running_speed', 'pupil'], 
                           protocol_name=protocol, 
                           prestim_duration=pre_stim)

    HMcond = compute_high_arousal_cond(episodes, 
                                       pre_stim = pre_stim,
                                       pupil_threshold = pupil_threshold, 
                                       running_speed_threshold = running_speed_threshold, 
                                       metric = 'locomotion')

    episode_n = random.randint(0, episodes.dFoF.shape[0]-1) #chosen randomly 1 
    
    fig, AX = plt.subplots(1, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.8)
    
    start = int(pre_stim*1000)
    end = int(start + episodes.time_duration[0]*1000)

    
    start_sec = 0
    end_sec = int(episodes.time_duration[0])
    
    state = ['active' if HMcond[episode_n] else 'resting']
    color = 'orangered' if HMcond[episode_n] else 'grey'
    #color = 'blue' 
    
    AX.plot(episodes.t, episodes.running_speed[episode_n, :],color=color)
    AX.hlines(episodes.running_speed[episode_n, start:end].mean(axis=0),
                  xmin = start_sec,
                  xmax = end_sec,
                  color='dimgray',
                  linestyle=':')
    
    AX.axvspan(0, episodes.time_duration[0], color='lightgrey')
    
    AX.set_xlabel('Time (s)', fontsize=9)
    AX.set_ylabel('locomotion (cm/s)', fontsize=9)
    AX.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
    AX.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
    AX.grid(False)
    AX.tick_params(axis='both', which='both', bottom=True, left=True)
    AX.axhline(0.5, color="crimson")
    
    return fig, AX

def plot_average_visually_evoked_activity(data_s,
                                              dataIndex=None,
                                              roiIndex=None,
                                              pupil_threshold=2.4,
                                              running_speed_threshold=0.5):

    """
    Computes and plots average visually-evoked activity across sessions.
    For each protocol:
        • High arousal trials (HAcond)
        • Low arousal trials (~HAcond)
    Only the final average across files is plotted (mean ± SEM).
    """


    if dataIndex is not None: 
        data_s = [data_s[dataIndex]]
        mode = 'single'
    else: 
        mode = 'average'

    protocols = [p for p in data_s[0].protocols
                 if (p not in ['grey-10min', 'black-2min', 'quick-spatial-mapping'])]

    RESULTS = {}

    for protocol in protocols:
        episodes0 = EpisodeData(
            data_s[0],
            quantities=['dFoF'],
            protocol_name=protocol,
            verbose=False)

        varied_keys = [k for k in episodes0.varied_parameters.keys()
                    if k != 'repeat']
        varied_vals = [episodes0.varied_parameters[k] for k in varied_keys]

        
        if varied_keys != []:
            RESULTS[protocol] = {'traces_a': {k: {v: [] for v in episodes0.varied_parameters[k]}
                                                 for k in varied_keys},
                                 'traces_r': {k: {v: [] for v in episodes0.varied_parameters[k]}
                                                 for k in varied_keys}}
        else: 
            RESULTS[protocol] = {'traces_a': {'no_key': {'no_val': []}},
                                 'traces_r': {'no_key': {'no_val': []}}}

    for session in data_s:

        for protocol in protocols:

            episodes = EpisodeData(session,
                                   quantities=[ 'Running-Speed', 'dFoF'],
                                   protocol_name=protocol,
                                   prestim_duration=0,
                                   verbose=False)

            HAcond = compute_high_arousal_cond(episodes,
                                               pre_stim=0,
                                               pupil_threshold=pupil_threshold,
                                               running_speed_threshold=running_speed_threshold,
                                               metric='locomotion')
            
            varied_keys = [k for k in episodes.varied_parameters.keys() if k != 'repeat']
            varied_vals = [episodes.varied_parameters[k] for k in varied_keys]
            
            if varied_keys != []:
                for key in varied_keys: 
                    
                    for value in itertools.product(*varied_vals):
                        stim_cond = episodes.find_episode_cond(key=key, value=value)

                        # ------ LOW AROUSAL ------
                        cond_r = stim_cond & (~HAcond)
                        if np.sum(cond_r) > 1:

                            trace = episodes.dFoF[cond_r]   # shape: trials × ROI × time

                            if roiIndex is not None:
                                trace = trace[:, roiIndex, :]
                                RESULTS[protocol]['traces_r'][key][value[0]].append(trace)

                            else: 
                                trace = np.nanmean(trace, axis=0)
                                RESULTS[protocol]['traces_r'][key][value[0]].append(trace)
                            
                        # ------ HIGH AROUSAL ------
                        cond_a = stim_cond & HAcond
                        if np.sum(cond_a) > 1:

                            trace = episodes.dFoF[cond_a]   # shape: trials × ROI × time
                        
                            if roiIndex is not None:
                                trace = trace[:, roiIndex, :]
                                RESULTS[protocol]['traces_a'][key][value[0]].append(trace)
                            else:
                                trace = np.nanmean(trace, axis=0)
                                RESULTS[protocol]['traces_a'][key][value[0]].append(trace)
            else:  
                stim_cond = episodes.find_episode_cond()
                # ------ LOW AROUSAL ------
                cond_r = stim_cond & (~HAcond)
                if np.sum(cond_r) > 1:
                    trace = episodes.dFoF[cond_r]   # shape: trials × ROI × time
                    if roiIndex is not None:
                        trace = trace[:, roiIndex, :]
                        RESULTS[protocol]['traces_r']['no_key']['no_val'].append(trace)
                    else: 
                        trace = np.nanmean(trace, axis=0)
                        RESULTS[protocol]['traces_r']['no_key']['no_val'].append(trace)

                # ------ HIGH AROUSAL ------
                cond_a = stim_cond & HAcond
                if np.sum(cond_a) > 1:
                    trace = episodes.dFoF[cond_a]   # shape: trials × ROI × time
                    if roiIndex is not None:
                        trace = trace[:, roiIndex, :]
                        RESULTS[protocol]['traces_a']['no_key']['no_val'].append(trace)
                    else:
                        trace = np.nanmean(trace, axis=0)  
                        RESULTS[protocol]['traces_a']['no_key']['no_val'].append(trace)
    
    fig, AX = pt.figure(axes_extents=[[ [1,1] for _ in protocols ] for _ in range(5)], figsize=(7,6))
    pt.plt.subplots_adjust(wspace=0.3)

    for p, protocol in enumerate(protocols):
        i=0
        episodes = EpisodeData(session,
                               quantities=[ 'Running-Speed', 'dFoF'],
                               protocol_name=protocol,
                               prestim_duration=0,
                               verbose=False)
        varied_keys = [k for k in episodes.varied_parameters.keys() if k != 'repeat']
        varied_vals = [episodes.varied_parameters[k] for k in varied_keys]
        if varied_keys != []:
            for key in varied_keys: 
                for value in itertools.product(*varied_vals):
                    #print("values : ", values)
                    ax = AX[i][p]

                    # ---- HIGH AROUSAL ----
                    if len(RESULTS[protocol]['traces_a'][key][value[0]]) > 0:
                        arr = np.vstack(RESULTS[protocol]['traces_a'][key][value[0]])
                        #print(arr)
                        #print(len(arr))
                        mean1 = arr.mean(axis=0)
                        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])

                        #x = np.arange(len(mean1))
                        x = episodes.t
                        ax.plot(x, mean1, color='tab:orange')
                        ax.fill_between(x, mean1 - sem, mean1 + sem,
                                        color='tab:orange', alpha=0.25)
                    # ---- LOW AROUSAL ----
                    if len(RESULTS[protocol]['traces_r'][key][value[0]]) > 0:
                        arr = np.vstack(RESULTS[protocol]['traces_r'][key][value[0]])
                        x = episodes.t
                        #print(arr)
                        #print(len(arr))

                        if len(arr)==1: ax.plot(x, arr, color='tab:blue')
                            
                        if len(arr)>1:
                            mean2 = arr.mean(axis=0)
                            sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
                            ax.plot(x, mean2, color='tab:blue')
                            ax.fill_between(x, mean2 - sem, mean2 + sem,
                                            color='tab:blue', alpha=0.25)
                            
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("ΔF/F")
                    i+=1
                    try:
                        y1 = 1.2 * max(np.max(mean1), np.max(mean2))
                    except:
                        y1 = 1.2 * max(mean2)
                    ax.fill_between([0, np.mean(episodes.time_duration)], y1 = y1, color='grey', alpha=.2, lw=0)
                    
        else: 
            ax = AX[0][p]
            # ---- HIGH AROUSAL ----
            if len(RESULTS[protocol]['traces_a']['no_key']['no_val']) > 0:
                arr = np.vstack(RESULTS[protocol]['traces_a']['no_key']['no_val'])
                mean1 = arr.mean(axis=0)
                sem = arr.std(axis=0) / np.sqrt(arr.shape[0])

                x = np.arange(len(mean1))
                x = episodes.t
                ax.plot(x, mean1, color='tab:orange')
                ax.fill_between(x, mean1 - sem, mean1 + sem,
                                color='tab:orange', alpha=0.25)
            # ---- LOW AROUSAL ----
            if len(RESULTS[protocol]['traces_r']['no_key']['no_val']) > 0:
                arr = np.vstack(RESULTS[protocol]['traces_r']['no_key']['no_val'])
                mean2 = arr.mean(axis=0)
                sem = arr.std(axis=0) / np.sqrt(arr.shape[0])

                x = np.arange(len(mean2))
                x = episodes.t
                ax.plot(x, mean2, color='tab:blue')
                ax.fill_between(x, mean2 - sem, mean2 + sem,
                                color='tab:blue', alpha=0.25)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F")
            try: 
                y1 = 1.2 * max(np.max(mean1), np.max(mean2))
            except: 
                y1 = 1.2 * max(mean2)
            ax.fill_between([0,np.mean(episodes.time_duration)], y1 = y1, color='grey', alpha=.2, lw=0)


        AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                          (0.5,1.4),
                          xycoords='axes fraction', ha='center')

    
    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            fig.text(0.0, -0.02,'single session: %s ,   n=%i ROIs' %
                                (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),fontsize=7)
        else:
            fig.text(0.0, -0.02,'average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),fontsize=7)
            
    else:
        if mode == "single":
            fig.text(0.0, -0.02,'single session: %s ,   roi #%i' %
                                (data_s[0].filename.replace('.nwb',''), 1+roiIndex),fontsize=7)
        else:
            fig.text(0.0, -0.02,'average over %i sessions ,  roi #%i ' % 
                                (len(data_s), 1+roiIndex),fontsize=7)

    for ax in pt.flatten(AX):
        if len(ax.lines)==0 and len(ax.patches)==0 and len(ax.images)==0:
            fig.delaxes(ax)
    
    return fig

###################################################################################################################
#running overview necessary to filter good files. 
#%% Load Data
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_run')
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

#%%
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

#example file
dataIndex = 3
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(**dFoF_options, verbose=True)

# Plot locomotion example for a specific protocol
#%%
plot_locomotion(data, 
                protocol='static-patch',
                running_speed_threshold=0.5,
                pupil_threshold=2.9, 
                pre_stim = 1)

#%%
################### Plot average activity for all protocols one session ############################
dataIndex = 7
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(**dFoF_options, verbose=False)

#%%
data_s = [data]
#%%
fig = plot_average_visually_evoked_activity(data_s,
                                            dataIndex = 0, 
                                            roiIndex=None,
                                            pupil_threshold=2.4,
                                            running_speed_threshold=0.)

#%%
data_s = []
for i in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][i], verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data_s.append(data)
#%% some individuals 
for i in range(5):
    fig = plot_average_visually_evoked_activity(data_s,
                                                dataIndex = i, 
                                                roiIndex=None,
                                                pupil_threshold=2.4,
                                                running_speed_threshold=0.5)
    
################### Plot average activity for all protocols all sessions ############################
#%% average of all
fig = plot_average_visually_evoked_activity(data_s,
                                            dataIndex = None, 
                                            roiIndex=None,
                                            pupil_threshold=2.4,
                                            running_speed_threshold=0.5)
