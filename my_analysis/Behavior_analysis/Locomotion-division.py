# %% [markdown]
# # Locomotion analysis

# running speed vs pupil size 
# plot locomotion on episode -> play with condition
# running speed vs pupil size (per file - plot with all files)

# %%
# load packages:
import os, sys
sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.utils  import plot_tools as pt
from physion.analysis.episodes.build import EpisodeData

import numpy as np

import matplotlib.pylab as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import random
import sys

#%%
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

def plot_loco_pupil(data,
                    ax=None,
                    running_speed_threshold=0.1,
                    pupil_threshold=2.9, 
                    metric=None, 
                    mylabel=False, 
                    prestim_dur = 0):
    
    protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(2,1.3))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    else:
        fig = None

    HAcount, LAcount = [], []
    for p, protocol in enumerate(protocols):
        try:
            behav_episodes = EpisodeData(data, 
                                         quantities=['dFoF', 'Pupil', 'Running-Speed'],
                                         protocol_name=protocol,
                                         prestim_duration=prestim_dur,
                                         verbose=False)
        except:
            try: 
                behav_episodes = EpisodeData(data, 
                                 quantities=['dFoF', 'Running-Speed'],
                                 protocol_name=protocol,
                                 prestim_duration=prestim_dur,
                                 verbose=False)
            except: 
                print("error when computing behavioral episodes")

        # HAcond: high arousal condition
        HAcond = compute_high_arousal_cond(behav_episodes, prestim_dur, pupil_threshold, running_speed_threshold, metric=metric)

        start = int(prestim_dur*1000)
        end = int(start + behav_episodes.time_duration[0]*1000)
        
        ax.scatter(behav_episodes.pupil_diameter[:, start:end].mean(axis=1)[~HAcond],
                   behav_episodes.running_speed[:, start:end].mean(axis=1)[~HAcond],
                   color='grey',
                   s=2)
        ax.scatter(behav_episodes.pupil_diameter[:, start:end].mean(axis=1)[HAcond],
                   behav_episodes.running_speed[:, start:end].mean(axis=1)[HAcond],
                   color='orangered',
                   s=2)
        
        HAcount.append(np.sum(HAcond))
        LAcount.append(np.sum(~HAcond))
        
    ax.set_ylabel('run. speed (cm/s)', fontsize=9)
    ax.set_xlabel('pupil size (mm)', fontsize=9)
    ax.annotate('\n n=%i ep.' % np.sum(HAcount), (0, 1), color='orangered', xycoords='axes fraction', va='top', fontsize=7)
    ax.annotate(' n=%i ep.' % np.sum(LAcount), (0, 1), color='grey', xycoords='axes fraction', va='top', fontsize=7)
    ax.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', bottom=True, left=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_title('Behavior across \n visual stimulation episodes', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    if metric=="locomotion":
        ax.axhline(running_speed_threshold, color = 'black', label = 'threshold', linewidth=0.6)
    elif metric=="pupil":
        ax.axvline(pupil_threshold, color = 'black', label = 'threshold', linewidth=0.6)  
    if mylabel:
        ax.annotate('\n high arousal', (1, 1), color='orangered', xycoords='axes fraction', va='top', fontsize=7)
        ax.annotate(' low arousal', (1, 1), color='grey', xycoords='axes fraction', va='top', fontsize=7)
    
    
    return fig, ax

def plot_locomotion(data,  
                    protocol,
                    running_speed_threshold=0.1,
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

    episode_n = random.randint(0, episodes.dFoF.shape[0]-1) #chosen randomly
    
    fig, AX = plt.subplots(1, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.8)
    
    start = int(pre_stim*1000)
    end = int(start + episodes.time_duration[0]*1000)
    
    start_sec = 0
    end_sec = int(episodes.time_duration[0])
    
    state = ['active' if HMcond[episode_n] else 'resting']
    color = 'orangered' if HMcond[episode_n] else 'grey'
    #color = 'blue' 
    print(f"Specific episode # {episode_n} ({state})") 
    
    AX.plot(episodes.t, episodes.running_speed[episode_n, :],color=color)
    AX.hlines(episodes.running_speed[episode_n, start:end].mean(axis=0),
                  xmin = start_sec,
                  xmax = end_sec,
                  color='dimgray',
                  linestyle=':')
    print("value to compare to threshold : ", episodes.running_speed[episode_n,start:end].mean(axis=0))

    AX.axvspan(0, episodes.time_duration[0], color='lightgrey')
    
    AX.set_xlabel('Time (s)', fontsize=9)
    AX.set_ylabel('locomotion (cm/s)', fontsize=9)
    AX.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
    AX.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
    AX.grid(False)
    AX.tick_params(axis='both', which='both', bottom=True, left=True)
    AX.axhline(0.1, color="crimson")
    
    return fig, AX

#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%%
#manual parameters
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

running_speed_threshold=0.1
pupil_threshold = 2.9
pre_stim = 1

#%%
index = 2  #for example this file 
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=True)
data.build_pupil_diameter()
data.build_running_speed()

#%%
fig, ax = plot_loco_pupil(data, running_speed_threshold=running_speed_threshold, pupil_threshold=pupil_threshold, metric="locomotion", mylabel=False, prestim_dur =0)

#%%
fig, ax = plot_loco_pupil(data, running_speed_threshold=running_speed_threshold, pupil_threshold=pupil_threshold, metric="pupil", mylabel=True)

#%%
protocol = "moving-dots" #Load episodes by protocol!

plot_locomotion(data, 
                protocol=protocol,
                running_speed_threshold=running_speed_threshold, 
                pupil_threshold=pupil_threshold, 
                pre_stim = pre_stim)

#%%

rows = 3
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,7))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    plot_loco_pupil(data, 
                    ax = AX[int(f/5)][f%5],
                    running_speed_threshold=running_speed_threshold, 
                    pupil_threshold=pupil_threshold, 
                    metric="locomotion", 
                    mylabel=False, 
                    prestim_dur =0)
    #plot_behavior_in_episodes(data, ax=AX[int(f/5)][f%5], metric='locomotion', mylabel=False)
    AX[int(f/5)][f%5].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)
    
for i in range(rows*cols-len(SESSIONS['files'])):
    AX[-1][i*(-1) - 1].axis('off')

fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/all_behavior_locomotion_pupil1.png", dpi=300, bbox_inches='tight')

#%%
rows = 3
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,7))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    plot_loco_pupil(data, 
                    ax = AX[int(f/5)][f%5],
                    running_speed_threshold=running_speed_threshold, 
                    pupil_threshold=pupil_threshold, 
                    metric="pupil", 
                    mylabel=False, 
                    prestim_dur =0)
    
    #plot_behavior_in_episodes(data, ax=AX[int(f/5)][f%5], metric='pupil', mylabel=False)
    AX[int(f/5)][f%5].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)
    
for i in range(rows*cols-len(SESSIONS['files'])):
    AX[-1][i*(-1) - 1].axis('off')

fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/all_behavior_locomotion_pupil2.png", dpi=300, bbox_inches='tight')


