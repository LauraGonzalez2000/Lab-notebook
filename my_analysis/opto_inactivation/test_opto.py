# %%
import numpy as np
import sys, os

# physion
sys.path += ['../../physion/src'] # add src code directory for physion

#import physion

# plot
from physion.utils import plot_tools as pt
# pt.set_style('dark')
pt.set_style('manuscript')

from physion.dataviz.imaging import get_FOV_image

from pathlib import Path

from physion.analysis.episodes.build import EpisodeData
from physion.analysis.read_NWB import Data

cmap = pt.get_linear_colormap('lightgreen', 'darkgreen')

from scipy import stats

sys.path += ['../']
from utils_.Responsiveness_methods import calc_responsiveness2
from utils_.my_math import calc_stats, plot_stats

import random

#%%
def plot_effect(Ep, data,
                roi=None,
                title=''):

    episodes = np.arange(len(Ep.time_start))
    blank_cond = (episodes%2==0)
    stim_cond = (episodes%2==1)

    fig, AX = pt.figure(\
        ax_scale=(0.9,0.95),
        axes=(len(Ep.varied_parameters['contrast']),1),
        wspace=0.2, hspace=0.5, top=1.5)

    fig.suptitle(title)

    if roi is None:
        dFoF = Ep.dFoF.mean(axis=1)
    else:
        dFoF = Ep.dFoF[:,roi,:]

    for i, c in enumerate(Ep.varied_parameters['contrast']):

        print(i, c)

        c_cond = Ep.find_episode_cond('contrast', value=c)

        if i%2==0:
            pt.plot(Ep.t, dFoF[c_cond & blank_cond,:].mean(axis=0),
                    stats.sem(dFoF[c_cond & blank_cond,:], axis=0),
                    ax=AX[i])
            pt.annotate(AX[i], 'blank', (1,1), ha='right', va='top')
        else:
            pt.plot(Ep.t, dFoF[c_cond & stim_cond,:].mean(axis=0),
                    stats.sem(dFoF[c_cond & stim_cond,:], axis=0),
                    ax=AX[i])
            pt.annotate(AX[i], 'stim', (1,1), ha='right', va='top')
        pt.annotate(AX[i], 'c=%.2f' % c, (0.5, 0), va='top', ha='center')

    pt.set_common_ylims(AX)
    y0, y1 = AX[0].get_ylim()
    for i, ax in enumerate(AX):
        # visual stim
        ax.fill_between([0,1],y0*np.ones(2),y1*np.ones(2), alpha=.2, lw=0)
        #ax.axis('off')
        # photo-stim
        if i%2==1:
            ax.fill_between([-1,2],y0*np.ones(2),y1*np.ones(2), 
                                color='tab:blue', alpha=.3, lw=0)

    pt.draw_bar_scales(AX[0],
                    Ybar=0.2, Ybar_label='0.2$\\Delta$F/F',
                    Xbar=1, Xbar_label='1s')
    
    return fig

def plot_traces(Ep, 
                roi=None, 
                title='', 
                average_by = None, 
                LED_on = [], 
                traces='dFoF', 
                color_led = 'lightskyblue'):
    """
    Takes as arguments : 
        roi can be None (average all rois), 
            a specific roi, 
            or a set of specific rois (average of that rois)
    """

    if len(LED_on)==0:
        LED_on, LED_off = find_EP_LED_on_off(Ep)

    if len(LED_on) != len(Ep.opto):
        LED_on = LED_on[:len(Ep.opto)]

    keys = list(Ep.varied_parameters.keys())
    n_keys = len(keys)

    if average_by == 'stimulus':
        n_cols = 1 * 2
        n_rows = 1
    else: 
        if n_keys == 3:
            n_cols = len(Ep.varied_parameters[keys[0]]) * 2
            n_rows = len(Ep.varied_parameters[keys[1]])
        elif n_keys == 2:
            n_cols = len(Ep.varied_parameters[keys[0]]) * 2
            n_rows = 1
        elif n_keys == 1:
            n_cols = 1 * 2
            n_rows = 1
    fig, AX = pt.figure(ax_scale=(0.9, 0.95), axes=(n_cols, n_rows),
                        wspace=0.7, hspace=1.0, top=2.7)
    fig.suptitle(title)

    if roi is None:
        if traces=='dFoF':
            trace = Ep.dFoF.mean(axis=1)
        if traces=='raw_fluo':
            trace = Ep.rawFluo.mean(axis=1)
            neuropil = Ep.neuropil.mean(axis=1)
    elif np.isscalar(roi):
        if traces=='dFoF':
            trace = Ep.dFoF[:, roi, :]
        if traces=='raw_fluo':
            trace = Ep.rawFluo[:, roi, :]
            neuropil = Ep.neuropil[:, roi, :]
    else:
        if traces=='dFoF':
            trace = Ep.dFoF[:, roi, :].mean(axis=1)
        if traces=='raw_fluo':
            trace = Ep.rawFluo[:, roi, :].mean(axis=1)
            neuropil = Ep.neuropil[:, roi, :].mean(axis=1)

    # Baseline subtraction
    baseline = Ep.t < 0
    trace = trace - trace[:, baseline].mean(axis=1, 
                                                  keepdims=True)
    if traces == 'raw_fluo':
        neuropil = neuropil - neuropil[:, baseline].mean(axis=1, 
                                                        keepdims=True)

    if average_by == 'stimulus':
        for i in range(n_cols):
            # LED vs blank
            if i % 2 == 0:
                LED = LED_off
            else:
                LED = LED_on
            
            ax = AX[i]
            if i % 2 == 1:
                ax.axvspan(-1, Ep.time_duration[0]+1,color=color_led,
                                       alpha=.3,lw=0)
            cond = LED
            pt.plot(Ep.t,
                    trace[cond, :].mean(axis=0),
                    ax=ax,
                    no_set=True,
                    color='tab:green')
            if traces=='raw_fluo':
                pt.plot(Ep.t,
                        neuropil[cond, :].mean(axis=0),
                        ax=ax,
                        no_set=True,
                        color='tab:red')
            # Stimulus shading
            if Ep.protocol_name != 'protocol-grey-screen':
                ax.axvspan(0, Ep.time_duration[0],
                            color='tab:gray',
                            alpha=.2,
                            lw=0)
            ax.axis('off')

    else: 
        for i in range(n_cols):
            # value of first varied parameter
            c = Ep.varied_parameters[keys[0]][i // 2]
            c_cond = Ep.find_episode_cond(keys[0],
                                        value=c)
            # LED vs blank
            if i % 2 == 0:
                LED = LED_off
            else:
                LED = LED_on

            for j in range(n_rows):
                # Get axis
                if n_rows == 1:
                    ax = AX[i]
                else:
                    ax = AX[j][i]

                if n_keys == 3: # Second varied parameter
                    s = Ep.varied_parameters[keys[1]][j]
                    s_cond = Ep.find_episode_cond(keys[1],
                                                value=s)
                    cond = c_cond & s_cond
                    label = 'c=%.2f\nr=%.2f' % (c, s) #generalize

                elif n_keys == 2:
                    cond = c_cond
                    label = 'r=%.2f' % c
                elif n_keys ==1:
                    cond = Ep.find_episode_cond()
                    label = ''

                # Add LED condition
                cond = cond & LED
                if i % 2 == 1:
                    ax.axvspan(-1, Ep.time_duration[0]+1,color=color_led,
                            alpha=.3,lw=0)

                pt.plot(Ep.t,
                        trace[cond, :].mean(axis=0),
                        ax=ax,
                        no_set=True,
                        color='tab:green')
                
                if traces== 'raw_fluo':
                    pt.plot(Ep.t,
                            neuropil[cond, :].mean(axis=0),
                            ax=ax,
                            no_set=True,
                            color='tab:red')

                if label:
                    pt.annotate(ax,
                                label,
                                (0.5, 0),
                                va='top',
                                ha='center')

                # Stimulus shading
                if Ep.protocol_name != 'protocol-grey-screen':
                    ax.axvspan(0, Ep.time_duration[0],
                                color='tab:gray',
                                alpha=.2,
                                lw=0)
                ax.axis('off')

    # --------------------------------------------------
    # Common formatting
    # --------------------------------------------------
    pt.set_common_ylims(AX)

    if n_rows == 1:
        last_ax = AX[-1]
        first_ax = AX[0]
    else:
        last_ax = AX[0][-1]
        first_ax = AX[0][0]
    
    if traces == 'raw_fluo':
        pt.annotate(last_ax,
                    'neuropil ',
                    (0, 1),
                    ha='right',
                    color='tab:red')

        pt.annotate(last_ax,
                    'raw ROI fluo',
                    (0.2, 1),
                    ha='left',
                    color='seagreen')
    if traces == 'dFoF':
        pt.annotate(last_ax,
                            'dFoF',
                            (0.5, 1),
                            ha='left',
                            color='seagreen')

    pt.draw_bar_scales(first_ax,
                        Xbar=1,
                        Xbar_label='1s', 
                        Ybar_label_format='%.2f')
    return fig

def find_EP_LED_on_off(Ep, OPTO_THRESHOLD = 0.05 ):
    # 1. Define time masks
    pre_mask = (Ep.t >= -0.5) & (Ep.t < 0.0)
    dur_mask = (Ep.t >= 0.0) & (Ep.t <= Ep.time_duration[0])
 
    # 2. Condition 1: Active for at least 0.5s before Ep start
    # (i.e. high for >= 90-95% of the -0.5s to 0s window to tolerate minor edge jitter)
    pre_active_fraction = np.mean(Ep.opto[:, pre_mask] > OPTO_THRESHOLD, axis=1)
    cond_pre = pre_active_fraction >= 0.90  # Use 1.0 if you require strict 100% of the 0.5s

    # 3. Condition 2: Active for at least 90% of the stimulus duration
    dur_active_fraction = np.mean(Ep.opto[:, dur_mask] > OPTO_THRESHOLD, axis=1)
    cond_dur = dur_active_fraction >= 0.90

    # 4. Combine both criteria
    LED_on = cond_pre & cond_dur

    print("# LED on episodes", np.sum(LED_on), "/", len(LED_on))

    # 2. Condition 1: inctive for at least 0.5s before Ep start
    # (i.e. high for >= 90-95% of the -0.5s to 0s window to tolerate minor edge jitter)
    pre_active_fraction = np.mean(Ep.opto[:, pre_mask] < OPTO_THRESHOLD, axis=1)
    cond_pre = pre_active_fraction >= 0.90  # Use 1.0 if you require strict 100% of the 0.5s

    # 3. Condition 2: Active for at least 90% of the stimulus duration
    dur_active_fraction = np.mean(Ep.opto[:, dur_mask] < OPTO_THRESHOLD, axis=1)
    cond_dur = dur_active_fraction >= 0.90

    # 4. Combine both criteria
    LED_off = cond_pre & cond_dur


    print("# LED off episodes", np.sum(LED_off), "/", len(LED_off))

    '''
    fig, ax = pt.plt.subplots(figsize=(12, 5))
    for i in range(Ep.opto.shape[0]):
        if LED_on[i]:
            ax.plot(Ep.opto[i, :], label=f'Episode {i}', color='blue')
        else: 
            ax.plot(Ep.opto[i, :], label=f'Episode {i}', color='k')
    ax.set_xlabel('Frame')
    ax.set_ylabel('LED signal')
    ax.legend()
    pt.plt.show()
    '''
    return LED_on, LED_off

def plot_variation_dFoF(Ep, average_by='stimulus', roi=None, LED_on = [], color_led='lightskyblue'):
    
    keys = list(Ep.varied_parameters.keys())
    n_keys = len(keys)

    if average_by == 'stimulus':
            n_cols = 1
            n_rows = 1
    else: 
        if n_keys == 3:
            n_cols = len(Ep.varied_parameters[keys[0]])
            n_rows = len(Ep.varied_parameters[keys[1]])
        elif n_keys == 2:
            n_cols = len(Ep.varied_parameters[keys[0]])
            n_rows = 1
        elif n_keys == 1:
            n_cols = 1
            n_rows = 1
    fig, ax = pt.figure(ax_scale=(0.9, 0.95), axes=(n_cols, n_rows),
                            wspace=0.7, hspace=1.0, top=2)

    if roi is None:
        dFoF = Ep.dFoF.mean(axis=1)
    elif np.isscalar(roi):
        dFoF = Ep.dFoF[:, roi, :]
    else:
        dFoF = Ep.dFoF[:, roi, :].mean(axis=1)

    #baseline blank vs baseline opto
    if Ep.protocol_name == 'protocol-grey-screen':
        baseline_t = (Ep.t > -2) & (Ep.t < -1)
        peak_t = (Ep.t > -1 ) & (Ep.t < Ep.time_duration[0] + 1)
    else: 
        baseline_t = (Ep.t > -1) & (Ep.t < 0)
        peak_t = (Ep.t > Ep.time_duration[0] - 0.5) & (Ep.t < Ep.time_duration[0])
    
    dFoF_baseline = dFoF[:, baseline_t].mean(axis=1, 
                                    keepdims=True)
    dFoF_peak = dFoF[:, peak_t].mean(axis=1, 
                                    keepdims=True)

    peak_val = dFoF_peak - dFoF_baseline

    if LED_on == []:
        LED_on, LED_off = find_EP_LED_on_off(Ep)

    peak_val_blank = peak_val[LED_off]
    peak_val_opto = peak_val[LED_on]

    print(peak_val_blank)
    print(np.mean(peak_val_blank))

    variation_dFoF = [np.mean(peak_val_blank), np.mean(peak_val_opto)]
    sy = [np.std(peak_val_blank)/np.sqrt(len(peak_val_blank)), np.std(peak_val_opto)/np.sqrt(len(peak_val_blank))]
    pt.bar(ax=ax,x=[0,1],y= variation_dFoF, sy= sy, 
                COLORS = ['grey', color_led])

    pt.set_plot(ax=ax, 
                #ylim=[-0.05, 0.05],
                xticks=[0, 1],
                xticks_labels=["Blank", "Opto"], 
                num_xticks=2,
                ylabel='d dFoF')
    print("variation dFoF", variation_dFoF)
    try : 
        stats = calc_stats('Variation dFoF',peak_val_blank,peak_val_opto, debug=True)
        plot_stats(ax, n_groups=2, stats=stats, y_pos1=1, y_pos2=1)
    except: 
        "problem with stats"

    return fig, ax
# %%
# 8ori2contrasts
#base2 = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'ffSG-8ori-2contrasts-opto')
#datafolder = os.path.join(base2,'NDNF-Cre','NWBs')
#path = os.path.join(base2,'NDNF-Cre','NWBs', '2026_08_17-12-18-40.nwb')
#path = os.path.join(base2,'NDNF-Cre','NWBs', '2026_08_17-14-17-42.nwb')

# other files
#stg old config
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03', 'NWBs_')
#path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03','NWBs_', '2025_10_03-12-02-05.nwb')
#path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03','NWBs_', '2025_10_03-14-48-12.nwb')

#%%
#vison-survey
# organised by LED-virus

base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto')

##################################################################################
#stgACR2 on NDNF and GCaMP on NDNF
#NONE -> RECORD MORE HERE

##################################################################################
#ArchT on NDNF and GCaMP on NDNF
#datafolder = os.path.join(base,'NDNF-Cre','NWBs')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-28-15.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-57-29.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_01-14-26-23.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_02-10-50-31.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_03-12-12-18.nwb')

###################################################################################
#stg on NDNF and GCaMP on PYR
#datafolder = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_28-15-45-15.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_09_02-11-56-52.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-11-00-02.nwb')
#new
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-41-06.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-11-38-44.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_05-16-58-19.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-01-46.nwb')

###################################################################################
#ArchT on NDNF and GCaMP on PYR
#datafolder = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs')
path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-15-41-49.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-16-16-41.nwb')
#issue with file
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-13-30-18.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-15-35-23.nwb')


#%%
#organised by LED intensity

base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto')
#LED M450LP2 -> 450 nm (blue)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-41-06.nwb')
#900mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_09_02-11-56-52.nwb')
#1000mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb')

#LED M455L3 -> 455 nm (blue)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-11-38-44.nwb')
#900mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_05-16-58-19.nwb')
#1000mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-01-46.nwb')


#LED M590L4 -> 590 nm (orange)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-15-41-49.nwb')
#900mA
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-28-15.nwb')
#1000mA
path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-16-16-41.nwb')

paths = [os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-41-06.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_09_02-11-56-52.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-11-38-44.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_05-16-58-19.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-01-46.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-15-41-49.nwb'),
         os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-28-15.nwb'),
         os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-16-16-41.nwb')]



#%%
data = Data(path)
data.build_visual_stim()
data.build_visual_stim()
protocol_id = 0
Ep = EpisodeData(data, protocol_id=protocol_id, 
                quantities=['dFoF','rawFluo', 'neuropil','opto'], 
                prestim_duration=2,
                verbose=True)
for index in range(100):
    image = data.visual_stim.plot_stim_picture(index=index)#, with_scale=True)

#%% 
#base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto')

#control animal with no opsin (3 animals)
#path = os.path.join(base,'C57BL6J','NWBs', '2026_10_02-17-18-59.nwb')
#path = os.path.join(base,'C57BL6J','NWBs', '2026_10_01-14-07-58.nwb')
#path = os.path.join(base,'C57BL6J','NWBs', '2026_10_01-15-04-50.nwb')

#for path in paths: 
data = Data(path)
data.build_rawFluo()
data.build_neuropil()
data.build_dFoF(neuropil_correction_factor=0)
data.build_visual_stim()

#[0 - center patch - 'protocol-center-patch' 
# 1 - drifting grating - 'protocol-natural-images'  #error in building protocol!! this was fixed for next ones 
# 2 - grey screen -      'protocol-grey-screen'
# 3 - natural images - 'protocol-natural-images' 
# 4 - looming        -  'protocol-looming'
protocol_id = 0
protocol_name= "static-patch"
color = 'greenyellow'
#color = 'lightskyblue'
Ep = EpisodeData(data, protocol_id=protocol_id, 
                quantities=['dFoF','rawFluo', 'neuropil','opto'], 
                prestim_duration=2,
                verbose=True)

#Average all ROIs
_ = plot_traces(Ep,
            title='%s,\nmean over all ROIs\n %s' % (data.filename, protocol_name), 
            average_by = "stimulus", LED_on = [],#opto_for_protocol,
            traces='raw_fluo', 
            color_led=color)

_ = plot_traces(Ep,
            title='%s,\nmean over all ROIs\n %s' % (data.filename, protocol_name), 
            average_by = "stimulus", LED_on = [],#opto_for_protocol,
            traces='dFoF',
            color_led=color)

plot_variation_dFoF(Ep, color_led=color)
'''
# select responsive ROIs
resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep, 
                                                data.nROIs, 
                                                alpha=0.05, 
                                                t_window = 1.0, 
                                                repetition_keys = ['repeat', 'contrast', 'radius'], 
                                                filtering_cond=None, 
                                                param=None, 
                                                param_type=None)
resp_roi_indexes = np.where(resp_roi)[0]
print(resp_roi_indexes)
random.seed(42)
_ = plot_traces(Ep,
            roi=resp_roi_indexes,       
            title='%s,\n mean over resp ROIs (%s/%s)\n %s' % (data.filename, len(resp_roi_indexes),data.nROIs,protocol_name), 
            average_by = 'stimulus', 
            color_led=color)

plot_variation_dFoF(Ep, 
                    roi=resp_roi_indexes,
                    color_led=color)
'''

''' #plot only few rois
# select responsive ROIs
resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep, 
                                                data.nROIs, 
                                                alpha=0.05, 
                                                t_window = 1.0, 
                                                repetition_keys = ['repeat', 'contrast', 'radius'], 
                                                filtering_cond=None, 
                                                param=None, 
                                                param_type=None)
resp_roi_indexes = np.where(resp_roi)[0]
print(resp_roi_indexes)
import random
random.seed(42)
#for roi in resp_roi_indexes[10:]:
for roi in random.sample(list(resp_roi_indexes), 10):
    _ = plot_traces(Ep,
                            roi = roi,
                            #roi=resp_roi_indexes,
                            title='%s, ROI %s,\n %s' % (data.filename, roi ,protocol_name), 
                #title='%s,\n mean over resp ROIs (%s/%s)\n %s' % (data.filename, len(resp_roi_indexes),data.nROIs,protocol_name), 
                average_by = "stimulus")

    plot_variation_dFoF(Ep, roi=roi)
'''

#%%
#for path in paths: 
# Looking at the neuropil and fluorescence time course
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-41-06.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb')
path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-01-46.nwb')
data = Data(path)
data.build_rawFluo()
data.build_neuropil()
data.build_dFoF(neuropil_correction_factor=0.0)
data.build_visual_stim()

#[0 - center patch - 'protocol-center-patch' 
# 1 - drifting grating - 'protocol-natural-images'  #error in building protocol!! this was fixed for next ones 
# 2 - grey screen -      'protocol-grey-screen'
# 3 - natural images - 'protocol-natural-images' 
# 4 - looming        -  'protocol-looming'
protocol_id = 2
protocol_name= "grey-screen"
Ep = EpisodeData(data, protocol_id=protocol_id, 
                quantities=['dFoF','rawFluo', 'neuropil','opto'], 
                prestim_duration=2,
                verbose=True)


# select responsive ROIs
resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep, 
                                                data.nROIs, 
                                                alpha=0.05, 
                                                t_window = 1.0, 
                                                repetition_keys = ['repeat', 'contrast', 'radius'], 
                                                filtering_cond=None, 
                                                param=None, 
                                                param_type=None)
resp_roi_indexes = np.where(resp_roi)[0]
print(resp_roi_indexes)
import random
random.seed(42)
#for roi in resp_roi_indexes[10:]:
for roi in random.sample(list(resp_roi_indexes), 10):
    _ = plot_traces(Ep,
                            roi = roi,
                            #roi=resp_roi_indexes,
                            title='%s, ROI %s,\n %s' % (data.filename, roi ,protocol_name), 
                #title='%s,\n mean over resp ROIs (%s/%s)\n %s' % (data.filename, len(resp_roi_indexes),data.nROIs,protocol_name), 
                average_by = "stimulus")

    plot_variation_dFoF(Ep, roi=roi)
        

#%%
'''
for i in range(20):
    fig, ax = pt.figure(\
            ax_scale=(1.5,0.95),
            axes=(1,1),
            wspace=0.5, hspace=1, top=4.5)
    
    pt.plot(ax=ax, x=Ep.t, y = Ep.opto[i])
    pt.plot(ax=ax, x=Ep.t, y = np.mean(Ep.dFoF[i], axis=0))
    pt.plot(ax=ax, x=Ep.t, y = np.mean(Ep.rawFluo[i], axis=0), color='red')
    ax.axvspan(0, Ep.time_duration[0], color='grey', alpha=0.3, lw=0, zorder=-10)
    #ax.axvspan(-1, Ep.time_duration[0] + 1, color='blue', alpha=0.3, lw=0, zorder=-10)
    #pt.set_plot(ax=ax,ylim =[-0.5,2])
    if opto_for_protocol[i] == True:

        ax.annotate("LED ON episode", xy=(1, 2))

        period = (Ep.t >= -0.5) & (Ep.t <= 1)

        if np.any(Ep.opto[i][period] < 0.1):
            ax.annotate("does not meet requirement", xy=(1.5, 1))
        else : 
            ax.annotate("OK", xy=(1, 1.7))


    if opto_for_protocol[i] == False:

        ax.annotate("LED OFF episode", xy=(1, 2))

        period = (Ep.t >= -1.5) & (Ep.t <= Ep.time_duration[0])

        if np.any(Ep.opto[i][period] > 0.1):
            ax.annotate("does not meet requirement", xy=(1.5, 1))
        else: 
            ax.annotate("OK", xy=(1, 1.7))
'''


'''
# select responsive ROIs
resp_roi, pos_roi, neg_roi = calc_responsiveness2(Ep, 
                                                  data.nROIs, 
                                                  alpha=0.05, 
                                                  t_window = 1.0, 
                                                  repetition_keys = ['repeat', 'contrast'], 
                                                  filtering_cond=None, 
                                                  param=None, 
                                                  param_type=None)
resp_roi_indexes = np.where(resp_roi)[0]
print(resp_roi_indexes)
_ = plot_traces(Ep,
                          roi=resp_roi_indexes,
            title='%s, mean over resp ROIs (%s/%s)\n %s' % (data.filename, len(resp_roi_indexes),data.nROIs,protocol_name), 
            average_by = None)

'''
# #for each ROI
#for roi in range(data.nROIs):
#    plot_traces(Ep, roi,
#                 title='%s, ROI #%i' % (data.filename, roi+1),
#                    average_by = 'stimulus')

#for roi in resp_roi_indexes:
#    plot_traces(Ep, roi,
#                 title='%s, ROI #%i' % (data.filename, roi+1), 
#                 average_by = 'stimulus')

#%% [markdown]
# ALL FILES
#%%
#all files
fig, ax = pt.figure(ax_scale=(1.2,1.2))
for i, f in enumerate(DATASET['files']):
    print(f)
    data = Data(f)
    data.build_neuropil()
    data.build_dFoF(neuropil_correction_factor=0.0)
    # _ = show_CaImaging_FOV(data, NL=4)
    meanImg, _ = get_FOV_image(data, 'meanImg')
    ax.plot(meanImg.mean(axis=1),
                color=cmap(i/len(DATASET['files'])))
    #ax.plot(meanImg.mean(axis=1)[0:50],
    #        color=cmap(i/len(DATASET['files'])))
    print(data.nROIs)
pt.set_plot(ax, 
            #yscale='log',
            xlabel='vertical pixels', ylabel='mean Img fluo.')
pt.bar_legend(ax, 
              label='incr. power',
              colormap=cmap)


#%%
#%% check ROI trace vs neuropil
data.build_dFoF(neuropil_correction_factor=0.0) #change this to accept all
data.build_neuropil()
cond = data.t_dFoF>10
roiIndices = np.arange(data.nROIs)
fig, AX = pt.figure((3,len(roiIndices)), ax_scale=(1.2,.8), wspace=0.6)
print(roiIndices)
for roi_i in roiIndices:
    pt.annotate(AX[roi_i][0], 'ROI #%i' % (1+roi_i), (0,1))
    #AX[roi_i][0].plot(data.t_dFoF[cond],
    #              data.neuropil[roi_i,:][cond],
    #              color='tab:red')
    AX[roi_i][0].plot(data.t_dFoF[cond], 
                  data.rawFluo[roi_i,:][cond],
                  color='tab:green')
    pt.set_plot(AX[roi_i][0], 
                xticks_labels=None if roi_i==2 else [],
                xlabel='time (s)' if roi_i==2 else '')
pt.annotate(AX[0][0], 'ROI fluo.   \n', (2,1.5), color='tab:green', ha='right')
pt.annotate(AX[0][0], 'neuropil   ', (1,1.45), color='tab:red', ha='right')
pt.annotate(AX[0][0], 'neuropil-subst.=0.0', (1,1.15), color='tab:green', ha='right')

data.build_dFoF(neuropil_correction_factor=0.7)
data.build_neuropil()
cond = data.t_dFoF>10
roiIndices = np.arange(data.nROIs)
for roi_i in roiIndices:
    AX[roi_i][1].plot(data.t_dFoF[cond],
                  data.dFoF[roi_i,:][cond],
                  color='tab:green')
    pt.set_plot(AX[roi_i][1], ylabel='$\Delta$F/F',
                xticks_labels=None if roi_i==2 else [],
                xlabel='time (s)' if roi_i==2 else '')
pt.annotate(AX[0][1], 'neuropil-subst.=0.7', (1,1.15), color='tab:green', ha='right')

data.build_dFoF(neuropil_correction_factor=1.0)
data.build_neuropil()
cond = data.t_dFoF>10
roiIndices = np.arange(data.nROIs)
for roi_i in roiIndices:
    AX[roi_i][2].plot(data.t_dFoF[cond],
                      data.dFoF[roi_i,:][cond],
                      color='tab:green')
    pt.set_plot(AX[roi_i][2], ylabel='$\Delta$F/F',
                xticks_labels=None if roi_i==2 else [],
                xlabel='time (s)' if roi_i==2 else '')
pt.annotate(AX[0][2], 'neuropil-subst.=1.0', (1,1.15), color='tab:green', ha='right')
