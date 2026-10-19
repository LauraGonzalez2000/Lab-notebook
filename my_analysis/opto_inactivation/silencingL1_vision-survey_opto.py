# %%
import numpy as np
import sys, os

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

from pathlib import Path
from scipy import stats

pt.set_style('manuscript')

#%%
#functions
def plot_effect(Ep,
                roi=None,
                title=''):
    
    fig, AX = pt.figure(\
        ax_scale=(1.5,0.95),
        axes=(3,4),
        wspace=0.5, hspace=1, top=4.5)
    
    fig2, AX2 = pt.figure(\
        ax_scale=(1.5,0.95),
        axes=(1,1),
        wspace=0.5, hspace=1, top=4.5)

    fig.suptitle(title)

    if roi is None:
        dFoF = Ep.dFoF.mean(axis=1)
    else:
        dFoF = Ep.dFoF[:,roi,:]

    LED_on = Ep.opto.mean(axis=1)>0
    conditions = [('vis. stim', ~LED_on),('vis. stim + OPTO', LED_on)]
    contrasts = [0.5, 1]

    amplitudes = []
    amplitudes_sem = []
    for i, (label, led_cond) in enumerate(conditions):
        for j, c in enumerate(contrasts):

            ax = AX[i][j]

            trial_cond = (Ep.find_episode_cond('contrast', value=c)\
                          & led_cond)
            
            pt.plot(Ep.t,
                    dFoF[trial_cond].mean(axis=0),
                    stats.sem(dFoF[trial_cond], axis=0),
                    ax=ax)

            pt.annotate(ax, f'c={c}\n {label}',(0.5, 1),ha='center',va='bottom')

            if i==1: 
                ax.axvspan(-1, 3, color='navy',alpha=0.2, lw=0, zorder=-10)
            
            ax.axvspan(0, 2, color='grey', alpha=0.3, lw=0, zorder=-10)
            #ax.axis('off')
            bsl_start= 0
            bsl_end = 3000
            resp_start = 3000
            resp_end = 5000

            baseline = dFoF[trial_cond].mean(axis=0)[bsl_start:bsl_end].mean(axis=0)
            response = dFoF[trial_cond].mean(axis=0)[resp_start:resp_end].mean(axis=0)
            amplitudes.append(response-baseline)

          
            baseline_trials = dFoF[trial_cond][:, bsl_start:bsl_end].mean(axis=1)
            response_trials = dFoF[trial_cond][:, resp_start:resp_end].mean(axis=1)
            amplitudes_sem.append((response_trials - baseline_trials).std() / np.sqrt(len(response_trials)))

         
            ax.axhline(xmin=3/8,
                       xmax=5/8,
                       y = response, 
                       color="red")
            
            ax.axhline(xmin=0,
                       xmax=3/8,
                       y = baseline, 
                       color="blue")
        
    pt.bar(y=[amplitudes[0],amplitudes[2],amplitudes[1],amplitudes[3]],
           x = [0, 1.2, 2.8, 4],
           sy=[amplitudes_sem[0],amplitudes_sem[2],amplitudes_sem[1],amplitudes_sem[3]],
           ax=AX2, 
           COLORS=["darkgray", "slateblue", "dimgrey", "midnightblue"])
    
    pt.set_plot(ax=AX2, 
                num_xticks=4,
                xticks=[0, 1.2, 2.8, 4],
                xticks_labels=['c=0.5',
                               'c=0.5 \n opto',
                               'c=1.0',
                               'c=1.0 \n opto'], 
                ylabel='Δ ΔF/F', 
                xticks_rotation=0)


    pt.set_common_ylims(AX)

    #pt.draw_bar_scales(AX[0][0], Ybar=0.2, Ybar_label='0.2$\\Delta$F/F',
    #                   Xbar=1, Xbar_label='1s')
    
    
    return fig

# %%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto','NDNF-Cre','NWBs')

SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.7}

data_s = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running()
    data.build_facemotion()
    data.build_pupil()
    data_s.append(data)
    print(idx, data.protocols)


#%% AVERAGE ALL ROIS
index = -1
NEUROPIL_FACTOR = 0.7
data = Data(SESSIONS['files'][index])
data.build_dFoF(neuropil_correction_factor=NEUROPIL_FACTOR)
data.build_visual_stim()
protocol = 'protocol-grey-screen'
#[0 - center patch - 'protocol-center-patch' 
# 1 - drifting grating - 'protocol-natural-images'  #error in building protocol!! this was fixed for next ones 
# 2 - grey screen -      'protocol-grey-screen'
# 3 - natural images - 'protocol-natural-images' 
# 4 - looming        -  'protocol-looming'
protocol_id = 0
Ep = EpisodeData(data, protocol_id = protocol_id, quantities=['dFoF', 'opto'], verbose=True)

#%%
total_length = len(data.visual_stim.experiment['protocol_id'])
theoretical_opto = (np.arange(total_length) % 2 == 1)

#%%
# Ensure your protocol list is a NumPy array for element-wise comparison
protocol_ids = np.array(data.visual_stim.experiment['protocol_id'])

# 2. Create a boolean mask (True where the protocol is 1, False elsewhere)
protocol_mask = (protocol_ids == protocol_id)

# 3. Apply the mask to your theoretical_opto array
opto_for_protocol = theoretical_opto[protocol_mask]

print(opto_for_protocol)
#%%
print(Ep.opto)
Episodes_LED_ok = []

for i in range(60):
    fig, ax = pt.figure(\
            ax_scale=(1.5,0.95),
            axes=(1,1),
            wspace=0.5, hspace=1, top=4.5)
    
    pt.plot(ax=ax, x=Ep.t, y = Ep.opto[i])
    pt.plot(ax=ax, x=Ep.t, y = np.mean(Ep.dFoF[i], axis=0))
    ax.axvspan(0, Ep.time_duration[0], color='grey', alpha=0.3, lw=0, zorder=-10)
    #ax.axvspan(-1, Ep.time_duration[0] + 1, color='blue', alpha=0.3, lw=0, zorder=-10)
    pt.set_plot(ax=ax,ylim =[-0.5,2])
    if opto_for_protocol[i]==True:
        ax.annotate("LED ON episode",xy=(1,2))
    if opto_for_protocol[i]==False:
            ax.annotate("LED OFF episode",xy=(1,2))

#%%
fig = plot_effect(Ep, 
            title='%s\n **ALL ROIs (mean dFoF) **\n\n' % data.filename+\
                'neuropil-substraction-factor=%.2f' % NEUROPIL_FACTOR)

#%% ALL ROIS ONE BY ONE
NEUROPIL_FACTOR = 0.7
data = Data(SESSIONS['files'][index])
data.build_dFoF(neuropil_correction_factor=NEUROPIL_FACTOR)
Ep = EpisodeData(data, protocol_id=0, quantities=['dFoF', 'opto'])
for roi in range(data.nROIs):
    fig = plot_effect(Ep, roi=roi,
                title='%s, ROI #%i \n' % (data.filename, roi)+\
                    'neuropil-substraction-factor=%.2f' % NEUROPIL_FACTOR)
    