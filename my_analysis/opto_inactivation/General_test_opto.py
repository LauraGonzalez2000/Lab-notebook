# %%
import numpy as np
import sys, os

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.utils import plot_tools as pt
from physion.dataviz.imaging import show_CaImaging_FOV, get_FOV_image
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

from pathlib import Path

from scipy import stats

pt.set_style('manuscript')

#%%
#functions
def plot_effect(Ep,
                roi=None,
                title='', 
                cond_LED = None):
    
    fig, AX = pt.figure(\
        ax_scale=(1.5,0.95),
        axes=(len(Ep.varied_parameters['contrast']),2),
        wspace=0.5, hspace=1, top=4.5)

    fig.suptitle(title)

    if roi is None:
        dFoF = Ep.dFoF.mean(axis=1)
    else:
        dFoF = Ep.dFoF[:,roi,:]

    conditions = [('vis. stim', ~cond_LED),('vis. stim + OPTO', cond_LED)]
    contrasts = [0.5, 1]

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

            #pt.annotate(ax, label, (0, 1),ha='center',va='bottom')

            if i==1: 
                ax.axvspan(-1, 3, color='navy',alpha=0.2, lw=0, zorder=-10)
            
            ax.axvspan(0, 2,
                    color='grey',
                    alpha=0.3,
                    lw=0,
                    zorder=-10
                )
            ax.axis('off')

    pt.set_common_ylims(AX)

    pt.draw_bar_scales(
        AX[0][0],
        Ybar=0.2,
        Ybar_label='0.2$\\Delta$F/F',
        Xbar=1,
        Xbar_label='1s'
    )

    return fig

def plot_neuropil_vs_fluo(Ep, 
                          roi=None,
                          title='', LED_cond=None):

    fig, AX = pt.figure(ax_scale=(1.5,0.95),
                        axes=(len(Ep.varied_parameters['contrast']),2),
                        wspace=0.5, hspace=0.5, top=2.5)

    fig.suptitle(title)

    if roi is None:
        rawFluo = Ep.rawFluo.mean(axis=1)
        neuropil = Ep.neuropil.mean(axis=1)
    else:
        rawFluo = Ep.rawFluo[:,roi,:]
        neuropil = Ep.neuropil[:,roi,:]

    # baseline pre-level
    # rawFluo -= rawFluo[:,Ep.t<0].mean(axis=1)
    # neuropil -= neuropil[:,Ep.t<0].mean(axis=1)
    rawFluo = np.transpose(rawFluo.T-rawFluo[:,Ep.t<0].T.mean(axis=0))
    neuropil = np.transpose(neuropil.T-neuropil[:,Ep.t<0].T.mean(axis=0))

    conditions = [('vis. stim', ~LED_cond),('vis. stim + OPTO', LED_cond)]
    contrasts = [0.5, 1]

    for i, (label, led_cond) in enumerate(conditions):
        for j, c in enumerate(contrasts):

            ax = AX[i][j]

            trial_cond = (Ep.find_episode_cond('contrast', value=c)\
                          & led_cond)

            pt.plot(Ep.t,
                    rawFluo[trial_cond].mean(axis=0),
                    stats.sem(rawFluo[trial_cond], axis=0),
                    ax=ax, color='tab:green')
            pt.plot(Ep.t, 
                    neuropil[trial_cond,:].mean(axis=0),
                    stats.sem(neuropil[trial_cond], axis=0),
                    ax=ax, no_set=True, color='tab:red')

            pt.annotate(ax, f'c={c}\n {label}',(0.5, 1),ha='center',va='bottom')

            if i==1: 
                ax.axvspan(-1, 3, color='navy',alpha=0.2, lw=0, zorder=-10)
            
            ax.axvspan(0, 2,
                    color='grey',
                    alpha=0.3,
                    lw=0,
                    zorder=-10
                )
            ax.axis('off')

    pt.draw_bar_scales(AX[0][0],
                     Ybar=0.2, Ybar_label='0.2$\\Delta$F/F',
                    Xbar=1, Xbar_label='1s')
    
    return fig

# %%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.7}

#%%
index = 0
data = Data(filename=SESSIONS['files'][index], verbose=False)
data.build_dFoF(**dFoF_options)
data.build_running_speed()
data.build_facemotion()
data.build_pupil_diameter()

#%%
_ = show_CaImaging_FOV(data, NL=4)
_ = show_CaImaging_FOV(data, NL=4, 
                       roiIndex = np.arange(data.nROIs),
                       with_ROI_annotation=True)
meanImg, _ = get_FOV_image(data, 'meanImg')
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1), color='tab:green')
pt.set_plot(ax, 
            yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

#%%
index = 1
data = Data(filename=SESSIONS['files'][index], verbose=False)
data.build_dFoF(**dFoF_options)
data.build_running_speed()
data.build_facemotion()
data.build_pupil_diameter()

#%%
_ = show_CaImaging_FOV(data, NL=4)
_ = show_CaImaging_FOV(data, NL=4, 
                       roiIndex = np.arange(data.nROIs),
                       with_ROI_annotation=True)
meanImg, _ = get_FOV_image(data, 'meanImg')
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1), color='tab:green')
pt.set_plot(ax, 
            yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

# %%
data = Data(filename=SESSIONS['files'][0], verbose=False)
data.build_dFoF(neuropil_correction_factor=0.7)

fig, AX = pt.figure((3,data.nROIs), ax_scale=(1.2,.8), wspace=0.8)
cond = data.t_dFoF>3

for roi in range(data.nROIs):

    pt.annotate(AX[roi][0], 'ROI #%i' % (1+roi), (0,1))
    
    #neuropil and rawFluo
    AX[roi][0].plot(data.t_dFoF[cond],
                    data.neuropil[roi,:][cond],
                    color='tab:red')
    AX[roi][0].plot(data.t_dFoF[cond], 
                    data.rawFluo[roi,:][cond],
                    color='tab:green')
    
    pt.set_plot(AX[roi][0], 
                ylabel='Raw F',
                xticks_labels=None if roi==2 else [],
                xlabel='time (s)' if roi==2 else '')
    
    #dFoF neuropil 0.7 
    AX[roi][1].plot(data.t_dFoF[cond],
                    data.dFoF[roi,:][cond],
                    color='tab:green')

    pt.set_plot(AX[roi][1], 
                ylabel='$\Delta$F/F',
                xticks_labels=None if roi==2 else [],
                xlabel='time (s)' if roi==2 else '')
    
pt.annotate(AX[0][0], 'ROI fluo.   \n', (1,1), color='tab:green', ha='right')
pt.annotate(AX[0][0], 'neuropil   ', (1,1), color='tab:red', ha='right')
pt.annotate(AX[0][1], 'neuropil-subst.=0.7', (1,1), color='tab:green', ha='right')

data.build_dFoF(neuropil_correction_factor=1.0)

#dFoF neuropil 1.0
for roi in range(data.nROIs):
    AX[roi][2].plot(data.t_dFoF[cond],
                    data.dFoF[roi,:][cond],
                    color='tab:green')
    pt.set_plot(AX[roi][2], ylabel='$\Delta$F/F',
                xticks_labels=None if roi==2 else [],
                xlabel='time (s)' if roi==2 else '')
pt.annotate(AX[0][2], 'neuropil-subst.=1.0', (1,1), color='tab:green', ha='right')


# %%
index = 0
data = Data(SESSIONS['files'][index])
data.build_dFoF(neuropil_correction_factor=0.7)
Ep = EpisodeData(data, protocol_id=0, quantities=['dFoF', 'LED'])
LED_on = Ep.LED.mean(axis=1)>0 # LED "On" episode condition

#%%
for NEUROPIL_FACTOR in [0.7, 1.0]:
    data = Data(SESSIONS['files'][index])
    data.build_dFoF(neuropil_correction_factor=NEUROPIL_FACTOR)
    Ep = EpisodeData(data, protocol_id=0, quantities=['dFoF', 'LED'])
    LED_on = Ep.LED.mean(axis=1)>0 # LED "On" episode condition
    plot_effect(Ep, 
                title='%s\n **ALL ROIs (mean dFoF) **\n\n' % data.filename+\
                    'neuropil-substraction-factor=%.2f' % NEUROPIL_FACTOR, 
                cond_LED = LED_on)


#%%
NEUROPIL_FACTOR = 0.7
index = 1
data = Data(SESSIONS['files'][index])
data.build_dFoF(neuropil_correction_factor=NEUROPIL_FACTOR)
Ep = EpisodeData(data, protocol_id=0, quantities=['dFoF', 'LED'])
LED_on = Ep.LED.mean(axis=1)>0 # LED "On" episode condition
for roi in range(data.nROIs):
    plot_effect(Ep, roi=roi,
                title='%s, ROI #%i \n' % (data.filename, roi)+\
                    'neuropil-substraction-factor=%.2f' % NEUROPIL_FACTOR, 
                    cond_LED = LED_on)
    


# %% [markdown]
# # Looking at the neuropil and fluorescence time course

#%%
data.nROIs
#%%
#data = Data(f)
data.build_rawFluo()
data.build_neuropil()
data.build_dFoF(neuropil_correction_factor=0.0)
Ep = EpisodeData(data, protocol_id=0, 
                 quantities=['rawFluo', 'neuropil', 'dFoF', 'LED'])

#%%
for roi in range(data.nROIs):
    summary_stats = Ep.pre_post_statistics(episode_cond= ~LED_on,
                        response_args={'quantity':'dFoF', 'roiIndex':roi},
                        stat_test_props={},
                        repetition_keys=['repeat'],
                        nMin_episodes=1)
    if summary_stats['significant'].any():
        plot_neuropil_vs_fluo(Ep, roi,
                    title='%s, ROI #%i' % (data.filename, roi+1), 
                    LED_cond=LED_on)
    else:
        print(roi, summary_stats)

# %%
repeats = np.arange(len(Ep.index))
blank = (repeats%2==0)
light = (repeats%2==1)
stim = Ep.find_episode_cond(key='contrast', value=1)
for i in range(10):
    pt.plt.plot(Ep.rawFluo[stim & blank,3,:][i, :])

#%%
#EACH ROI
for roi in range(data.nROIs):
    plot_neuropil_vs_fluo(Ep, roi,
                 title='%s, ROI #%i' % (data.filename, roi+1), 
                 LED_cond=LED_on)

#%%
#ALL ROIs
_ = plot_neuropil_vs_fluo(Ep, 
            title='%s, mean over all ROIs' % data.filename, 
            LED_cond=LED_on)
# %%

