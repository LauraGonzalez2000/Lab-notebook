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

from physion.dataviz.imaging import show_CaImaging_FOV, get_FOV_image
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from pathlib import Path

from physion.analysis.episodes.build import EpisodeData

cmap = pt.get_linear_colormap('lightgreen', 'darkgreen')

from scipy import stats
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
        ax.axis('off')
        # photo-stim
        if i%2==1:
            ax.fill_between([-1,2],y0*np.ones(2),y1*np.ones(2), 
                                color='tab:blue', alpha=.3, lw=0)

    pt.draw_bar_scales(AX[0],
                    Ybar=0.2, Ybar_label='0.2$\\Delta$F/F',
                    Xbar=1, Xbar_label='1s')
    
    return fig

def plot_neuropil_vs_fluo(Ep, 
                          roi=None, 
                          title=''):

    
    LED_on = Ep.opto.mean(axis=1) > 0
    print("led on ", LED_on)

    keys = list(Ep.varied_parameters.keys())
    n_keys = len(keys)
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
                        wspace=0.5, hspace=0.8, top=1.5)
    fig.suptitle(title)

    if roi is None:
        rawFluo = Ep.rawFluo.mean(axis=1)
        neuropil = Ep.neuropil.mean(axis=1)
    else:
        rawFluo = Ep.rawFluo[:, roi, :]
        neuropil = Ep.neuropil[:, roi, :]

    # Baseline subtraction
    baseline = Ep.t < 0
    print(rawFluo)
    rawFluo = rawFluo - rawFluo[:, baseline].mean(axis=1, 
                                                  keepdims=True)
    neuropil = neuropil - neuropil[:, baseline].mean(axis=1, 
                                                     keepdims=True)


    for i in range(n_cols):
        
        # value of first varied parameter
        c = Ep.varied_parameters[keys[0]][i // 2]
        c_cond = Ep.find_episode_cond(keys[0],
                                      value=c)

        # LED vs blank
        if i % 2 == 0:
            LED = ~LED_on
            text = ''
        else:
            LED = LED_on
            text = 'LED'

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
                label = 'a=%.2f\nc=%.2f' % (c, s) #generalize

            elif n_keys == 2:
                cond = c_cond
                label = 'c=%.2f' % c
            elif n_keys ==1:
                cond = Ep.find_episode_cond()
                label = ''

            # Add LED condition
            cond = cond & LED
            if i % 2 == 1:
                ax.axvspan(-1, Ep.time_duration[0]+1,color='tab:blue',
                           alpha=.3,lw=0)

            print(cond)
            pt.plot(Ep.t,
                    rawFluo[cond, :].mean(axis=0),
                    ax=ax,
                    no_set=True,
                    color='tab:green')

            pt.plot(Ep.t,
                    neuropil[cond, :].mean(axis=0),
                    ax=ax,
                    no_set=True,
                    color='tab:red')

            pt.annotate(ax,
                        text,
                        (1, 1),
                        ha='right',
                        va='top')

            if label:
                pt.annotate(ax,
                            label,
                            (0.5, 0),
                            va='top',
                            ha='center')

            # Stimulus shading
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

    pt.annotate(last_ax,
                'neuropil ',
                (0, 1),
                ha='right',
                color='tab:red')

    pt.annotate(last_ax,
                ' ROI fluo',
                (0, 1),
                ha='left',
                color='tab:green')

    pt.draw_bar_scales(first_ax,
                        Xbar=1,
                        Xbar_label='1s')

    return fig

# %%
#stg old config
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03', 'NWBs_')
#path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03','NWBs_', '2025_10_03-12-02-05.nwb')
#path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Other','test_opto','2025_10_03','NWBs_', '2025_10_03-14-48-12.nwb')


base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto')
base2 = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'ffSG-8ori-2contrasts-opto')

#stg on NDNF and GCaMP on PYR
#datafolder = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_28-15-45-15.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_28-17-15-05.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-11-00-02.nwb')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_09_02-11-56-52.nwb')

#ArchT on NDNF and GCaMP on NDNF
#datafolder = os.path.join(base,'NDNF-Cre','NWBs')
#datafolder = os.path.join(base2,'NDNF-Cre','NWBs')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-28-15.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-57-29.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_01-14-26-23.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_02-10-50-31.nwb')
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_09_03-12-12-18.nwb')
#path = os.path.join(base2,'NDNF-Cre','NWBs', '2026_08_17-12-18-40.nwb')
#path = os.path.join(base2,'NDNF-Cre','NWBs', '2026_08_17-14-17-42.nwb')


#ArchT on NDNF and GCaMP on PYR
datafolder = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs')
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-13-30-18.nwb')
path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_31-15-35-23.nwb')


DATASET = scan_folder_for_NWBfiles(datafolder)
data = Data(path)


#%% show FOV and pixel intensity distribution
_ = show_CaImaging_FOV(data, NL=4)
#_ = show_CaImaging_FOV(data, NL=10, roiIndex=12,
#                       fig_args=dict(ax_scale=(1.4,2.4)))
_ = show_CaImaging_FOV(data, NL=4, 
                       roiIndex = np.arange(data.nROIs))
roiIndices = np.arange(data.nROIs)
# roiIndices = [0,11,24] 
_ = show_CaImaging_FOV(data, NL=4, 
                       with_ROI_annotation=True,
                       roiIndex=roiIndices)
meanImg, _ = get_FOV_image(data, 'meanImg')
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1), color='tab:green')
pt.set_plot(ax, 
            # yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1)[0:50], color='tab:green')
pt.set_plot(ax, 
            # yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

# to add : this but for an image without opto and an image with opto (or an average)

#%% check ROI trace vs neuropil
data.build_dFoF(neuropil_correction_factor=0.0) #change this to accept all
data.build_neuropil()
cond = data.t_dFoF>10
roiIndices = np.arange(data.nROIs)
fig, AX = pt.figure((3,len(roiIndices)), ax_scale=(1.2,.8), wspace=0.6)
print(roiIndices)
for roi_i in roiIndices:
    pt.annotate(AX[roi_i][0], 'ROI #%i' % (1+roi_i), (0,1))
    AX[roi_i][0].plot(data.t_dFoF[cond],
                  data.neuropil[roi_i,:][cond],
                  color='tab:red')
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



#%% # Looking at the neuropil and fluorescence time course
data.build_rawFluo()
data.build_neuropil()
data.build_dFoF(neuropil_correction_factor=0.0)
Ep = EpisodeData(data, protocol_id=2, 
                 quantities=['rawFluo', 'neuropil', 'dFoF', 'opto'], 
                 verbose=True)

#Average ROIs
_ = plot_neuropil_vs_fluo(Ep,
            title='%s, mean over all ROIs' % data.filename)

# #for each ROI
#for roi in range(data.nROIs):
#    plot_neuropil_vs_fluo(Ep, roi,
#                 title='%s, ROI #%i' % (data.filename, roi+1))

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
