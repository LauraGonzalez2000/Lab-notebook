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

cmap = pt.get_linear_colormap('lightgreen', 'darkgreen')

import h5py


#%%
def plot_FOV(samples, title):
    fig, ax = pt.figure(ax_scale=(1.2,1.2))
    mean_im = f['data'][samples, :, :].mean(axis=0)
    vmin, vmax = np.percentile(mean_im, [1, 99])
    pt.plt.imshow(mean_im, cmap = pt.get_linear_colormap('k', 'g'), vmin=vmin, vmax=vmax)
    pt.set_plot(ax=ax, title=title)
    ax.axis('off')
    pt.plt.colorbar()
    return fig, ax 

def plot_distribution(samples, zoom=False):
    mean_im = f['data'][samples, :, :].mean(axis=0)
    fig, ax = pt.figure(ax_scale=(1.2,1.2))
    if zoom: 
        ax.plot(mean_im.mean(axis=1)[0:50], color='tab:green')
    else:
        ax.plot(mean_im.mean(axis=1), color='tab:green')
    pt.set_plot(ax, 
                # yscale='log',
                title='(max. LED power)',
                xlabel='vertical pixels', ylabel='mean Img fluo.')
    return fig, ax
#%%

base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'vision-survey-opto')
#LED M450LP2 -> 450 nm (blue)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-41-06.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_06', '14-41-06', 'h5-10062026--004', 'Ch2-Green-plane0.h5'))
#900mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_09_02-11-56-52.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_09_02', '11-56-52', 'h5-09022026--002', 'Ch2-Green-plane0.h5'))
#1000mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_08_26-15-34-27.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_08_26', '15-34-27', 'h5-08262026-007', 'Ch2-Green-plane0.h5'))

#LED M455L3 -> 455 nm (blue)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-11-38-44.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_06', '11-38-44', 'h5-10062026--002', 'Ch2-Green-plane0.h5'))
#900mA
path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_05-16-58-19.nwb')
f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_05', '16-58-19', 'h5-10052026-1134-004', 'Ch2-Green-plane0.h5'))
#1000mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-14-01-46.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_06', '14-01-46', 'h5-10062026--003', 'Ch2-Green-plane0.h5'))


#LED M590L4 -> 590 nm (orange)
#700mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-15-41-49.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_06', '14-01-46', 'h5-10062026--003', 'Ch2-Green-plane0.h5'))
#900mA #find another
#path = os.path.join(base,'NDNF-Cre','NWBs', '2026_08_27-14-28-15.nwb')
#f = h5py.File(os.path.join(base,'NDNF-Cre','Processed', '2026_08_27', '14-28-15', 'h5-08272026-004', 'Ch2-Green-plane0.h5'))
#1000mA
#path = os.path.join(base,'Thy1GCaMP-NDNF-Cre','NWBs', '2026_10_06-16-16-41.nwb')
#f = h5py.File(os.path.join(base,'Thy1GCaMP-NDNF-Cre','Processed', '2026_10_06', '16-16-41', 'h5-10062026--006', 'Ch2-Green-plane0.h5'))


#control - NO OPSIN - LED M590L4 -> 590 nm (orange) - 900mA
#path = os.path.join(base,'C57BL6J','NWBs', '2026_10_02-17-18-59.nwb')
#f = h5py.File(os.path.join(base,'C57BL6J','Processed', '2026_10_02', '17-18-59', 'h5-10022026-1644-001', 'Ch2-Green-plane0.h5'))


data = Data(path)
data.build_rawFluo()
opto_sampling_rawFluo = data.build_opto(specific_time_sampling=data.t_rawFluo)

#################################################################################
#################################################################################
#%% FOV mean image
_ = show_CaImaging_FOV(data, NL=4)
_ = show_CaImaging_FOV(data, NL=4, 
                       roiIndex = np.arange(data.nROIs))
roiIndices = np.arange(data.nROIs)

meanImg, _ = get_FOV_image(data, 'meanImg')

#pixel intensity distribution
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1), color='tab:green')
pt.set_plot(ax, 
            #yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

#zoom
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(meanImg.mean(axis=1)[0:50], color='tab:green')
pt.set_plot(ax, 
            #yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

#%% blank
few_samples_blank = np.arange(len(data.t_rawFluo))[opto_sampling_rawFluo==0][:1000]
fig, ax = plot_FOV(samples = few_samples_blank, title = 'Blank average')
fig, ax = plot_distribution(samples = few_samples_blank)
fig, ax = plot_distribution(samples = few_samples_blank, zoom=True)

#%% opto
few_samples_opto = np.arange(len(data.t_rawFluo))[opto_sampling_rawFluo==1][:1000]
fig, ax = plot_FOV(samples = few_samples_opto, title = 'Opto average')
fig, ax = plot_distribution(samples = few_samples_opto)
fig, ax = plot_distribution(samples = few_samples_opto, zoom=True)

#%% diff opto - blank
few_samples_opto = np.arange(len(data.t_rawFluo))[opto_sampling_rawFluo==1][:1000]
mean_im_opto = f['data'][few_samples_opto, :, :].mean(axis=0)

few_samples_blank = np.arange(len(data.t_rawFluo))[opto_sampling_rawFluo==0][:1000]
mean_im_blank = f['data'][few_samples_blank, :, :].mean(axis=0)

mean_im_diff = mean_im_opto - mean_im_blank

fig, ax = pt.figure(ax_scale=(1.2,1.2))
vmin, vmax = np.percentile(mean_im_opto, [1, 99])
pt.plt.imshow(
    mean_im_diff,
    cmap='coolwarm',
    vmin=-vmax,
    vmax=vmax
)
pt.plt.colorbar()
pt.set_plot(ax,
            title='opto-blank')
ax.axis('off')

fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(mean_im_diff.mean(axis=1), color='tab:green')# pt.get_linear_colormap('k', 'g'))
pt.set_plot(ax, 
            # yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')
fig, ax = pt.figure(ax_scale=(1.2,1.2))
ax.plot(mean_im_diff.mean(axis=1)[0:50], color='tab:green')
pt.set_plot(ax, 
            # yscale='log',
            title='(max. LED power)',
            xlabel='vertical pixels', ylabel='mean Img fluo.')

# %%
