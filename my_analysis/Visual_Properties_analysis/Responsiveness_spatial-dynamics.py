# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path += ['./../physion/src']

import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data,\
      scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.imaging import show_CaImaging_FOV

pt.set_style('manuscript')
from scipy import stats

from pathlib import Path

# %%
def compute(filename, protocol = 'drifting-gratings'):
    data = Data(filename)
    data.build_dFoF()
    ep = EpisodeData(data, 
                 quantities = ['dFoF'],
                 protocol_name=protocol)
    
    significant = {'positive':[], 'negative':[]}
    responses = {'positive':[], 'negative':[]}
    for i in range(ep.dFoF.shape[1]):

        for sign in ['positive', 'negative']:
            stat = ep.stat_test_for_evoked_responses(
                response_args=dict(roiIndex=i),
                interval_pre=[-1,0],
                interval_post=[1,2],
                sign=sign
            )
            if stat.significant(threshold=0.05):
                # fig, ax = pt.figure()
                # ax.plot(ep.t, ep.dFoF[:,i,:].mean(axis=0))
                significant[sign].append(i)
                responses[sign].append(\
                        ep.dFoF[:,i,:].mean(axis=0))
    return significant, responses, data, ep

def analyze(filename, protocol="drifting-gratings"):

    significant, responses, data, ep = compute(filename, protocol=protocol)

    fig, AX = pt.figure((4,1), ax_scale=(1,1.5), top=1.5)
    for ax1, ax2, sign in zip(AX[1::2], AX[::2],
                              ['positive', 'negative']):

        for i, resp in zip(significant[sign], responses[sign]):
            ax1.plot(ep.t, resp-resp[ep.t<0].mean())
        show_CaImaging_FOV(data, NL=4,
                        with_annotation=False,
                        with_ROI_annotation=False,
                        roiIndex=significant[sign],
                        ax=ax2)
        ax2.set_title('%i %s ROIs' %\
            (len(significant[sign]), sign))
    ratio = len(significant['negative'])/\
        (len(significant['negative'])+len(significant['positive']))
    fig.suptitle( '%s, %s : ratio = %.1f ' % (\
        data.metadata['subject_ID'],
        os.path.basename(filename), ratio))
    return ratio 

# %%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','2NatIm8contrasts', 'NDNF-Cre','NWBs')

filename = os.path.join(
    datafolder, '2026_04_21-15-11-13.nwb')

analyze(filename, protocol="drifting-grating")

# %%
DATASET = scan_folder_for_NWBfiles(datafolder)
results = {'file':[], 'ratio':[]}
protocol = 'drifting-gratings'
for f in DATASET['files']:
    try:
        ratio = analyze(f, protocol=protocol)
        results['ratio'].append(ratio)
        results['file'].append(os.path.basename(f))
    except BaseException as be:
        pass

# %%
fig, ax = pt.figure()
ax.hist(results['ratio'])
pt.set_plot(ax, xlabel='neg.-to-pos. resp.\nratio',
            ylabel='count')


#%%
##########################################################
# YANN's dataset #########################################
##########################################################
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs')
DATASET = scan_folder_for_NWBfiles(datafolder)

#%%
results = {'file':[], 'ratio':[]}
protocol = 'drifting-gratings'
for f in DATASET['files']:
    try:
        ratio = analyze(f, protocol='drifting-gratings')
       
        results['ratio'].append(ratio)
        results['file'].append(os.path.basename(f))
    except BaseException as be:
        pass

fig, ax = pt.figure()
ax.hist(results['ratio'])
pt.set_plot(ax, xlabel='neg.-to-pos. resp.\nratio',
            ylabel='count')

#%%
results = {'file':[], 'ratio':[]}
protocol = 'static-patch'
for f in DATASET['files']:
    try:
        ratio = analyze(f, protocol=protocol)
       
        results['ratio'].append(ratio)
        results['file'].append(os.path.basename(f))
    except BaseException as be:
        pass

fig, ax = pt.figure()
ax.hist(results['ratio'])
pt.set_plot(ax, xlabel='neg.-to-pos. resp.\nratio',
            ylabel='count')

#%%
results = {'file':[], 'ratio':[]}
protocol = "Natural-Images-4-repeats"
for f in DATASET['files']:
    try:
        ratio = analyze(f, protocol=protocol)
       
        results['ratio'].append(ratio)
        results['file'].append(os.path.basename(f))
    except BaseException as be:
        pass
fig, ax = pt.figure()
ax.hist(results['ratio'])
pt.set_plot(ax, xlabel='neg.-to-pos. resp.\nratio',
            ylabel='count')
# %%
