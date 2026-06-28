# %%
import numpy as np
import os, sys
sys.path += ['../../physion/src']

import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data,\
      scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

pt.set_style('manuscript')

from pathlib import Path

# %%
def compute(filename, protocol = 'drifting-gratings'):
    data = Data(filename)
    data.build_dFoF()
    ep = EpisodeData(data, 
                 quantities = ['dFoF'],
                 protocol_name=protocol, 
                 prestim_duration=2)
    print(len(ep.dFoF[0][0]))
    significant = {'positive':[], 'negative':[], 'non-responsive':[]}
    responses = {'positive':[], 'negative':[], 'non-responsive':[]}
    
    for i in range(ep.dFoF.shape[1]): # loop over ROIs
  
        stat_pos = ep.stat_test_for_evoked_responses(
                    response_args=dict(roiIndex=i),
                    interval_pre=[-1,0],
                    interval_post=[1,2],
                    sign='positive')
        
        stat_neg = ep.stat_test_for_evoked_responses(
                    response_args=dict(roiIndex=i),
                    interval_pre=[-1,0],
                    interval_post=[1,2],
                    sign='negative')

        if stat_pos.significant(threshold=0.05):
            # fig, ax = pt.figure()
            # ax.plot(ep.t, ep.dFoF[:,i,:].mean(axis=0))
            significant['positive'].append(i)
            responses['positive'].append(\
                    ep.dFoF[:,i,:].mean(axis=0))
        
        elif stat_neg.significant(threshold=0.05):
            # fig, ax = pt.figure()
            # ax.plot(ep.t, ep.dFoF[:,i,:].mean(axis=0))
            significant['negative'].append(i)
            responses['negative'].append(\
                    ep.dFoF[:,i,:].mean(axis=0))
        else: 
            significant['non-responsive'].append(i)
            responses['non-responsive'].append(\
                    ep.dFoF[:,i,:].mean(axis=0))

    return significant, responses, data, ep

def analyze(filename, protocol="drifting-gratings"):

    significant, responses, data, ep = compute(filename, protocol=protocol)

    fig, AX = pt.figure((3,3), 
                        ax_scale=(1,1.5), 
                        top=3, 
                        wspace=1.5, 
                        hspace=1.5)

    for  j, sign in enumerate(['positive', 'negative', 'non-responsive']):

        AX[0][j].set_title('%i %s ROIs' %\
            (len(significant[sign]), sign))
        
        for resp in responses[sign]:
            AX[0][j].plot(ep.t, resp-resp[ep.t<0].mean())
            AX[0][j].axvspan(0, 2, color='grey', alpha=0.2)
        
        #average single ROIs
        avg_resp = np.nanmean(responses[sign], axis=0)

        if not np.isnan(avg_resp).all():
            baseline = avg_resp[ep.t<0].mean()
            trace_norm = avg_resp - baseline
            sem_trace  = np.nanstd(responses[sign], axis=0) / np.sqrt(len(responses[sign]))
            AX[1][j].plot(ep.t, avg_resp-avg_resp[ep.t<0].mean())
            AX[1][j].axvspan(0, 2, color='grey', alpha=0.2)
            AX[1][j].fill_between(ep.t,
                                trace_norm - sem_trace,
                                trace_norm + sem_trace,
                                color="grey",
                                alpha=0.3)
            AX[1][j].set_title(f"Average\n {sign} ROIs")
        
    
    #average response (all ROIs)
    temp = [np.nanmean(responses[sign], axis=0)for sign in ['positive', 'negative', 'non-responsive']]
    avg_resp_all = np.nanmean([x for x in temp if not np.isscalar(x)], axis=0)
    baseline = avg_resp_all[ep.t<0].mean()
    trace_all_norm = avg_resp_all - baseline

    data_ = []
    for sign in responses:
        if len(responses[sign]) > 1:
            data_.extend(responses[sign])
    sem_all_trace = np.nanstd(data_, axis=0) / np.sqrt(len(data_))

    AX[2][1].plot(ep.t, trace_all_norm)
    AX[2][1].axvspan(0, 2, color='grey', alpha=0.2)
    AX[2][1].fill_between(ep.t,
                          trace_all_norm - sem_all_trace,
                          trace_all_norm + sem_all_trace,
                          color="grey",
                          alpha=0.3)
    AX[2][1].set_title("Average all ROIs")
    
    
    AX[2][0].axis('off')
    AX[2][2].axis('off')


    ratio = len(significant['negative'])/\
        (len(significant['negative'])+len(significant['positive']))
    fig.suptitle( '%s, %s : \n ratio = %.1f \n protocol %s' % (\
        data.metadata['subject_ID'],
        os.path.basename(filename), ratio, protocol))
    
    return ratio 

#%%
##########################################################
# YANN's dataset #########################################
##########################################################
datafolder = os.path.join(Path("E:/"), 
                          'DATA', 
                          'In_Vivo_experiments',
                          'NDNF-old-protocol', 
                          'NDNF-WT-Dec-2022',
                          'NWBs')
DATASET = scan_folder_for_NWBfiles(datafolder)

#%%
protocol = 'static-patch'
results = {'file':[], 'ratio':[]}
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
protocol = 'drifting-gratings'
results = {'file':[], 'ratio':[]}
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
protocol = "Natural-Images-4-repeats"
results = {'file':[], 'ratio':[]}
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
