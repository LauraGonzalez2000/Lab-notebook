# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from utils_.General_overview_episodes import compute_high_arousal_cond

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

pt.set_style('manuscript')
#%%  FUNCTIONS

def plot_evoked_pattern(self,  
                        pattern_cond = [], 
                        quantity='rawFluo',
                        rois=None,
                        with_stim_inset=True,
                        with_mean_trace=False,
                        factor_for_traces=2,
                        raster_norm='full',
                        Tbar=1,
                        min_dFof_range=4,
                        ax_scale=(1.3,.3), 
                        axR=None, 
                        axT=None, 
                        behavior_split=False):
    # CHECK EPISODES HAVE THE SAME NUMBER OF TRIALS AND FILTER
    ep_s = []
    for i in range(len(self)):  #for each file
        resp = np.array(getattr(self[i], quantity))
        if resp.shape[0] == self[0].dFoF.shape[0]:
            ep_s.append(self[i])
        else:
            print(f"file {i} discarded because some trials are missing : {resp.shape[0]} instead of {self[0].dFoF.shape[0]} - way to fix this? ")

    # HANDLE DATA MISSING ##################
    if with_stim_inset and (ep_s[0].visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_stim_inset = False

    # SET FIGURE #################################################################
    n_stim = len(np.unique(ep_s[0].index)) #assumes all files have same stimuli
    if not behavior_split:
        n_cond = n_stim
    elif behavior_split: 
        n_cond = n_stim*2 #columns are doubled 

    #Calculate behavior condition for each file :
    HMcond_s = []
    for ep in ep_s:
        HMcond = compute_high_arousal_cond(ep, pre_stim=1, running_speed_threshold=0.05, metric="locomotion")
        HMcond = np.array(HMcond)
        HMcond_s.append(HMcond)

    #Calculate pattern condition for each stimulus
    Patterncond_s = []
    for stim_id in range(n_stim):
        pattern_cond = np.array([ep_s[0].index[i] == stim_id for i in range(len(ep_s[0].index))])
        Patterncond_s.append(pattern_cond)

    ####### initialize figure
    if axR is None:
        fig, axR = pt.figure(axes_extents= [[[1,3]] * n_cond],
                                    ax_scale=ax_scale,
                                    left=0.3,
                                    top=(12 if with_stim_inset else 1),
                                    right=3)
    else:
        fig = None
    
    # CALCULATE RESPONSE  
    resp_s = []
    for i in range(len(ep_s)):  #for each file
        resp = np.array(getattr(ep_s[i], quantity))

        for stim_id in range(n_stim):
            if not behavior_split:
                pattern_cond = Patterncond_s[stim_id]
                temp = resp[pattern_cond,:,:]
                resp_s.append(temp)
            
            elif behavior_split:
                HMcond = HMcond_s[i]
                pattern_cond = Patterncond_s[stim_id]
                final_cond = HMcond & pattern_cond
                temp = resp[final_cond,:,:]
                resp_s.append(temp)
                final_cond2= ~HMcond & pattern_cond
                temp2 = resp[final_cond2,:,:]
                resp_s.append(temp2)
           
    column_lists = [[] for _ in range(n_cond)] # column[i] will store the responses of this specific stimuli with behavioral condition if needed
    
    for i, resp in enumerate(resp_s):
        stim_idx = i % n_cond      # cycles through 0,1,...,n_stimuli-1
        column_lists[stim_idx].append(resp)

    #FILL IMAGE
    for i in range(len(column_lists)):
        print(" Column : ", i)
        # VISUAL STIM ###############################################################
        stim_idx = i
        stim_inset = pt.inset(axR[stim_idx], [0.2, 1.3, 0.6, 0.6])
       
        if behavior_split==False:
            stim = stim_idx
            ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
            axR[stim_idx].set_title(f"STIM {stim_idx+1}", y=1)
        else: 
            stim = stim_idx // 2
            state = "ACT" if stim_idx % 2 == 0 else "REST"
            ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
            axR[stim_idx].set_title(f"STIM {stim} {state}", y=1)

        

        # RASTER ######################################################################
        # mean response for raster
        resp = column_lists[i]
        mean_resp = [np.nanmean(r, axis=0) for r in resp]
        combined = np.concatenate(mean_resp, axis=0)
        
        if raster_norm=='full':
            combined = (combined-combined.min(axis=1).reshape(len(combined),1))
        else:
            pass

        #reorder neurons by similarity
        dist = pdist(combined, metric='correlation')
        Z = linkage(dist, method='average')
        order = leaves_list(Z)
        combined = combined[order, :]

        # Plot raster
        axR[stim_idx].imshow(combined,
                             cmap=pt.binary,
                             aspect='auto', interpolation='none',
                             vmin=0, vmax=2,
                             extent=(ep_s[0].t[0], ep_s[0].t[-1], 0, combined.shape[0]))

        pt.set_plot(axR[stim_idx], [], xlim=[ep_s[0].t[0], ep_s[0].t[-1]])
       
        pt.bar_legend(axR[stim_idx], 
                      colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
                      colormap=pt.binary,
                      bar_legend_args={},
                      label='n. $\\Delta$F/F',
                      bounds=None,
                      ticks = None,
                      ticks_labels=None,
                      no_ticks=False,
                      orientation='vertical')
        
    return fig


# MY_version ###############################################
##############################################################
##############################################################

#LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

data_s = []
for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    data = Data(filename,verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.init_visual_stim()
    data_s.append(data)
#%%
#protocols = ["static-patch",  "drifting-gratings", "Natural-Images-4-repeats"]
protocols = ["Natural-Images-4-repeats"]

ep_s_ = []
for protocol in protocols: 
    ep_s = []
    for data in data_s: 
        data.init_visual_stim()
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed'])
        print(len(data.visual_stim.experiment['index']), len(ep.protocol_cond_in_full_data))
        ep.init_visual_stim(data)
        ep_s.append(ep)
    ep_s_.append(ep_s)


#%%
ep = ep_s[0]
#%% 
########################################################################
##################### NOT SPLITTING BEHAVIOR ###########################
########################################################################
for p, protocol in enumerate(protocols):
    plot_evoked_pattern(ep_s_[p], quantity='dFoF', with_stim_inset=True, behavior_split=False)

#%%
########################################################################
###################### SPLITTING BEHAVIOR ##############################
########################################################################
for p, protocol in enumerate(protocols):
    plot_evoked_pattern(ep_s_[p], quantity='dFoF', with_stim_inset=True, behavior_split=True)



#%%
# OTHER
'''
protocol = "static-patch" 
#protocol = "drifting-gratings"
#protocol = "Natural-Images-4-repeats"

pre_stim = 1
ep = EpisodeData(data,
                 protocol_name=protocol,
                 quantities=['dFoF', 'running_speed'])
ep.init_visual_stim(data)

HMcond = compute_high_arousal_cond(ep, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")
pattern_cond = []

for index in range(len(np.unique(ep.index))):
    pattern_cond_temp = [True if ep.index[i]==index else False for i in range(len(ep.index))]
    pattern_cond_temp_act = pattern_cond_temp and HMcond
    pattern_cond_temp_rest = pattern_cond_temp and ~HMcond
    pattern_cond.append(pattern_cond_temp_act)
    pattern_cond.append(pattern_cond_temp_rest)

pattern_cond_ = list(pattern_cond)
## careful, pattern cond only taken with 1 file
'''

# PLOT INDIVIDUAL TRACES #######################################################
'''

#start from resp 

if rois is None:
    print(resp[0].shape[1])
    rois = np.random.choice(np.arange(len(combined)), 5, replace=False)

print(rois)


n_trials = np.sum(cond_s[stim_id])

if n_trials == 0:
    print(f"Skipping plotting for condition : no valid trials")
else:
    for ir, roi in enumerate(rois):
        roi_resp = combined[roi]
        roi_resp = roi_resp - roi_resp.mean()
        scale = max([min_dFof_range, np.max(roi_resp)])
        roi_resp /= scale

        # vertical line
        axT[stim_idx].plot([self[0].t[-1], self[0].t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

        if with_mean_trace:
            pt.plot(self[0].t, ir+roi_resp.mean(axis=0), 
                    sy=roi_resp.std(axis=0), ax=axT[stim_idx], no_set=True)

        # individual trial traces
        for iep in range(n_trials):
            axT[stim_idx].plot(self[0].t, ir+roi_resp[iep,:], 
                        color=pt.tab10(iep/(n_trials-1)), lw=.5)

'''
'''
for ir, r in enumerate(rois):
    roi_resp = resp_s[cond, r, :]
    roi_resp = roi_resp - roi_resp.mean()
    scale = max([min_dFof_range, np.max(roi_resp)])
    roi_resp /= scale
    axT[k].plot([self[0].t[-1], self[0].t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

    if with_mean_trace:
        pt.plot(self[0].t, ir+roi_resp.mean(axis=0), 
                sy=roi_resp.std(axis=0),ax=axT[k], no_set=True)
    
    for iep in range(np.sum(cond)):
        axT[k].plot(self[0].t, ir+roi_resp[iep,:], color=pt.tab10(iep/(np.sum(cond)-1)), lw=.5)

pt.set_plot(axT[k], [], xlim=[self[0].t[0], self[0].t[-1]])
pt.draw_bar_scales(axT[k], Xbar=Tbar, Xbar_label=str(Tbar)+'s', Ybar=1e-12)

pt.bar_legend(axT[k], 
                X=np.arange(np.sum(cond)),
                colorbar_inset=dict(rect=[1.1,1-.8/factor_for_traces,
                                        .04,.8/factor_for_traces], facecolor=None),
            colormap=pt.jet,
            label='trial ID',
            no_ticks=True,
            orientation='vertical')

if vse is not None:
    for t in [0]+list(vse['t'][vse['t']<self[0].visual_stim.protocol['presentation-duration']]):
        axR[k].plot([t,t], axR[k].get_ylim(), 'r-', lw=0.3)
        axT[k].plot([t,t], axT[k].get_ylim(), 'r-', lw=0.3)
'''
#pt.set_style('manuscript')


#%% DEMO #####################################################
##############################################################
##############################################################


'''
def plot_evoked_pattern_demo(self, 
                        pattern_cond, 
                        quantity='rawFluo',
                        rois=None,
                        with_stim_inset=True,
                        with_mean_trace=False,
                        factor_for_traces=2,
                        raster_norm='full',
                        Tbar=1,
                        min_dFof_range=4,
                        ax_scale=(1.3,.3), axR=None, axT=None):

    resp = np.array(getattr(self, quantity))

    if rois is None:
        rois = np.random.choice(np.arange(resp.shape[1]), 5, replace=False)

    if (axR is None) or (axT is None):
        fig, [axR, axT] = pt.figure(axes_extents=[[[1,3]],
                                                  [[1,int(3*factor_for_traces)]]], 
                                    ax_scale=ax_scale, left=0.3,
                                    top=(12 if with_stim_inset else 1),
                                    right=3)
    else:
        fig = None

    
    if with_stim_inset and (self.visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_screen_inset = False
   
    if with_stim_inset:
        stim_inset = pt.inset(axR, [0.2,1.3,0.6,0.6])
        self.visual_stim.plot_stim_picture(np.flatnonzero(pattern_cond)[0],
                                           ax=stim_inset,
                                           vse=True)
        vse = self.visual_stim.get_vse(np.flatnonzero(pattern_cond)[0])

    # mean response for raster
    mean_resp = resp[pattern_cond,:,:].mean(axis=0)
    if raster_norm=='full':
        mean_resp = (mean_resp-mean_resp.min(axis=1).reshape(resp.shape[1],1))
    else:
        pass

    # raster
    axR.imshow(mean_resp,
               cmap=pt.binary,
               aspect='auto', interpolation='none',
               vmin=0, vmax=2, 
               #origin='lower',
               extent = (self.t[0], self.t[-1],
                         0, resp.shape[1]))

    pt.set_plot(axR, [], xlim=[self.t[0], self.t[-1]])
    # pt.annotate(axR, '1 ', (0,0), ha='right', va='center', size='small')
    # pt.annotate(axR, '%i ' % resp.shape[1], (0,1), ha='right', va='center', size='small')
    # pt.annotate(axR, 'ROIs', (0,0.5), ha='right', va='center', size='small', rotation=90)
    # pt.annotate(axR, 'n=%i trials' % np.sum(pattern_cond), (self.t[-1], resp.shape[1]),
    #             xycoords='data', ha='right', size='x-small')

    # raster_bar_inset = pt.inset(axR, [0.2,1.3,0.6,0.6])
    pt.bar_legend(axR, 
                  colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
                  colormap=pt.binary,
                  bar_legend_args={},
                  label='n. $\\Delta$F/F',
                  bounds=None,
                  ticks = None,
                  ticks_labels=None,
                  no_ticks=False,
                  orientation='vertical')

    for ir, r in enumerate(rois):
        roi_resp = resp[pattern_cond, r, :]
        roi_resp = roi_resp-roi_resp.mean()
        scale = max([min_dFof_range, np.max(roi_resp)])
        roi_resp /= scale
        axT.plot([self.t[-1], self.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

        if with_mean_trace:
            pt.plot(self.t, ir+roi_resp.mean(axis=0), 
                    sy=roi_resp.std(axis=0),ax=axT, no_set=True)
        # pt.annotate(axT, 'roi#%i' % (r+1), (self.t[0], ir), xycoords='data',
        #             #rotation=90, 
        #             ha='right', size='xx-small')
        for iep in range(np.sum(pattern_cond)):
            axT.plot(self.t, ir+roi_resp[iep,:], color=pt.tab10(iep/(np.sum(pattern_cond)-1)), lw=.5)

    # pt.annotate(axT, '1$\\Delta$F/F', (self.t[-1], 0), xycoords='data',
    #             rotation=90, size='small')
    pt.set_plot(axT, [], xlim=[self.t[0], self.t[-1]])
    pt.draw_bar_scales(axT, Xbar=Tbar, Xbar_label=str(Tbar)+'s', Ybar=1e-12)

    pt.bar_legend(axT, X=np.arange(np.sum(pattern_cond)),
                  colorbar_inset=dict(rect=[1.1,1-.8/factor_for_traces,
                                            .04,.8/factor_for_traces], facecolor=None),
                  colormap=pt.jet,
                  label='trial ID',
                  no_ticks=True,
                  orientation='vertical')

    if vse is not None:
        for t in [0]+list(vse['t'][vse['t']<self.visual_stim.protocol['presentation-duration']]):
            axR.plot([t,t], axR.get_ylim(), 'r-', lw=0.3)
            axT.plot([t,t], axT.get_ylim(), 'r-', lw=0.3)
            
    return fig

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs-test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
index = 0
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)
data.build_dFoF()
data.init_visual_stim()
protocol = "static-patch"
ep = EpisodeData(data,
                 protocol_name=protocol,
                 quantities=['dFoF'])
ep.init_visual_stim(data)

for index in range(len(np.unique(ep.index))):
    pattern_cond = [True if ep.index[i]==index else False for i in range(len(ep.index))]
    plot_evoked_pattern_demo(ep, pattern_cond,
                        quantity='dFoF')
    pt.plt.show()
'''