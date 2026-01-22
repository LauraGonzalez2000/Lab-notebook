# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

sys.path += ['..']
from utils_.General_overview_episodes import compute_high_arousal_cond

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

pt.set_style('manuscript')
#%%  FUNCTIONS

def plot_evoked_pattern(data, 
                        EP_s,  
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
                        behavior_split=False,
                        protocol=''):
    # CHECK EPISODES HAVE THE SAME NUMBER OF TRIALS AND FILTER - HANDLE MISSING DATA
    ep_s = []
    for i in range(len(EP_s)):  #for each file
        resp = np.array(getattr(EP_s[i], quantity))
        if resp.shape[0] == EP_s[0].dFoF.shape[0]:
            ep_s.append(EP_s[i])
        else:
            print(f"file {i} discarded because some trials are missing : {resp.shape[0]} instead of {EP_s[0].dFoF.shape[0]} - way to fix this? ")

    if with_stim_inset and (ep_s[0].visual_stim is None):
        print('\n [!!] visual stim of episodes was not initialized  [!!]  ')
        print('    --> screen_inset display desactivated ' )
        with_stim_inset = False

   
    nRois = np.sum([getattr(EP_s[i], quantity).shape[1] for i in range(len(EP_s))])
    #print("nROIS : ", nRois)
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
                             right=3, 
                             figsize=(12, nRois * 0.15))
        
    else:
        fig = None
    
    # CALCULATE RESPONSE  
    resp_s = []
    for i in range(len(ep_s)):  #for each file

        resp = np.array(getattr(ep_s[i], quantity))
        #print("resp", resp.shape)

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
    order_mantained = []
    for i in range(len(column_lists)):
        print(" Column : ", i)
        # VISUAL STIM ###############################################################
        stim_idx = i
        stim_inset = pt.inset(axR[stim_idx], [0.2, 1.3, 0.6, 0.6])
       
        if behavior_split==False:
            stim = stim_idx
            print(protocol)
            if protocol != "Natural-Images-4-repeats":
                ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
                axR[stim_idx].set_title(f"STIM {stim_idx+1}", y=1)

            elif protocol == "Natural-Images-4-repeats":
                protocol_id = 4
                prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
                im_cond = np.array(data.visual_stim.experiment['Image-ID'])==stim+1
                cond =  prot_cond  & im_cond
                iStim = np.flatnonzero(cond)[0] # first index with this stim cond
                print(iStim)
                image =  data.visual_stim.get_image(iStim)
                #pt.matrix(image, ax = stim_inset)

                image =  np.rot90(image, k=1)
                stim_inset.imshow(image, cmap=pt.plt.cm.binary_r,vmin=0, vmax=1)
                stim_inset.axis('off')

        else: 
            stim = stim_idx // 2
            state = "ACT" if stim_idx % 2 == 0 else "REST"
            if protocol != "Natural-Images-4-repeats":
                ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
                axR[stim_idx].set_title(f"STIM {stim} {state}", y=1)
            elif protocol == "Natural-Images-4-repeats":
              
                protocol_id = 4
                prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
                im_cond = np.array(data.visual_stim.experiment['Image-ID'])==stim+1
                cond =  prot_cond  & im_cond
                iStim = np.flatnonzero(cond)[0] # first index with this stim cond
                image =  data.visual_stim.get_image(iStim)
                axR[stim_idx].imshow(image)
                #pt.matrix(image)

        
        # RASTER ######################################################################
        # mean response for raster
        resp = column_lists[i]
        mean_resp = [np.nanmean(r, axis=0) for r in resp]
        combined = np.concatenate(mean_resp, axis=0)
        
        if raster_norm=='full':
            combined = (combined-combined.min(axis=1).reshape(len(combined),1))
        else:
            pass
        
        #reorder neurons by similarity (but keep same order between act and rest)
        dist = pdist(combined, metric='correlation')
        Z = linkage(dist, method='average')
        
        if not behavior_split: 
            order = leaves_list(Z)
        if behavior_split and state == "ACT":
            order = leaves_list(Z)
        elif behavior_split and state== "REST":
            order = order_mantained #not recalculated, taking the previous one (ACT of the same stim)

        combined = combined[order, :]
        order_mantained = order
        
        #print(len(combined))
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

for p, protocol in enumerate(protocols):
    ep_s = ep_s_[p]
    plot_evoked_pattern(data = data_s[0],
                        EP_s=ep_s, quantity='dFoF', with_stim_inset=True, behavior_split=False, protocol=protocol)



#%%
# MY_version ###############################################
##############################################################
##############################################################

#LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_rebuilt-test')
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
    data.init_visual_stim() #initializes visual stim (7 protocols (experiments) per file)
    data_s.append(data)

#%%
index = 0
data = Data(SESSIONS['files'][index])
data.init_visual_stim()

#%%
protocol_id = 4
image_id = 1
prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
im_cond = np.array(data.visual_stim.experiment['Image-ID'])==image_id
cond =  prot_cond  & im_cond
iStim = np.flatnonzero(cond)[0] # first index with this stim cond
image =  data.visual_stim.get_image(iStim)
pt.matrix(image)
#%%
protocol_id = 4
image_id = 5
prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
im_cond = np.array(data.visual_stim.experiment['Image-ID'])==image_id
cond =  prot_cond  & im_cond
    
print(cond)
iStim = np.flatnonzero(cond)[0] # first index with this stim cond
image =  data.visual_stim.get_image(iStim)
pt.matrix(image)


#%%
#protocols = ["static-patch",  "drifting-gratings", "Natural-Images-4-repeats"]
protocols = ["Natural-Images-4-repeats"]

ep_s_ = []

for protocol in protocols: 
    ep_s = []
    for i, data in enumerate(data_s): 
        print(i)
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed'])
        ep.init_visual_stim(data) 
        
        ep_s.append(ep)
    ep_s_.append(ep_s)
#%%
ep1 = ep_s[0]
data = data_s[0]
ep1.init_visual_stim(data)

fig, ax = pt.figure()
ep1.visual_stim.plot_stim_picture(episode=0, ax=ax, vse=True)
ep1.visual_stim.get_image()


#%%
protocols = ["static-patch"]

ep_s_ = []

for protocol in protocols: 
    ep_s = []
    for i, data in enumerate(data_s): 
        print(i)
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed'])
        ep.init_visual_stim(data) 
        
        ep_s.append(ep)
    ep_s_.append(ep_s)
#%%
ep2 = ep_s[0]
data = data_s[0]
ep2.init_visual_stim(data)

fig, ax = pt.figure()
ep2.visual_stim.plot_stim_picture(episode=3, ax=ax, vse=True)
ep2.visual_stim.get_image()

#%% 
########################################################################
##################### NOT SPLITTING BEHAVIOR ###########################
########################################################################
for p, protocol in enumerate(protocols):
    ep_s = ep_s_[p]
    plot_evoked_pattern(data = data_s[0], EP_s=ep_s, quantity='dFoF', with_stim_inset=True, behavior_split=False, protocol=protocol)

#%%
########################################################################
###################### SPLITTING BEHAVIOR ##############################
########################################################################
for p, protocol in enumerate(protocols):
    plot_evoked_pattern(data=data_s[0], EP_s=ep_s_[p], quantity='dFoF', with_stim_inset=True, behavior_split=True, protocol=protocol)



# %%

def plot_evoked_pattern(data, protocol,
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

    ep = EpisodeData(data,
                     quantities=[quantity],
                     protocol_id=data.get_protocol_id(protocol))
  
        
    ep.init_visual_stim(data)

    varied_param = list(ep.varied_parameters.keys())[0]
    param_values = ep.varied_parameters[varied_param]

    n_stim = len(param_values)
    if not behavior_split:
        n_cond = n_stim
    elif behavior_split: 
        n_cond = n_stim*2 #columns are doubled 


    fig, axR = pt.figure(axes=(len(param_values),1),
                         ax_scale=(1, 2.5),
                         left=0.3,
                         top=(12 if with_stim_inset else 1),
                         right=3)

    for p, param in enumerate(param_values):

        # STIM INSET ##################################################################
        stim_inset = pt.inset(axR[p], [0.2, 1.3, 0.6, 0.6])
        cond = ep.find_episode_cond(key=varied_param,
                                    value=param)
        iStim = np.flatnonzero(cond)[0]
        image =  ep.visual_stim.get_image(iStim)
        image =  np.rot90(image, k=1)
        stim_inset.imshow(image, cmap=pt.plt.cm.binary_r,vmin=0, vmax=1)
        stim_inset.axis('off')

        # RASTER ######################################################################
        # mean response for raster
        resp = column_lists[i]
        mean_resp = [np.nanmean(r, axis=0) for r in resp]
        combined = np.concatenate(mean_resp, axis=0)
        
        if raster_norm=='full':
            combined = (combined-combined.min(axis=1).reshape(len(combined),1))
        else:
            pass
        
        #reorder neurons by similarity (but keep same order between act and rest)
        dist = pdist(combined, metric='correlation')
        Z = linkage(dist, method='average')
        
        if not behavior_split: 
            order = leaves_list(Z)
        if behavior_split and state == "ACT":
            order = leaves_list(Z)
        elif behavior_split and state== "REST":
            order = order_mantained #not recalculated, taking the previous one (ACT of the same stim)

        combined = combined[order, :]
        order_mantained = order
        
        #print(len(combined))
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

    """

    column_lists = [[] for _ in range(n_cond)] # column[i] will store the responses of this specific stimuli with behavioral condition if needed
    
    for i, resp in enumerate(resp_s):
        stim_idx = i % n_cond      # cycles through 0,1,...,n_stimuli-1
        column_lists[stim_idx].append(resp)

    #FILL IMAGE
    order_mantained = []
    for i in range(len(column_lists)):
        print(" Column : ", i)
        # VISUAL STIM ###############################################################
        stim_idx = i
       
        if behavior_split==False:
            stim = stim_idx
            print(protocol)
            if protocol != "Natural-Images-4-repeats":
                ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
                axR[stim_idx].set_title(f"STIM {stim_idx+1}", y=1)

            elif protocol == "Natural-Images-4-repeats":
                protocol_id = 4
                prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
                im_cond = np.array(data.visual_stim.experiment['Image-ID'])==stim+1
                cond =  prot_cond  & im_cond
                iStim = np.flatnonzero(cond)[0] # first index with this stim cond
                print(iStim)
                image =  data.visual_stim.get_image(iStim)
                #pt.matrix(image, ax = stim_inset)

                image =  np.rot90(image, k=1)
                stim_inset.imshow(image, cmap=pt.plt.cm.binary_r,vmin=0, vmax=1)
                stim_inset.axis('off')

        else: 
            stim = stim_idx // 2
            state = "ACT" if stim_idx % 2 == 0 else "REST"
            if protocol != "Natural-Images-4-repeats":
                ep_s[0].visual_stim.plot_stim_picture(stim, ax=stim_inset, vse=True)
                axR[stim_idx].set_title(f"STIM {stim} {state}", y=1)
            elif protocol == "Natural-Images-4-repeats":
              
                protocol_id = 4
                prot_cond = np.array(data.visual_stim.experiment['protocol_id'])==protocol_id
                im_cond = np.array(data.visual_stim.experiment['Image-ID'])==stim+1
                cond =  prot_cond  & im_cond
                iStim = np.flatnonzero(cond)[0] # first index with this stim cond
                image =  data.visual_stim.get_image(iStim)
                axR[stim_idx].imshow(image)
                #pt.matrix(image)

        
        # RASTER ######################################################################
        # mean response for raster
        resp = column_lists[i]
        mean_resp = [np.nanmean(r, axis=0) for r in resp]
        combined = np.concatenate(mean_resp, axis=0)
        
        if raster_norm=='full':
            combined = (combined-combined.min(axis=1).reshape(len(combined),1))
        else:
            pass
        
        #reorder neurons by similarity (but keep same order between act and rest)
        dist = pdist(combined, metric='correlation')
        Z = linkage(dist, method='average')
        
        if not behavior_split: 
            order = leaves_list(Z)
        if behavior_split and state == "ACT":
            order = leaves_list(Z)
        elif behavior_split and state== "REST":
            order = order_mantained #not recalculated, taking the previous one (ACT of the same stim)

        combined = combined[order, :]
        order_mantained = order
        
        #print(len(combined))
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
    """
        
    return fig

# general python modules for scientific analysis
import os, sys
import numpy as np

sys.path += ['../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

#%%
protocol = "static-patch"#'Natural-Images-4-repeats'
quantity = 'rawFluo'

filename = 'C:\\Users\\laura.gonzalez\\DATA\\In_Vivo_experiments\\NDNF-WT-Dec-2022\\NWBs_rebuilt-test\\2022_12_14-13-27-41.nwb'
data = Data(filename)
data.init_visual_stim()
plot_evoked_pattern(data, protocol,
                    quantity=quantity,
                        with_stim_inset=True, behavior_split=False)
data.close()


#%%
# for debug
ep = EpisodeData(data,
                     quantities=[quantity],
                     protocol_id=data.get_protocol_id(protocol))
ep.init_visual_stim(data)
print("ep.visual_stim.blank_color\n",ep.visual_stim.blank_color)
print("ep.visual_stim.experiment['bg-color']\n", ep.visual_stim.experiment['bg-color'])

