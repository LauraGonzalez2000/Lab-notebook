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
from pathlib import Path
pt.set_style('manuscript')

#%%
#LOAD DATA
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence
#%%
index = 0 #file number
data = Data(filename=SESSIONS['files'][index], verbose = False)
data.init_visual_stim()

###################### ACCESS image from ep.visual_stim.get_image() ##############################
#%%
## drifting gratings ! 
protocol = "drifting-gratings"
ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed', 'rawFluo'])
ep.init_visual_stim(data) 

#%%
ep.visual_stim
image =  ep.visual_stim.get_image(0)
image =  np.rot90(image, k=1)
print("image pixel values (720x1280) \n", image)
fig, ax = pt.figure()
ax.axis("off")
ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)

#%%
#natural images !! ISSUE, all values are 666.
protocol = "Natural-Images-4-repeats"
ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed', 'rawFluo'])
ep.init_visual_stim(data) 

#%%
image = ep.visual_stim.get_image(1)
image = np.rot90(image, k=1)
print("image pixel values (720x1280) \n", image)
fig, ax = pt.figure()
ax.axis("off")
ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)

#%%
############################ ACCESS image from data.visual_stim.get_image(index) ########################################
index_episode = 10  #1 for natural images
image = data.visual_stim.get_image(index=index_episode)
image =  np.rot90(image, k=1)
fig, ax = pt.figure()
ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)
ax.axis('off')

# %%
def get_imgs_of_episodes(ep, data):
    cond = \
        (data.visual_stim.experiment['protocol_id']==ep.protocol_id) &\
        (data.visual_stim.experiment['repeat']==0)
    cond = \
        (data.visual_stim.experiment['protocol_id']==4) &\
        (data.visual_stim.experiment['Image-ID']==5)
     
    IMGS = []
    for i in np.arange(len(data.visual_stim.experiment['repeat']))[cond]:
        print(i)
        data.visual_stim.plot_stim_picture(i)
        #IMGS.append(data.visual_stim.get_image(i))

    return IMGS
         
IMGS = get_imgs_of_episodes(ep, data)
for img in IMGS:
    fig, ax = pt.figure()
    ax.axis("off")
    ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)


#%% my data ##############################################################
##########################################################################
##########################################################################

datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','Vision-survey', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

data_s_sofia = []
for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    data = Data(filename,verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.init_visual_stim() #initializes visual stim (7 protocols (experiments) per file)
    data_s_sofia.append(data)

#%%
protocols = ["drifting-grating", "Natural-Images-4-repeats", 'static-patch']
#protocols = ["drifting-grating", 
#             "Natural-Images-4-repeats",
#             'static-patch', 
#             'looming-stim',
#             'moving-dots']

ep_s_ = []
for protocol in protocols: 
    ep_s = []
    for i, data in enumerate(data_s_sofia): 
        print("File ", i)
        ep = EpisodeData(data, protocol_name=protocol, quantities=['dFoF', 'running_speed', 'rawFluo'])
        ep.init_visual_stim(data) 
        ep_s.append(ep)
    ep_s_.append(ep_s)

#%%
#drifting
for index in range(10):
    ep = ep_s_[0][0]
    ep.visual_stim
    image =  ep.visual_stim.get_image(index)
    image =  np.rot90(image, k=1)
    print(image)
    fig, ax = pt.figure()
    ax.axis("off")
    ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)

#%%
#natural
ep = ep_s_[1][0]
ep.visual_stim
image =  ep.visual_stim.get_image(1)
image =  np.rot90(image, k=1)
print(image)
fig, ax = pt.figure()
ax.axis("off")
ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)

#%%
for index in range(len(data.visual_stim.experiment['protocol_id'])):
    print("number episode ",index, " / ", len(data.visual_stim.experiment['protocol_id']) )
    print("ID visual stimuli :",data.visual_stim.experiment['protocol_id'][index], "\n")
    image = data.visual_stim.get_image(index=index)
    image =  np.rot90(image, k=1)
    fig, ax = pt.figure()
    ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)
    ax.axis('off')
    
#%%

data = data_s_yann[5]
#%%
indexes = []
n_tot_ep = len(data.visual_stim.experiment['protocol_id'])
for index in range(0,n_tot_ep):
    if data.visual_stim.experiment['protocol_id'][index]==4: #4 for Yann
        #if data.visual_stim.experiment['Image-ID'][index]==4:
            indexes.append(index)
#%%
for index in indexes[:15]:#n_tot_ep]:
    #print("Image ID", data.visual_stim.experiment['Image-ID'][index])
    print("radius:",data.visual_stim.experiment['radius'][index])
    image = data.visual_stim.get_image(index=index)
    #NIarray = data.visual_stim.experiment['protocol_id'][index].NIarray
    #print(NIarray)
    image =  np.rot90(image, k=1)
    fig, ax = pt.figure()
    ax.imshow(image, cmap=pt.plt.cm.binary_r, vmin=0, vmax=1)
    ax.axis('off')
    
#%%
#np.sum[data.visual_stim.experiment['Image-ID'][index]==3 for index in indexes]
print("Total natural images : ", len(indexes))
for i in range(1,6):
    print("for Image Id ", i, "there is ", np.sum([data.visual_stim.experiment['Image-ID'][index] == i for index in indexes]))

