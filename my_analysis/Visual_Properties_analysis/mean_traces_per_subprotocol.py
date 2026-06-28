# %% [markdown]
# MEAN TRACES FOR DIFFERENT PROTOCOLS

#%%
import os, sys
from pathlib import Path

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

sys.path += ['../']
from utils_.General_overview_episodes import plot_dFoF_of_protocol

#%% #############################################################################
########################################       STATIC GRATINGS      #############
######################################## 2 ORIENTATIONS 8 contrasts #############
########################################            NDNF            #############
#################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 
                          'In_Vivo_experiments','Ori-contrasts', 
                          'NDNF-Cre', 'NWBs_8contrasts2ori')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con.append(data)
    print(idx, data.protocols)

#%%
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_con, 
                                          protocol=data_s_con[0].protocols[0], 
                                          ylim = [-0.15, 0.15])

#%%
#################################################################################
########################################       STATIC GRATINGS      #############
######################################## 2 ORIENTATIONS 8 contrasts #############
########################################     SST (CIBELE's DATA)    #############
#################################################################################
#%% SST CIBELE DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'SST-cells_WT_Adult_V1', 'NWBs_contrast')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con_SST = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_SST.append(data)

#%%
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_con_SST, 
                                          protocol=data_s_con[0].protocols[0], 
                                          ylim = [-0.15, 0.45])

#%%
#################################################################################
########################################     VISION SURVEY   ####################
########################################           NDNF          ################
#################################################################################
#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','Vision-survey', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con_vs = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_vs.append(data)
    print(idx, data.protocols)

#%%
protocol = "static-patch"
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_con_vs, 
                                          protocol=protocol, 
                                          ylim = [-0.15, 0.45])
#%%
protocol = "drifting-grating" #1 DIRECTION 3 CONTRASTS
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_con_vs, 
                                          protocol=protocol, 
                                          ylim = [-0.15, 0.45])


#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol','NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_oldNDNF = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_oldNDNF.append(data)
    print(idx, data.protocols)
#%%
protocol = "static-patch"
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_oldNDNF, 
                                          protocol=protocol, 
                                          ylim = [-0.5, 0.5])
fig_traces.savefig(f'avg_traces-{protocol}.png', format='png', dpi=600, transparent=True)
    
#%%
protocol = "drifting-gratings"
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_oldNDNF, 
                                          protocol=protocol, 
                                          ylim = [-0.5, 0.5])
fig_traces.savefig(f'avg_traces-{protocol}.png', format='png', dpi=600, transparent=True)
#%%
protocol = "Natural-Images-4-repeats"
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_oldNDNF, 
                                          protocol=protocol, 
                                          ylim = [-0.5, 0.5])
fig_traces.savefig(f'avg_traces-{protocol}.png', format='png', dpi=600, transparent=True)
#%%

#%%
#####################################################################################################################
########################################     NATURAL IMAGES      ####################################################
########################################   2 IMAGES 8 CONTRASTS  ####################################################
########################################           NDNF          ####################################################
#####################################################################################################################
#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','2NatIm8contrasts', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con_natIm = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_natIm.append(data)
    print(idx, data.protocols)

#%%
fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s_con_natIm, 
                                          protocol=data_s_con[0].protocols[0], 
                                          ylim = [-0.15, 0.45])