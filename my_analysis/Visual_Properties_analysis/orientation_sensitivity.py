# %% [markdown]
# STUDIES ORIENTATION SENSITIVITY ON DIFFERENT DATASETS

#%%
import os, sys

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

sys.path += ['../']
from utils_.contrast_angle_sensitivity_methods import\
            generate_figures, \
            create_PDF, \
            generate_figures_GROUP, \
            create_group_PDF

# Variables : 
test = "angle"
#%% ##############################################################################################################
########################################       STATIC GRATINGS      ##############################################
######################################## 8 ORIENTATIONS 2 contrasts ##############################################
########################################            NDNF            ##############################################
##################################################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_8ori2contrasts')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}
data_s_ori = []
nROIS = 0
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_ori.append(data)
    print(idx, data.protocols)
    nROIS += data.nROIs

#%%
nROIS
#%% [markdown]
## All individual files
#%%
#dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures(data_s_ori, subplots_n=16)
#create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type='NDNF')
#%% [mardown]
## GROUPED ANALYSIS
#%%
fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = generate_figures_GROUP(data_s_ori, subplots_n=16, test=test, ylim=[-0.3,0.2])
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, 'NDNF', test=test)
