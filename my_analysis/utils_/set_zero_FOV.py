#%%
#For each subject, define the zero and write it in the dataTable
#%%
# %%
import os
from utils_.window_FOV import pick_zero
from pathlib import Path
#%%
%matplotlib qt
base_ = os.path.join(Path("E:/"), 
                     'DATA', 
                     'In_Vivo_experiments', 
                     'opto', 
                     'vision-survey-opto', 
                     'Thy1GCaMP-NDNF-Cre', 
                     'Processed')

zero = pick_zero(
    window_image_file = os.path.join(base_, 'intrinsic_img', '2026_08_26', '16-37-45', 'vasculature.tif'))  # window image
    #zero_image_file = os.path.join(base_, '2026_11_04', '15-46-20'))   # raw 2P session folder (finds the *zero* image)