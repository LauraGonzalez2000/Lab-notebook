# %% [markdown]
# # Generate PNG summary

#%%
import os, sys

import numpy as np

sys.path += ['../../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.intrinsic.tools import *
from physion.intrinsic.analysis import RetinotopicMapping
from physion.intrinsic import tools as intrinsic_analysis

import matplotlib.pylab as plt
from PIL import Image
sys.path += ['..']
from PDF_layout import PDF4
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

#%%
def figure_to_array(fig):
            """Convert a matplotlib Figure to a NumPy array"""
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            fig_arr = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            return fig_arr

def generate_figures(dataFolder, segmentation_params):
    # fig 1 :vasculature picture
    imVasc = np.array(Image.open(os.path.join(dataFolder, 'vasculature.tif')))
    fig1, ax = pt.figure(ax_scale=(2,5))
    ax.imshow(imVasc**1, cmap=plt.cm.gray) 
    plt.axis('off');

    maps = np.load(os.path.join(dataFolder, 'raw-maps.npy') , 
                allow_pickle=True).item()
    #fig 2 : altitude map
    fig2 = intrinsic_analysis.plot_retinotopic_maps(maps, map_type='altitude');

    #fig 3 : azimuth map
    fig3 = intrinsic_analysis.plot_retinotopic_maps(maps, map_type='azimuth');

    #fig 4, 5 : sign maps; power maps
    data = intrinsic_analysis.build_trial_data(maps)
    data['vasculatureMap'] = imVasc[::int(imVasc.shape[0]/data['aziPosMap'].shape[0]),\
                                    ::int(imVasc.shape[1]/data['aziPosMap'].shape[1])]
    
    data['params'] = segmentation_params
    trial = RetinotopicMapping.RetinotopicMappingTrial(**data)
    trial.processTrial(isPlot=False) # TURN TO TRUE TO VISUALIZE THE SEGMENTATION STEPS
    
    fig4, AX4 = plt.subplots(2, 3, figsize=(6, 4))

    AX4[0, 0].imshow(trial.altPosMap)
    AX4[0, 1].imshow(trial.aziPosMap)
    AX4[0, 2].imshow(trial.signMap)

    AX4[1, 0].imshow(trial.altPosMapf)
    AX4[1, 1].imshow(trial.aziPosMapf)
    AX4[1, 2].imshow(trial.signMapf)

    titles = [
    "Altitude Pos", "Azimuth Pos", "Sign",
    "Altitude Pos \nfiltered", "Azimuth Pos \nfiltered", "Sign \nfiltered"
    ]
    for ax, title in zip(AX4.flat, titles):
        ax.set_title(title)
        ax.axis("off")

    fig5, AX5 = plt.subplots(1, 2, figsize=(6, 4))

    AX5[0].imshow(trial.altPowerMap)
    AX5[1].imshow(trial.aziPowerMap)
    
    titles = ["Altitude Power Map", "Azimuth Power Map"]
    for ax, title in zip(AX5.flat, titles):
        ax.set_title(title)
        ax.axis("off")
    
    #patches
    fig6, ax = pt.figure(ax_scale=(2,5))
    h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                    plotaxis=ax)
    ax.imshow(imVasc, cmap=pt.plt.cm.gray, 
            vmin=imVasc.min(), vmax=imVasc.max(), 
            extent=[*ax.get_xlim(), *ax.get_ylim()])
    h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                    plotaxis=ax)
    ax.axis('off');

    #center
    fig7, ax = pt.figure(ax_scale=(2,5))
    m = 0*maps['altitude-retinotopy']
    cond = ( np.abs(maps['altitude-retinotopy'])<10) &\
            ( np.abs(maps['azimuth-retinotopy'])<10)
    try: 
        # NOT WORKING
        centerV1 = trial.finalPatches['patch01'].getCenter()
        plt.plot([centerV1[1]], [centerV1[0]], 'ro', 
                alpha=0.3, ms=25)
        m[~cond] = 1

        h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                        alpha=0, plotaxis=ax)
        ax.imshow(imVasc, cmap=plt.cm.gray, 
                vmin=imVasc.min(), vmax=imVasc.max(), 
                extent=[*ax.get_xlim(), *ax.get_ylim()])
        ax.axis('off')
        ax.axis('equal');
    except: 
        print("No patches found or issue")
    

    fig1 = figure_to_array(fig1)
    fig2 = figure_to_array(fig2)
    fig3 = figure_to_array(fig3)
    fig4 = figure_to_array(fig4)
    fig5 = figure_to_array(fig5)
    fig6 = figure_to_array(fig6)
    fig7 = figure_to_array(fig7)
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, segmentation_params, cell_type):
    try: 
        pdf1 = PDF4()
 
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, segmentation_params)
     
        fig_p1 = pdf1.fig
       
        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Window_Summary_PDF/ID{dict_annotation['Subject_ID']}_window_summary.pdf'
      
        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")

        print("Window PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual window PDF file : {e}")

    return 0
######################################################################################################

#%%
base_folder = os.path.join(os.path.expanduser('~'),'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch3','Processed', 'intrinsic_img', '2026_01_16')

'''
segmentation_params={'phaseMapFilterSigma': 8.,
                        'signMapFilterSigma': 12.,
                        'signMapThr': 0.55,
                        'eccMapFilterSigma': 10.,
                        'splitLocalMinCutStep': 15.,
                        'mergeOverlapThr': 1,
                        'closeIter': 3,
                        'openIter': 3,
                        'dilationIter': 15,
                        'borderWidth': 1,
                        'smallPatchThr': 100,
                        'visualSpacePixelSize': 0.5,
                        'visualSpaceCloseIter': 15,
                        'splitOverlapThr': 1.3}

segmentation_params={'phaseMapFilterSigma': 2,
                    'signMapFilterSigma': 1.5,
                    'signMapThr': 0.5,
                    'eccMapFilterSigma': 10.,
                    'splitLocalMinCutStep': 5.,
                    'mergeOverlapThr': 0.2,
                    'closeIter': 3,
                    'openIter': 3,
                    'dilationIter': 15,
                    'borderWidth': 1,
                    'smallPatchThr': 100,
                    'visualSpacePixelSize': 0.5,
                    'visualSpaceCloseIter': 15,
                    'splitOverlapThr': 1.2}
'''
segmentation_params={'phaseMapFilterSigma': 1.5,
                    'signMapFilterSigma': 2,
                    'signMapThr': 0.5,
                    'eccMapFilterSigma': 10.,
                    'splitLocalMinCutStep': 10.,
                    'mergeOverlapThr': 0.2,
                    'closeIter': 3,
                    'openIter': 3,
                    'dilationIter': 15,
                    'borderWidth': 1,
                    'smallPatchThr': 100,
                    'visualSpacePixelSize': 0.5,
                    'visualSpaceCloseIter': 15,
                    'splitOverlapThr': 1.2}


# Loop over all subfolders (each expected to be a timestamp like "15-09-29", etc.)
for subfolder in sorted(os.listdir(base_folder)):
    
    datafile = os.path.join(base_folder, subfolder)
    print("datafile : ", datafile)

    metadata_path = os.path.join(datafile, 'metadata.json')
    with open(metadata_path, 'r') as f:
            metadata = json.load(f)   # Load JSON into a Python dict
    subject = metadata.get('subject', 'unknown')


    print(f"Processing file: {datafile}")

    df = pd.read_excel(os.path.join(os.path.expanduser('~'),'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch2', "DataTable.xlsx"))
    recordings_per_subject_t = (df.groupby('subject')['time'].apply(list).to_dict())
    recordings_per_subject_d = (df.groupby('subject')['day'].apply(list).to_dict())
    recordings_t = recordings_per_subject_t.get(subject, [])
    recordings_d = recordings_per_subject_d.get(subject, [])


    dict_annotation = {
            'name': f' {datafile.split('\\')[-2]} - {datafile.split('\\')[-1]}',
            'Subject_ID': subject, 
            'Recordings': [recordings_d, recordings_t]}
    
    fig1, fig2, fig3, fig4, fig5, fig6, fig7 = generate_figures(dataFolder=datafile, segmentation_params=segmentation_params)

    create_PDF(dict_annotation=dict_annotation, fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7, segmentation_params=segmentation_params, cell_type='NDNF')
