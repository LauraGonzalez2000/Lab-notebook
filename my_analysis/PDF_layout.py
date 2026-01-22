import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physion.utils  import plot_tools as pt
import matplotlib.pyplot as plt  #try to remove this

class PDF:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, ax  = pt.figure(figsize=(8.27, 11.69))
        ax.axis('off')
        self.AXs = {}

        # build the axes one by one
        X0, Y0, DX, DY = 0.12, 0.5, 6, 0.6
        self.AXs['Notes'] = self.create_panel([X0, Y0, DX, DY], 'Notes')
        self.AXs['Notes'].axis('off')

        X0, Y0, DX, DY = 0.12, 1.1, 6, 2
        self.AXs['FOV'] = self.create_panel([X0, Y0, DX, DY], 'Field of view')

        X0, Y0, DX, DY = 0.12, 3.1, 6, 2
        self.AXs['Traces_all'] = self.create_panel([X0, Y0, DX, DY], 'Example traces')

        X0, Y0, DX, DY = 0.12, 5.1, 6, 2
        self.AXs['Traces_zoom1'] = self.create_panel([X0, Y0, DX, DY], 'Traces zoom1')

        X0, Y0, DX, DY = 0.12, 7.1, 6, 2
        self.AXs['Traces_zoom2'] = self.create_panel([X0, Y0, DX, DY], 'Traces zoom2')

    def create_panel(self, coords, title=None):
        """ 
        coords: (x0, y0, dx, dy)
                from left to right
                from top to bottom (unlike matplotlib)
        """
        coords[1] = 1-coords[1]-coords[3]
        ax = pt.inset(self.fig, rect=coords)

        if title:
            ax.set_title(title, loc='left', pad=2, fontsize=8)
        return ax

    def fill_PDF(self, 
                 dict_annotation, 
                 image1, 
                 image2, 
                 image3, 
                 image4): 
        '''
        txt = (
            f"ID file: {dict_annotation['name']}\n"
            f"Protocol: {dict_annotation['protocol']}\n"
            f"Subject ID: {dict_annotation['Subject_ID']}\n"
        )

        self.fig.text(
            0.05,   # x position in figure coordinates (0=left, 1=right)
            0.95,   # y position in figure coordinates (0=bottom, 1=top)
            txt,
            va='top', ha='left',
            fontsize=10,
            wrap=True
        )
        '''
        
        for key in self.AXs:
            self.AXs[key].axis('off')

            
            if key=='Notes':
                self.AXs[key].axis('off')
                txt = (
                    f"ID file: {dict_annotation['name']}\n"
                    f"Protocol: {dict_annotation['protocol']}\n"
                    f"Subject ID: {dict_annotation['Subject_ID']}\n"
                )
                self.AXs[key].text(0, 1, txt, va='top', ha='left', fontsize=10, wrap=True)
                self.AXs[key].axis('off')
            
            if key=='FOV':
                self.AXs[key].imshow(image1)
                
            elif key=='Traces_all':
                self.AXs[key].imshow(image2)
                
            elif key=='Traces_zoom1':
                self.AXs[key].imshow(image3)

            elif key=='Traces_zoom2':
                self.AXs[key].imshow(image4)

class PDF2:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, ax  = pt.figure(figsize=(8.27, 11.69))
        ax.axis('off')
        self.AXs = {}

        X0, Y0, DX, DY = 0.12, 0.12, 6, 5
        self.AXs['Resp_protocol'] = self.create_panel([X0, Y0, DX, DY], 'dFoF for each protocol')

        X0, Y0, DX, DY = 0.12, 5.5, 6, 1.5
        self.AXs['barplots'] = self.create_panel([X0, Y0, DX, DY], 'Variation dFoF across protocols')

    def create_panel(self, coords, title=None):
            """ 
            coords: (x0, y0, dx, dy)
                    from left to right
                    from top to bottom (unlike matplotlib)
            """
            coords[1] = 1-coords[1]-coords[3]
            ax = pt.inset(self.fig, rect=coords)

            if title:
                ax.set_title(title, loc='left', pad=2, fontsize=8)
            return ax
    
    def fill_PDF2(self, 
                 image1, 
                 image2): 
        
        for key in self.AXs:
            self.AXs[key].axis('off')
            if key=='Resp_protocol':
                self.AXs[key].imshow(image1)
            
            elif key=='barplots':
                self.AXs[key].imshow(image2)
            
class PDF3:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, ax  = pt.figure(figsize=(8.27, 11.69))
        ax.axis('off')
        self.AXs = {}

        X0, Y0, DX, DY = 0.12, 0.5, 6, 1.5
        self.AXs['ex_ROI_positive_natIm'] = self.create_panel([X0, Y0, DX, DY], 'Example ROI positive response for natural images')

        X0, Y0, DX, DY = 0.12, 2.0, 6, 1.5
        self.AXs['ex_ROI_negative_natIm'] = self.create_panel([X0, Y0, DX, DY], 'Example ROI negative response for natural images')

        X0, Y0, DX, DY = 0.12, 3.5, 6, 1.5
        self.AXs['ex_ROI_ns_natIm'] = self.create_panel([X0, Y0, DX, DY], 'Example ROI non responsive for natural images')

        X0, Y0, DX, DY = 0.12, 5, 6, 1.5
        self.AXs['responsiveness'] = self.create_panel([X0, Y0, DX, DY], 'Responsiveness pie plots')

    def create_panel(self, coords, title=None):
            """ 
            coords: (x0, y0, dx, dy)
                    from left to right
                    from top to bottom (unlike matplotlib)
            """
            coords[1] = 1-coords[1]-coords[3]
            ax = pt.inset(self.fig, rect=coords)

            if title:
                ax.set_title(title, loc='left', pad=2, fontsize=8)
            return ax
    
    def fill_PDF3(self, 
                 image1, 
                 image2, 
                 image3, 
                 image4): 
        for key in self.AXs:
            self.AXs[key].axis('off')

            if key=='ex_ROI_positive_natIm':
                self.AXs[key].imshow(image1)

            elif key=='ex_ROI_negative_natIm':
                self.AXs[key].imshow(image2)
            
            elif key=='ex_ROI_ns_natIm':
                self.AXs[key].imshow(image3)

            elif key=='responsiveness':
                self.AXs[key].imshow(image4)

class PDF3_:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, ax  = pt.figure(figsize=(8.27, 11.69))
        ax.axis('off')
        self.AXs = {}

        X0, Y0, DX, DY = 0.12, 0.5, 6, 1.5
        self.AXs['fract_resp_ROIs'] = self.create_panel([X0, Y0, DX, DY], 'fraction responsive neurons (all ROIs)')

        X0, Y0, DX, DY = 0.12, 2.0, 6, 1.5
        self.AXs['fract_resp_means'] = self.create_panel([X0, Y0, DX, DY], 'fraction responsive neurons (means of sessions)')

    def create_panel(self, coords, title=None):
        """ 
        coords: (x0, y0, dx, dy)
                from left to right
                from top to bottom (unlike matplotlib)
        """
        coords[1] = 1-coords[1]-coords[3]
        ax = pt.inset(self.fig, rect=coords)

        if title:
            ax.set_title(title, loc='left', pad=2, fontsize=8)
        return ax
    
    def fill_PDF3(self, 
                 image1, 
                 image2): 
        for key in self.AXs:
            self.AXs[key].axis('off')

            if key=='fract_resp_ROIs':
                self.AXs[key].imshow(image1)

            elif key=='fract_resp_means':
                self.AXs[key].imshow(image2)
            
class PDF4:

    def __init__(self, structure_dict=None, debug=False):
        if structure_dict is None:
            structure_dict = {}

        self.debug = debug

        # --- Create A4 figure (in inches)
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.AXs = {}

        # --- Define panels using inch-based coordinates
        # Format: X0, Y0, DX, DY (in inches)
        self.AXs['Notes'] = self.create_panel([0.5, 0.5, 3, 2], 'Notes', hide_axis=True)
        self.AXs['Vasculature'] = self.create_panel([4.5, 0.5, 5, 2], 'Vasculature')
        self.AXs['Altitude_maps'] = self.create_panel([0.5, 2, 4, 4], 'Altitude maps')
        self.AXs['Azimuth_maps'] = self.create_panel([4.5, 2, 4, 4], 'Azimuth maps')

        self.AXs['Gradient'] = self.create_panel([0.5, 5, 4, 5], 'Gradient')
        self.AXs['Power']    = self.create_panel([4.5, 6, 4, 2.5], 'Power')

        self.AXs['Notes2'] = self.create_panel([0.5, 8.5, 3, 2], 'Segmentation Parameters', hide_axis=True)
        self.AXs['Patches'] = self.create_panel([3, 8.5, 4, 4], 'Patches')
        self.AXs['Center'] = self.create_panel([5.5, 8.5, 4, 3], 'Center')

        # Optional: add debugging border boxes
        if self.debug:
            for ax in self.AXs.values():
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(0.5)

        plt.tight_layout()  # helps avoid overlap when saving

    def create_panel(self, coords_in_inches, title=None, hide_axis=False):
        """
        coords_in_inches: [x0, y0, dx, dy] in inches
                          left-to-right, top-to-bottom (A4 layout convention)
        """
        fig_w, fig_h = self.fig.get_size_inches()
        x0, y0, dx, dy = coords_in_inches

        # Convert to bottom-up and normalized coordinates
        y0 = fig_h - y0 - dy
        rect_norm = [x0 / fig_w, y0 / fig_h, dx / fig_w, dy / fig_h]

        ax = self.fig.add_axes(rect_norm)
        if title:
            ax.set_title(title, loc='left', pad=2, fontsize=8)
        if hide_axis:
            ax.axis('off')

        return ax
    
    def fill_PDF(self, 
                 dict_annotation, 
                 image1, 
                 image2, 
                 image3, 
                 image4, 
                 image5, 
                 image6, 
                 image7, 
                 segmentation_params): 
        
        for key in self.AXs:
            self.AXs[key].axis('off')

            
            if key=='Notes':
                self.AXs[key].axis('off')

                recordings_text = "\n".join(f"• {day} — {time}" for day, time in zip(dict_annotation['Recordings'][0], dict_annotation['Recordings'][1]))

                txt = (
                    f"File: {dict_annotation.get('name', 'N/A')}\n"
                    f"Mouse ID: {dict_annotation.get('Subject_ID', 'N/A')}\n"
                    f"Recordings:\n{recordings_text}\n"
                )

                self.AXs[key].text(0.1, 0.9, txt, va='top', ha='left', fontsize=10, wrap=True)
                self.AXs[key].axis('off')

            
            if key=='Vasculature':
                self.AXs[key].imshow(image1)
            
            elif key=='Altitude_maps':
                self.AXs[key].imshow(image2)
                
            elif key=='Azimuth_maps':
                self.AXs[key].imshow(image3)

            elif key=='Gradient':
                self.AXs[key].imshow(image4)

            elif key=='Power':
                self.AXs[key].imshow(image5)

            elif key=='Notes2':
                self.AXs[key].axis('off')
                txt = (f"phaseMapFilterSigma: {segmentation_params['phaseMapFilterSigma']}\n"
                       f"signMapFilterSigma: {segmentation_params['signMapFilterSigma']}\n"
                       f"signMapThr: {segmentation_params['signMapThr']}\n"
                       f"eccMapFilterSigma: {segmentation_params['eccMapFilterSigma']}\n"
                       f"splitLocalMinCutStep: {segmentation_params['splitLocalMinCutStep']}\n"
                       f"mergeOverlapThr: {segmentation_params['mergeOverlapThr']}\n"
                       f"closeIter: {segmentation_params['closeIter']}\n"
                       f"openIter: {segmentation_params['openIter']}\n"
                       f"dilationIter: {segmentation_params['dilationIter']}\n"
                       f"borderWidth: {segmentation_params['borderWidth']}\n"
                       f"smallPatchThr: {segmentation_params['smallPatchThr']}\n"
                       f"visualSpacePixelSize: {segmentation_params['visualSpacePixelSize']}\n"
                       f"visualSpaceCloseIter: {segmentation_params['visualSpaceCloseIter']}\n"
                       f"splitOverlapThr: {segmentation_params['splitOverlapThr']}\n")
                self.AXs[key].text(0.1, -0.3, txt, va='bottom', ha='left', fontsize=10, wrap=True)
                self.AXs[key].axis('off')

            elif key=='Patches':
                self.AXs[key].imshow(image6)

            elif key=='Center':
                self.AXs[key].imshow(image7)

class PDF_angle_contrast:

    def __init__(self, structure_dict=None, debug=False):
        if structure_dict is None:
            structure_dict = {}

        self.debug = debug

        # --- Create A4 figure (in inches)
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.AXs = {}

        # --- Define panels using inch-based coordinates
        # Format: X0, Y0, DX, DY (in inches)
        self.AXs['Notes']                  = self.create_panel([0.5, 0.5, 7, 1], 'Notes', hide_axis=True)
        self.AXs['Traces']                 = self.create_panel([0.5, 1.5, 7, 2], 'Traces', hide_axis=True)
        self.AXs['Variation dFoF']         = self.create_panel([0.5, 3.5, 2, 2], 'Variation dFoF')
        self.AXs['Responsiveness roi']     = self.create_panel([3.5, 3.5, 2, 2], 'Responsiveness roi')
        self.AXs['Responsiveness session'] = self.create_panel([6.5, 3.5, 2, 2], 'Responsiveness session')

        plt.tight_layout()

    def create_panel(self, coords_in_inches, title=None, hide_axis=False):
        """
        coords_in_inches: [x0, y0, dx, dy] in inches
                          left-to-right, top-to-bottom (A4 layout convention)
        """
        fig_w, fig_h = self.fig.get_size_inches()
        x0, y0, dx, dy = coords_in_inches

        # Convert to bottom-up and normalized coordinates
        y0 = fig_h - y0 - dy
        rect_norm = [x0 / fig_w, y0 / fig_h, dx / fig_w, dy / fig_h]

        ax = self.fig.add_axes(rect_norm)
        if title:
            ax.set_title(title, loc='left', pad=2, fontsize=8)
        if hide_axis:
            ax.axis('off')

        return ax
    
    def fill_PDF(self, 
                 image1, 
                 image2, 
                 image3, 
                 image4): 
        
        for key in self.AXs:
            self.AXs[key].axis('off')

            if key=="Notes": 
                self.AXs[key].axis('off')
                txt = "Protocol Orientations and Contrasts "
                self.AXs[key].text(0.1, 0.9, txt, va='top', ha='left', fontsize=10, wrap=True)


            elif key=='Traces':
                self.AXs[key].imshow(image1)
            
            elif key=='Variation dFoF':
                self.AXs[key].imshow(image2)
                
            elif key=='Responsiveness roi':
                self.AXs[key].imshow(image3)

            elif key=='Responsiveness session':
                self.AXs[key].imshow(image4)
          