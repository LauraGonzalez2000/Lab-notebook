"""
Cranial-window image (from the intrinsic imaging sessions) with the position
of the 2P field of view (FOV) drawn on top.

Where the information comes from
--------------------------------
- FOV position: stage X/Y (um) at acquisition, relative to the "zero" landmark,
    stored in the NWB by physion:  imaging_plane.location = 'coords from 0-ref: (x=..., y=...)'
- FOV size: image shape x imaging_plane.grid_spacing (um/pixel)
- window image: <dataset>/Processed/intrinsic_img/<intrinsic_session>/<window_image>
- zero landmark position in the window image (pixels): clicked once with `pick_zero`
- visual areas: <intrinsic_session>/segmentation.npy, saved by Visual-Areas-Segmentation.py

DataTable.xlsx (or the first DataTable*.xlsx, e.g. DataTable_.xlsx)
--------------  -> full instructions in General_summary/README-window-FOV.md
"Subjects" sheet, one row per mouse (columns to add):
    intrinsic_session   e.g. "2026_11_03/12-11-28"
    window-image (or window_image)  e.g. "vasculature" or "vasculature-227.tif"
                        (".tif" added if missing; first "vasculature*.tif" if empty)
    zero_px_x, zero_px_y   position of the zero landmark in the window image (pixels)

"Recordings" sheet, optional override when the zero was re-set for a recording:
    zero_px_x, zero_px_y

Stage -> window image orientation (to calibrate once for the rig)
-------------------------------------------------------------
    STAGE_X_SIGN, STAGE_Y_SIGN : +1/-1, direction of the stage +X/+Y axes in the image
                                 (image x to the right, image y downward)
    STAGE_ANGLE_DEG : rotation of the stage axes relative to the image axes
"""
import os, re, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



# --- rig calibration (TO CALIBRATE: move the stage along +X and +Y and check in the image) ---
STAGE_X_SIGN = 1
STAGE_Y_SIGN = 1
STAGE_ANGLE_DEG = 0.

# height of the window (intrinsic camera) image, if not in the intrinsic metadata.json
DEFAULT_CAMERA_HEIGHT_MM = 2.7


#####################################################
###       FOV position and size from the NWB      ###
#####################################################

def get_FOV_position(data):
    """
    stage position of the FOV relative to the zero landmark, in um
        -> (x, y), or None if not stored in the file (older files: location='V1')
    """
    for plane in data.nwbfile.imaging_planes.values():
        m = re.search(r'x=\s*([-\d.eE+]+)\s*,\s*y=\s*([-\d.eE+]+)', plane.location)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def get_FOV_depth(data):
    """ depth of the imaging plane (um), None if not found """
    for plane in data.nwbfile.imaging_planes.values():
        m = re.search(r'Depth=\s*([-\d.]+)', plane.description)
        if m:
            return float(m.group(1))
    return None


def get_FOV_size(data, key='meanImg'):
    """ size of the FOV in um: (width, height) """
    img = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:]
    plane = list(data.nwbfile.imaging_planes.values())[0]
    # grid_spacing = (um/px along Y, um/px along X), labelled "meters" but in um
    dy, dx = np.array(plane.grid_spacing[:2], dtype=float)
    return img.shape[1]*dx, img.shape[0]*dy


#####################################################
###       window image from the DataTable          ###
#####################################################

def find_processed_folder(dataset_folder):
    """ "Processed" or "processed" subfolder """
    for name in ['Processed', 'processed']:
        if os.path.isdir(os.path.join(dataset_folder, name)):
            return os.path.join(dataset_folder, name)
    raise FileNotFoundError(' [!!] no "Processed" folder in %s [!!]' % dataset_folder)


def _row(df, column, value):
    """ row of `df` where `column` matches `value` (as strings), None if no match """
    if (df is None) or (column not in df):
        return None
    rows = df[df[column].astype(str).str.strip()==str(value).strip()]
    return rows.iloc[0] if len(rows)>0 else None


def _value(row, key):
    if (row is None) or (key not in row) or pd.isna(row[key]):
        return None
    return row[key]


def find_intrinsic_sessions(processed_folder, subject):
    """ intrinsic sessions ("day/time") whose metadata.json has this subject """
    sessions = []
    for md in sorted(glob.glob(os.path.join(processed_folder, 'intrinsic_img', '*', '*', 'metadata.json'))):
        with open(md) as f:
            if str(json.load(f).get('subject', '')).strip()==str(subject).strip():
                sessions.append('/'.join(pathlib_parts(os.path.dirname(md))[-2:]))
    return sessions


def pathlib_parts(path):
    return os.path.normpath(path).split(os.sep)


def find_datatable(dataset_folder, datatable='DataTable.xlsx'):
    """ `datatable` if it exists, otherwise the first DataTable*.xlsx (e.g. DataTable_.xlsx) """
    if os.path.isfile(os.path.join(dataset_folder, datatable)):
        return os.path.join(dataset_folder, datatable)
    files = sorted(f for f in glob.glob(os.path.join(dataset_folder, 'DataTable*.xlsx'))
                        if not os.path.basename(f).startswith('~$'))   # skip Excel lock files
    return files[0] if len(files)>0 else None


def get_window_info(data, dataset_folder,
                    datatable='DataTable.xlsx'):
    """
    window image and zero landmark of the mouse of this recording

    returns a dict with:
        'image_file', 'zero_px' (x, y) or None, 'um_per_px', 'intrinsic_session'
    or None if no window image was found
    """
    subject = data.nwbfile.subject.subject_id
    recording = os.path.basename(getattr(data, 'filename', '')).replace('.nwb', '')
    processed = find_processed_folder(dataset_folder)

    sheets = {}
    datatable_file = find_datatable(dataset_folder, datatable)
    if datatable_file is not None:
        sheets = pd.read_excel(datatable_file, sheet_name=None)
    else:
        print(' [!!] no DataTable*.xlsx in %s [!!]' % dataset_folder)
    srow = _row(sheets.get('Subjects'), 'subject', subject)

    # --- intrinsic session ---
    session = _value(srow, 'intrinsic_session')
    if session is None:
        sessions = find_intrinsic_sessions(processed, subject)
        if len(sessions)==0:
            print(' [!!] no window image found for subject "%s" [!!]' % subject)
            return None
        session = sessions[0]
        print(' [!!] subject "%s": no "intrinsic_session" in DataTable, using %s (out of %s) [!!]' %\
                (subject, session, sessions))
    session_folder = os.path.join(processed, 'intrinsic_img', *str(session).replace('\\', '/').split('/'))

    # --- window image ---
    image = _value(srow, 'window_image')
    if image is None:
        image = _value(srow, 'window-image')
    if (image is not None) and (os.path.splitext(str(image))[1]==''):
        image = str(image)+'.tif'      # "vasculature" -> "vasculature.tif"
    if image is None:
        images = sorted(glob.glob(os.path.join(session_folder, 'vasculature*.tif')))
        if len(images)==0:
            print(' [!!] no vasculature*.tif in %s [!!]' % session_folder)
            return None
        image = os.path.basename(images[0])
    image_file = os.path.join(session_folder, image)

    # --- zero landmark: recording override > subject ---
    zero_px = None
    rrow = None
    if 'Recordings' in sheets and recording:
        rec = sheets['Recordings']
        if ('day' in rec) and ('time' in rec):
            key = rec['day'].astype(str)+'-'+rec['time'].astype(str)
            rows = rec[key==recording]
            rrow = rows.iloc[0] if len(rows)>0 else None
    for row in [rrow, srow]:
        if (_value(row, 'zero_px_x') is not None) and (_value(row, 'zero_px_y') is not None):
            zero_px = (float(row['zero_px_x']), float(row['zero_px_y']))
            break

    # --- scale of the window image ---
    height_mm = DEFAULT_CAMERA_HEIGHT_MM
    md_file = os.path.join(session_folder, 'metadata.json')
    if os.path.isfile(md_file):
        with open(md_file) as f:
            height_mm = float(json.load(f).get('Height-of-Microscope-Camera-Image-in-mm', height_mm))
    img = load_window_image(image_file)
    um_per_px = 1e3*height_mm/img.shape[0]

    return {'image_file':image_file, 'image':img,
            'zero_px':zero_px, 'um_per_px':um_per_px,
            'intrinsic_session':session, 'subject':subject}


def load_window_image(image_file, percentiles=(1, 99.5)):
    """ window image (16-bit tif), normalized to [0,1] """
    img = np.array(plt.imread(image_file), dtype=float)
    if img.ndim==3:
        img = img.mean(axis=2)
    vmin, vmax = np.percentile(img, percentiles)
    return np.clip((img-vmin)/(vmax-vmin), 0, 1)


#####################################################
###       plot                                     ###
#####################################################

DISPLAY_MAX_PX = 600   # size of the window image in the figures (keeps the .svg small)

def imshow_window(ax, img, max_px=DISPLAY_MAX_PX):
    """
    window image, subsampled for display, in the pixel coordinates of the full image
        (so that zero_px, FOV and areas stay at the right place)
    """
    H, W = img.shape
    f = max(1, int(np.ceil(max(H, W)/max_px)))
    ax.imshow(img[::f, ::f], cmap='gray', vmin=0, vmax=1, interpolation='none',
              extent=[-0.5, W-0.5, H-0.5, -0.5])
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)


def stage_to_image(xy_um, zero_px, um_per_px,
                   x_sign=None, y_sign=None, angle_deg=None):
    """ stage coordinates (um, relative to zero) -> window image pixels """
    x_sign = STAGE_X_SIGN if x_sign is None else x_sign
    y_sign = STAGE_Y_SIGN if y_sign is None else y_sign
    angle = np.deg2rad(STAGE_ANGLE_DEG if angle_deg is None else angle_deg)
    xy = np.atleast_2d(xy_um).astype(float)*[x_sign, y_sign]
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return np.array(zero_px)+(xy@R.T)/um_per_px


def show_window_with_FOV(data, dataset_folder,
                         ax=None, color='green', lw=1.5,
                         datatable='DataTable.xlsx',
                         with_zero=True, with_FOV=True, with_title=True):
    """
    window image of the mouse with the FOV outline (and the zero landmark)
        with_FOV=False -> only the window image
        -> returns the info dict (None if no window image)
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    info = get_window_info(data, dataset_folder, datatable=datatable)
    if info is None:
        ax.annotate('no window image', (0.5, 0.5), ha='center', xycoords='axes fraction')
        ax.axis('off')
        return None

    imshow_window(ax, info['image'])
    ax.axis('off')

    title = ''
    if with_FOV:
        title += draw_FOV(data, info, ax, color=color, lw=lw, with_zero=with_zero)

    if with_title:
        ax.set_title(title, fontsize=7)
    return info


def draw_FOV(data, info, ax, color='green', lw=1.5, with_zero=True):
    """
    FOV outline (and zero landmark) on an axis showing the window image
        -> returns a text for the title
    """
    fov_xy = get_FOV_position(data)
    if info['zero_px'] is None:
        return '\nno zero_px in DataTable'
    if fov_xy is None:
        return '\nno FOV coordinates in NWB'
    if with_zero:
        ax.plot(*info['zero_px'], '+', color='yellow', ms=8, mew=1.5)
    w, h = get_FOV_size(data)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])+fov_xy
    ax.add_patch(Polygon(stage_to_image(corners, info['zero_px'], info['um_per_px']),
                         closed=True, fill=False, edgecolor=color, lw=lw))
    depth = get_FOV_depth(data)
    z = ', z=%.0f' % depth if depth is not None else ''
    return '\nFOV\n (x=%.0f, y=%.0f%s)um' % (fov_xy[0], fov_xy[1], z)


#####################################################
###       visual areas (Visual-Areas-Segmentation)  ###
#####################################################

# colors by visual field sign, as in Visual-Areas-Segmentation.py
#   (RetinotopicMapping.plotPatches: "jet" colormap, sign +1 -> red, -1 -> blue)
AREA_COLORS = {1: 'tab:red', -1: 'tab:blue', 0: 'grey'}

def load_segmentation(session_folder, filename='segmentation.npy'):
    """ areas saved by Visual-Areas-Segmentation.py, None if not there """
    f = os.path.join(session_folder, filename)
    if not os.path.isfile(f):
        return None
    return np.load(f, allow_pickle=True).item()


def show_window_with_areas(data, dataset_folder,
                           ax=None, color='green', lw=1.5,
                           datatable='DataTable.xlsx',
                           alpha=0.3, with_names=False,
                           with_zero=True, with_FOV=True, with_title=True):
    """
    window image with the visual areas (segmentation.npy in the intrinsic session folder)
        and the FOV outline on top
        -> returns the segmentation dict (None if not found)
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    info = get_window_info(data, dataset_folder, datatable=datatable)
    if info is None:
        ax.annotate('no window image', (0.5, 0.5), ha='center', xycoords='axes fraction')
        ax.axis('off')
        return None

    imshow_window(ax, info['image'])
    ax.axis('off')
    title = 'map\n (%s)' % info['intrinsic_session']

    seg = load_segmentation(os.path.dirname(info['image_file']))
    if seg is None:
        title += '\nno segmentation.npy (run Visual-Areas-Segmentation.py)'
    else:
        H, W = info['image'].shape
        # the maps cover the full camera image, at a lower resolution
        #   -> stretch the masks over the window image
        extent = [-0.5, W-0.5, H-0.5, -0.5]
        Hm, Wm = seg['map_shape']
        if abs(Hm/Wm - H/W) > 0.02*H/W:
            print(' [!!] segmentation map (%ix%i) and window image (%ix%i) have different aspect ratios [!!]' %\
                    (Hm, Wm, H, W))
        for key, patch in seg['patches'].items():
            mask = np.array(patch['mask'], dtype=float)
            c = AREA_COLORS.get(int(patch['sign']), 'grey')
            # vector fill (not an image layer: stays visible in the .svg)
            ax.contourf(mask, levels=[0.5, 1.5], colors=[c], alpha=alpha,
                        extent=extent, origin='upper')
            ax.contour(mask, levels=[0.5], colors=[c], linewidths=0.8,
                       extent=extent, origin='upper')
            if with_names:
                iy, ix = np.argwhere(mask>0).mean(axis=0)
                ax.annotate(patch.get('name', key),
                            ((ix+0.5)*W/Wm-0.5, (iy+0.5)*H/Hm-0.5),
                            color='w', fontsize=6, ha='center', va='center')
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])

    if with_FOV:
        draw_FOV(data, info, ax, color=color, lw=lw, with_zero=with_zero)

    if with_title : 
        ax.set_title(title, fontsize=7)

    return seg


#####################################################
###       click the zero landmark (once per mouse) ###
#####################################################

def find_zero_image(raw_session_folder):
    """
    2P image of the zero landmark in a raw session folder (e.g. Processed/2026_11_17/16-29-08):
        first .tif in a folder whose name contains "zero" (any case, any depth),
        e.g. TSeries-*/SingleImage-...-zero227/References/...-16bit-Reference.tif
    """
    zero_folders = sorted(os.path.join(root, d)\
                            for root, dirs, _ in os.walk(raw_session_folder)\
                                for d in dirs if 'zero' in d.lower())
    for pattern in ['*16bit*Reference.tif', '*Reference.tif', '*.tif']:
        for folder in zero_folders:
            files = sorted(glob.glob(os.path.join(folder, '**', pattern), recursive=True))
            if len(files)>0:
                return files[0]
    return None


def pick_zero(window_image_file, zero_image_file=None, timeout=120):
    """
    click the zero landmark on the window image
        (the 2P image of the zero, e.g. TSeries-.../SingleImage-...-zero<ID>, shown as a reference)

    run with an interactive backend (e.g. "%matplotlib qt"), then
        copy the printed values in the "zero_px_x" / "zero_px_y" columns of the DataTable

    timeout: seconds to wait for the click (0 = forever)
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if ('inline' in backend) or (backend in ['agg', 'pdf', 'svg', 'ps', 'cairo', 'template']):
        raise RuntimeError(' [!!] non-interactive matplotlib backend ("%s"): no window to click in.\n'
                           '      run "%%matplotlib qt" first (restart the kernel if it has no effect) [!!]' % backend)

    n = 2 if zero_image_file else 1
    fig, AX = plt.subplots(1, n, figsize=(6*n, 6))
    AX = np.atleast_1d(AX)
    AX[0].imshow(load_window_image(window_image_file), cmap='gray')
    AX[0].set_title('click the zero landmark (then close the window)')
    if zero_image_file:
        if os.path.isdir(zero_image_file):
            # a raw session folder or a SingleImage-...-zero folder
            zero_image_file = find_zero_image(zero_image_file)
        if zero_image_file and os.path.isfile(zero_image_file):
            AX[1].imshow(load_window_image(zero_image_file), cmap='gray')
            AX[1].set_title('2P image of the zero')
        else:
            AX[1].set_title('2P zero image not found', fontsize=7)
    for ax in AX:
        ax.axis('off')

    # show the window and bring it in front of VS Code
    fig.show()
    try:
        win = fig.canvas.manager.window
        win.raise_(); win.activateWindow()      # Qt
    except Exception:
        pass
    print(' -> click the zero landmark on the left image (waiting %s) ...' %\
            ('%is' % timeout if timeout else 'without time limit'))

    pts = fig.ginput(1, timeout=timeout)
    if len(pts)==0:
        print(' [!!] no point clicked [!!]')
        return None
    x, y = pts[0]
    AX[0].plot(x, y, '+', color='yellow', ms=12, mew=2)
    fig.canvas.draw()
    print('zero_px_x = %.0f    zero_px_y = %.0f' % (x, y))
    return x, y
