# %% [markdown]
# # Opsin / GCaMP colocalization
#
# Remove the bleed-through of the green channel (GCaMP) into the red channel (opsin)
# on the 2P reference images of a FOV.
#
# Method adapted from `preprocess_RCL` in physion/imaging/red_label.py:
# - in each bin of green intensity, take a low percentile of the red intensity
#   (= red signal explained by the green bleed-through only)
# - fit a line red = a*green + b through these points
# - corrected red images:
#     red - (a*green+b)      (subtraction, clipped at 0)
#     red / (a*green+b)      (ratio)

# %%
import os, glob
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from physion.utils.plot_tools import figure, get_linear_colormap

# %% [markdown]
# ## Load the images

# %%
# folder with the reference images of the FOV (Bruker "SingleImage-.../References")
folder = r'E:\DATA\In_Vivo_experiments\opto\vision-survey-opto\SST-Cre\Processed\2026_11_05\18-17-02\h5-11052026--008\SingleImage-11052026-1055-1148_FOV_960\References'

def load_channel(folder, channel):
    """ raw 16-bit reference image of a channel (Bruker: Ch1 = red, Ch2 = green) """
    files = glob.glob(os.path.join(folder, '*-%s-16bit-Reference.tif' % channel))
    if len(files)==0:
        raise FileNotFoundError('no "*-%s-16bit-Reference.tif" in %s' % (channel, folder))
    return np.array(Image.open(files[0]), dtype=float)

green = load_channel(folder, 'Ch2')   # GCaMP
red = load_channel(folder, 'Ch1')     # opsin
print('image size:', green.shape)

# %% [markdown]
# ## Fit the bleed-through (red vs green)

# %%
def fit_bleedthrough(green, red, nbins=15,
                     percentile=5, green_range=(1, 99.5), min_pixels=50):
    """
    linear fit of the lower envelope of red vs green
        -> polynomial coefficients p (red = p[0]*green + p[1]),
           and the points used for the fit (X, Y)

    differences with preprocess_RCL (red_label.py), which uses the min of each bin
    over the full green range: here a few very bright pixels would set the slope, so
        - bins only over the bulk of the pixels (green_range: percentiles of green)
        - low `percentile` of red in each bin instead of the min (percentile=0 -> min)
        - bins with less than `min_pixels` pixels are skipped
    """
    x, y = green.flatten(), red.flatten()
    X, Y = [], []
    bins = np.linspace(*np.percentile(x, green_range), nbins+1)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        cond = (x>=b0) & (x<b1)
        if np.sum(cond)>=min_pixels:
            X.append(np.mean(x[cond]))
            Y.append(np.percentile(y[cond], percentile))
    return np.polyfit(X, Y, 1), np.array(X), np.array(Y)

p, X, Y = fit_bleedthrough(green, red)
print('red = %.3f * green + %.1f' % (p[0], p[1]))

fig, ax = figure(ax_scale=(1.5,2))
# subsample the pixels for the plot (512x512 = 262k points)
i = np.random.choice(green.size, min(20000, green.size), replace=False)
ax.scatter(green.flatten()[i], red.flatten()[i], s=1, alpha=0.3)
ax.scatter(X, Y, s=8, color='r')
xx = np.linspace(green.min(), green.max(), 2)
ax.plot(xx, np.polyval(p, xx), color='r')
ax.set_xlabel('green (Ch2, GCaMP)'); ax.set_ylabel('red (Ch1, opsin)')

# %% [markdown]
# ## Corrected red images

# %%
contamination = np.polyval(p, green)
red_minus = np.clip(red-contamination, 0, np.inf)           # red - (a*green+b)
red_ratio = red/np.clip(contamination, 1, np.inf)           # red / (a*green+b)

def norm(img, exponent=0.5, percentiles=(1, 99.5)):
    """ normalize to [0,1] between percentiles (a few bright pixels would make the image dark),
        with a non-linearity (as the "Display Exponent" of red_label) """
    vmin, vmax = np.percentile(img, percentiles)
    return np.clip((img-vmin)/(vmax-vmin), 0, 1)**exponent

fig, AX = figure(axes=(4,1), ax_scale=(1.3,2.4), wspace=0.2)
for ax, img, title, cmap in zip(AX,
        [green, red, red_minus, red_ratio],
        ['green (GCaMP)', 'red (opsin)', 'red - X*green', 'red / (X*green)'],
        ['tab:green', 'tab:red', 'tab:red', 'tab:red']):
    ax.imshow(norm(img), cmap=get_linear_colormap('k', cmap), interpolation='none')
    ax.set_title(title, fontsize=8)
    ax.axis('off')

# %% [markdown]
# ## Overlay: green and corrected red

# %%
fig, ax = figure(ax_scale=(2,4))
rgb = np.zeros((*green.shape, 3))
rgb[...,0] = norm(red_minus)   # red: opsin (corrected)
rgb[...,1] = norm(green)       # green: GCaMP
ax.imshow(rgb, interpolation='none')
ax.set_title('GCaMP (green) + opsin corrected (red)', fontsize=8)
ax.axis('off')
plt.show()

# %% [markdown]
# # All recordings of a dataset, by opsin
#
# - opsin: "notes" of the recording metadata.json ("stg ..." / "ArchT ...");
#   if no note, taken from the other recordings of the same mouse
# - single images: Processed/<day>/<time>/*/SingleImage-*/ (laser wavelength from the .xml),
#   skipping the ones with the red PMT off (gain 0, e.g. the "zero" images)

# %%
import re, json

dataset_folder = r'E:\DATA\In_Vivo_experiments\opto\vision-survey-opto\SST-Cre'

OPSINS = {'stg': 'stGtACR2', 'archt': 'ArchT'}   # keyword in the notes (lower case) -> name

def get_opsin(notes):
    """ opsin from the notes of the metadata, None if not found """
    for key, name in OPSINS.items():
        if key in str(notes).lower():
            return name
    return None

def single_image_info(single_image_folder):
    """ laser wavelength (nm), PMT gains, averaging, from the Bruker .xml """
    xml = glob.glob(os.path.join(single_image_folder, '*.xml'))
    info = {'folder': single_image_folder, 'wavelength': None,
            'gain_red': None, 'gain_green': None, 'averaging': None}
    if len(xml)==0:
        return info
    with open(xml[0], encoding='utf-8', errors='ignore') as f:
        x = f.read()
    m = re.search(r'key="laserWavelength".*?value="(\d+)"', x, re.S)
    if m: info['wavelength'] = int(m.group(1))
    m = re.search(r'key="pmtGain".*?</PVStateValue>', x, re.S)
    if m:
        for value, ch in re.findall(r'value="([\d.]+)" description="([^"]+)"', m.group(0)):
            if 'Red' in ch: info['gain_red'] = float(value)
            if 'Green' in ch: info['gain_green'] = float(value)
    m = re.search(r'key="rastersPerFrame" value="(\d+)"', x)
    if m: info['averaging'] = int(m.group(1))
    return info

def find_recordings(dataset_folder):
    """
    recordings of the dataset with their opsin and single images
        -> list of dict: recording, subject, opsin, notes, images (list of single_image_info)
    """
    recordings = []
    for md in sorted(glob.glob(os.path.join(dataset_folder, 'Processed', '20*', '*', 'metadata.json'))):
        with open(md) as f:
            metadata = json.load(f)
        session = os.path.dirname(md)
        images = [single_image_info(d) for d in
                    sorted(glob.glob(os.path.join(session, '*', 'SingleImage*')))]
        recordings.append({
            'recording': '%s-%s' % tuple(session.split(os.sep)[-2:]),
            'subject': str(metadata.get('subject_ID', '')),
            'notes': metadata.get('notes', ''),
            'opsin': get_opsin(metadata.get('notes', '')),
            # red channel on (PMT gain>0) and reference images there
            'images': [im for im in images if (im['gain_red'] or 0)>0 and\
                        len(glob.glob(os.path.join(im['folder'], 'References', '*-Ch1-16bit-Reference.tif')))>0],
        })
    # no opsin in the notes -> opsin of the other recordings of the same mouse
    for rec in recordings:
        if rec['opsin'] is None:
            others = set(r['opsin'] for r in recordings
                            if r['subject']==rec['subject'] and r['opsin'] is not None)
            if len(others)==1:
                rec['opsin'] = others.pop()
                print(' [!!] %s: no opsin in the notes, %s from the other recordings of mouse %s' %\
                        (rec['recording'], rec['opsin'], rec['subject']))
            else:
                print(' [!!] %s: opsin unknown (notes: "%s") [!!]' % (rec['recording'], rec['notes']))
    return recordings

recordings = find_recordings(dataset_folder)
for rec in recordings:
    print(rec['recording'], 'mouse', rec['subject'], rec['opsin'],
          ['%s (%snm)' % (os.path.basename(im['folder']), im['wavelength']) for im in rec['images']])

# %%
def colocalization_row(AX, green, red, title=''):
    """ one row: green | red | red corrected | overlay | fit """
    p, X, Y = fit_bleedthrough(green, red)
    red_minus = np.clip(red-np.polyval(p, green), 0, np.inf)

    for ax, img, cmap in zip(AX[:3], [green, red, red_minus],
                             ['tab:green', 'tab:red', 'tab:red']):
        ax.imshow(norm(img), cmap=get_linear_colormap('k', cmap), interpolation='none')
    rgb = np.zeros((*green.shape, 3))
    rgb[...,0], rgb[...,1] = norm(red_minus), norm(green)
    AX[3].imshow(rgb, interpolation='none')
    for ax in AX[:4]:
        ax.axis('off')
    AX[0].annotate(title, (-0.05, 0.5), xycoords='axes fraction',
                   rotation=90, ha='right', va='center', fontsize=7)

    i = np.random.choice(green.size, min(5000, green.size), replace=False)
    AX[4].scatter(green.flatten()[i], red.flatten()[i], s=0.5, alpha=0.3)
    AX[4].scatter(X, Y, s=4, color='r')
    xx = np.linspace(*np.percentile(green, (0, 99.9)), 2)
    AX[4].plot(xx, np.polyval(p, xx), color='r', lw=1)
    AX[4].set_title('red=%.2f*green%+.0f' % tuple(p), fontsize=6)
    # axes on the bulk of the pixels (a few very bright pixels would squeeze the cloud)
    AX[4].set_xlim(np.percentile(green, 0), np.percentile(green, 99.9))
    AX[4].set_ylim(np.percentile(red, 0), np.percentile(red, 99.9))
    AX[4].tick_params(labelsize=6)
    return p

# one image per recording and wavelength (the first one)
for opsin in sorted(set(r['opsin'] for r in recordings if r['opsin'])):
    rows = []
    for rec in [r for r in recordings if r['opsin']==opsin]:
        for wl in sorted(set(im['wavelength'] for im in rec['images'])):
            im = [im for im in rec['images'] if im['wavelength']==wl][0]
            rows.append((rec, im))
    if len(rows)==0:
        continue

    fig, AX = figure(axes=(5, len(rows)), ax_scale=(1.,2.),
                     wspace=0.3, hspace=0.3, reshape_axes=False)
    fig.suptitle(opsin, fontsize=10)
    for AXrow, (rec, im) in zip(AX, rows):
        refs = os.path.join(im['folder'], 'References')
        colocalization_row(AXrow,
                           load_channel(refs, 'Ch2'), load_channel(refs, 'Ch1'),
                           title='%s\nmouse %s, %snm' % (rec['recording'], rec['subject'], im['wavelength']))
    for ax, title in zip(AX[0], ['green (GCaMP)', 'red (%s)' % opsin, 'red - X*green',
                                 'overlay', 'red vs green']):
        ax.set_title(title, fontsize=7) if title!='red vs green' else None
    AX[-1][4].set_xlabel('green', fontsize=7); AX[-1][4].set_ylabel('red', fontsize=7)
plt.show()

# %%
