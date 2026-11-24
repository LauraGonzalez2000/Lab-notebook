import os, subprocess, pathlib
import itertools
import json

Screen="Dell-2020"


# stimulus-center position:
Saccade_Amplitude="200.0"

x=45.
y=20.

protocol_folder = os.path.join(os.path.expanduser('~'),
            'OneDrive - ICM', 'Lab-Notebook', 'visualStim-protocols')

protocol_folder_demo = os.path.join(os.path.expanduser('~'),
            'OneDrive - ICM', 'Lab-Notebook', 'visualStim-protocols', 'demo')

# '~' kept unexpanded: physion expands it when loading the images (works for any user name)
NI_BANK_ONEDRIVE = '~/OneDrive - ICM/Lab-Notebook/visualStim-protocols/NI_bank'

def build_movie(json_protocol, name='temp', rm=True, demo=False):

    with open('%s.json' % name, 'w') as f:
        f.write(json_protocol)

    if not demo:
        cmd = 'python -m physion.visual_stim.build ..\\..\\%s.json -o "%s"' % (name, protocol_folder)
    else:
        cmd = 'python -m physion.visual_stim.build ..\\..\\%s.json -o "%s"' % (name, protocol_folder_demo)
    p = subprocess.Popen(cmd,
            cwd = os.path.join('.', 'physion', 'src'),
            shell=True)

    p.wait() # need to wait for completion

    # clean up generated json afterwards
    if rm:
        os.remove('%s.json' % name)


###########################################
###      center patch                   ###
###########################################

def center_patch(Screen='', Nrepeat=10, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "grating",
        "movie_refresh_freq":30.0,
        "Screen": "%s",
        "units":"cm",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 2,
        "speed": 0,
        "phase": 0,
        "spatial-freq": 0.04,
        "radius": 20,
        "x-center": %.1f,
        "y-center": %.1f,
        "-----------------------------------------------------------------------2":0,
        "angle-1": 90, "angle-2": 0, "N-angle": 2,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i,
        "presentation-interstim-period": 4
}  """ % (Screen, x, y, Nrepeat)

###########################################
###      natural images                 ###
###########################################

def natural_images(Screen='', Nrepeat=10, x=0, y=0,
                   NI_FOLDER=NI_BANK_ONEDRIVE):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "images",
        "Screen": "%s",
        "units": "lin-deg",
        "NI_FOLDER": %s,
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 2.0,
        "degree-per-pixel-of-image": 0.1432,
        "-----------------------------------------------------------------------2":0,
        "Image-ID-1": "2.png",
        "Image-ID-2": "3.png",
        "Image-ID-3": "4.png",
        "Image-ID-4": "5.png",
        "Image-ID-5": "6.png",
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i,
        "presentation-interstim-period": 4
}  """ % (Screen, json.dumps(NI_FOLDER), Nrepeat)

###########################################
###      looming protocol               ###
###########################################

def looming_stim(Screen='', Nrepeat=10, x=0, y=0):
    return """{
    "Presentation": "Stimuli-Sequence",
    "Stimulus": "looming_stim",
    "Screen": "%s",
    "units": "lin-deg",
    "-----------------------------------------------------------------------1": 0,
    "presentation-duration": 5.0,
    "looming-duration": 1.0,
    "end-duration": 4.0,
    "looming-nonlinearity": 3,
    "radius-start": 0.5,
    "radius-end": 120,
    "x-center":  %.1f,
    "y-center":  %.1f,
    "-----------------------------------------------------------------------3": 0,
    "N-repeat": %i,
    "presentation-prestim-period": 6,
    "presentation-poststim-period": 6,
    "presentation-interstim-period": 4,
    "presentation-blank-screen-color": 0.5,
    "movie_refresh_freq": 30.0
}"""% (Screen,x,y,Nrepeat)

###########################################
###   common settings of the subprotocols ###
###########################################
#   values of the recorded vision-survey (e.g. 2026_01_30-13-50-59.nwb)
#   N.B. in a multiprotocol, "Screen", "units" and "movie_refresh_freq"
#        are overwritten by those of the multiprotocol

def subprotocol_base(Screen, Stimulus, duration):
    return {
        "Presentation": "Stimuli-Sequence",
        "Stimulus": Stimulus,
        "Screen": Screen,
        "units": "cm",
        "movie_refresh_freq": 30.0,
        "presentation-duration": duration,
        "presentation-prestim-period": 6,
        "presentation-poststim-period": 6,
        "presentation-interstim-period": 4,
        "presentation-blank-screen-color": 0.5}

###########################################
###      drifting grating (full-field)  ###
###########################################
def drifting_grating(Screen=Screen, Nrepeat=10, x=0, y=0,
                     contrasts=(0.2, 1.0, 3)):
    """ full-field drifting grating, 3 contrasts: 0.2, 0.6, 1.0 """
    return json.dumps({
        **subprotocol_base(Screen, "grating", 2),
        "x-center": x, "y-center": y,
        "radius": 200,
        "angle": 0,
        "spatial-freq": 0.04,
        "speed": 2,
        "contrast-1": contrasts[0], "contrast-2": contrasts[1],
        "N-contrast": contrasts[2],
        "N-repeat": Nrepeat}, indent=4)

###########################################
###      drifting_surround              ###
###########################################
def drifting_surround(Screen=Screen, Nrepeat=10, x=0, y=0):
    """ grey center patch (contrast 0, radius 20) in a full-field drifting grating """
    return json.dumps({
        **subprotocol_base(Screen, "grating", 2),
        "x-center": x, "y-center": y,
        "radius": 20,
        "contrast": 0,
        "angle-surround": 0,
        "contrast-surround": 1,
        "speed-surround": 2,
        "spatial-freq-surround": 0.04,
        "radius-surround": 200,
        "N-repeat": Nrepeat}, indent=4)

###########################################
###      moving dots                    ###
###########################################
def moving_dots(Screen=Screen, Nrepeat=10, x=0, y=0):
    """ 10 black dots moving upward (direction 90), 2 speeds: 60, 90 """
    return json.dumps({
        **subprotocol_base(Screen, "scattered-moving-dots", 3.0),
        "speed-1": 60.0, "speed-2": 90.0, "N-speed": 2,
        "size": 5,
        "spacing": 15,
        "direction": 90,
        "ndots": 10,
        "seed": 1,
        "dotcolor": -1,
        "N-repeat": Nrepeat}, indent=4)

###########################################
###       grey-10min                    ###
###########################################
def grey_10min(Screen=Screen, Nrepeat=1, x=0, y=0):
    """ 10 min of grey screen (always a single episode, Nrepeat is ignored) """
    return json.dumps({
        **subprotocol_base(Screen, "uniform-bg", 600),
        "screen-color": 0.5, # "bg-color" in former physion versions
        "N-repeat": 1}, indent=4)

###########################################
###       black-2min                    ###
###########################################
def black_2min(Screen=Screen, Nrepeat=1, x=0, y=0):
    """ 2 min of black screen (always a single episode, Nrepeat is ignored) """
    return json.dumps({
        **subprotocol_base(Screen, "uniform-bg", 120),
        "screen-color": 0, # "bg-color" in former physion versions
        "N-repeat": 1}, indent=4)

###########################################
###       quick_spatial_mapping         ###
###########################################
def quick_spatial_mapping(Screen=Screen, Nrepeat=4, x=0, y=0):
    """ small drifting-grating patches (radius 12) on a 3x3 grid of positions """
    return json.dumps({
        **subprotocol_base(Screen, "grating", 1),
        "x-center-1": -36.0, "x-center-2": 36.0, "N-x-center": 3,
        "y-center-1": -23.0, "y-center-2": 23.0, "N-y-center": 3,
        "radius": 12,
        "angle": 0,
        "spatial-freq": 0.06,
        "speed": 2,
        "N-repeat": Nrepeat}, indent=4)


###########################################
###       full-field static gratings    ###
###########################################
def ff_gratings(angles=(0, 157.5, 8), contrasts=(0.5, 1.0, 2), Nrepeat=15,
                Screen=Screen, shuffling=None, seed=1,jitter=0):
    """
    full-field static gratings: angles=(angle-1, angle-2, N-angle)
                                contrasts=(contrast-1, contrast-2, N-contrast)

    shuffling=None -> episodes in order (as in the recorded
                        "ff-gratings-8orientation-2contrasts-15repeats" movie)
    shuffling='full' -> randomized order (with `seed`)
    """
    protocol = {
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "grating",
        "Screen": Screen,
        "units": "lin-deg",
        "movie_refresh_freq": 30.0,
        "shuffling-seed": seed,
        "presentation-duration": 2,
        "presentation-prestim-period": 4,
        "presentation-poststim-period": 4,
        "presentation-interstim-period": 4,
        "presentation-interstim-jitter": jitter,
        "presentation-blank-screen-color": 0.5,
        "N-repeat": Nrepeat,
        "angle-1": angles[0], "angle-2": angles[1], "N-angle": angles[2],
        "contrast-1": contrasts[0], "contrast-2": contrasts[1], "N-contrast": contrasts[2],
        "speed": 0,
        "spatial-freq": 0.04,
        "radius": 200,
        "phase": 90.0}
    if shuffling is not None:
        protocol["shuffling"] = shuffling
    return json.dumps(protocol, indent=4)


###########################################
###   natural images x contrasts        ###
###########################################

def natural_images_contrasts(images=['%i.png' % i for i in range(2, 12)],
                             contrasts=(0.5, 1.0, 2), Nrepeat=10,
                             Screen=Screen,
                             NI_FOLDER=NI_BANK_ONEDRIVE,
                             units='cm',
                             degree_per_pixel=0.1432,
                             interstim=5, jitter=2,
                             shuffling='full', seed=1):
    """
    full-screen natural images at several contrasts
        images: filenames in NI_FOLDER -> "Image-ID-1", "Image-ID-2", ... (in that order)
        contrasts=(contrast-1, contrast-2, N-contrast)

    units / degree_per_pixel (Dell-2020, 15cm from the eye):
        'cm'      -> real flat-screen geometry: the image is defined in visual angle,
                        so it appears stretched toward the screen borders (perspective correction)
                        0.0906 deg/px: image spans the full screen width (2*arctan(24/15)/1280px)
        'lin-deg' -> constant angle per screen pixel: no distortion
                        0.1432 deg/px = 1 image pixel per screen pixel

    defaults = recorded "10NaturalImages-2contrasts-10repeats"
        (N.B. there Image-ID 1 is "2.png", ..., Image-ID 10 is "11.png")
    """
    protocol = {
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "images",
        "Screen": Screen,
        "units": units,
        "movie_refresh_freq": 30.0,
        "presentation-duration": 2,
        "presentation-prestim-period": 4,
        "presentation-poststim-period": 4,
        "presentation-interstim-period": interstim,
        "presentation-interstim-jitter": jitter,
        "presentation-blank-screen-color": 0.5,
        "N-repeat": Nrepeat,
        "NI_FOLDER": NI_FOLDER,
        "degree-per-pixel-of-image": degree_per_pixel,
        "radius": 200,
        "contrast-1": contrasts[0], "contrast-2": contrasts[1], "N-contrast": contrasts[2]}
    for i, image in enumerate(images):
        protocol["Image-ID-%i" % (i+1)] = image
    if shuffling is not None:
        protocol["shuffling"] = shuffling
        protocol["shuffling-seed"] = seed
    return json.dumps(protocol, indent=4)


###########################################
###       vision survey                 ###
###########################################

# (function, name, N-repeat of the recorded vision-survey), in the recorded order
#   -> same "protocol_id" as in the NWB files (protocol_id = position - 1)
VISION_SURVEY = [
    (center_patch,          'center-patch',          10),
    (looming_stim,          'looming-stim',          10),
    (natural_images,        'natural-images',        10),
    (drifting_grating,      'drifting-grating',      10),
    (drifting_surround,     'drifting-surround',     10),
    (moving_dots,           'moving-dots',           10),
    (grey_10min,            'grey-10min',             1),
    (black_2min,            'black-2min',             1),
    (quick_spatial_mapping, 'quick-spatial-mapping',  4),
]


def MultiProtocol(Screen='', Nrepeat=None,
                  subprotocols=VISION_SURVEY,
                  shuffling='full',
                  seed=23,
                  prestim=6.0,
                  poststim=6.0,
                  interstim=4.0,
                  jitter=0.0,
                  x=0, y=0):
    """
    multiprotocol of the vision survey, defaults = recorded vision-survey
        Nrepeat=None -> N-repeat of each subprotocol from `subprotocols`
        Nrepeat=n    -> n repeats for all (grey/black screens stay single episodes)

    writes the "protocol-<name>.json" subprotocol files in the current folder
        and returns the multiprotocol json (as a string)
    """
    multiprotocol = {
        "Presentation": "multiprotocol",
        "shuffling": shuffling,
        "shuffling-seed": seed,
        "movie_refresh_freq": 30.0,
        "units": "cm",
        "Screen": Screen,
        "presentation-prestim-period": prestim,
        "presentation-poststim-period": poststim,
        "presentation-interstim-period": interstim,
        "presentation-blank-screen-color": 0.5}
    if jitter>0:
        multiprotocol["presentation-interstim-jitter"] = jitter

    for i, (protocol_func, protocol_name, nrepeats) in enumerate(subprotocols):

        n = nrepeats if Nrepeat is None else Nrepeat

        protocol = protocol_func(Screen=Screen, Nrepeat=n, x=x, y=y)

        # subprotocol settings matching the multiprotocol
        protocol = json.loads(protocol)
        protocol["presentation-interstim-period"] = interstim
        with open('protocol-%s.json' % protocol_name, 'w') as f:
            json.dump(protocol, f, indent=4)

        multiprotocol["Protocol-%i" % (i+1)] = "protocol-%s.json" % protocol_name

    return json.dumps(multiprotocol, indent=4)



if 0:
  x, y =0, 0
  build_movie(center_patch(Screen=Screen, Nrepeat=10, x=x, y=y),
              name='center-patch')

if 0:
  x, y =0, 0
  build_movie(looming_stim(Screen=Screen, Nrepeat=10, x=x, y=y),
              name='looming-stim')

if 0:
  x, y =0, 0
  build_movie(natural_images(Screen=Screen, Nrepeat=10, x=x, y=y),
              name='Natural-Images-4-repeats')

if 0:
  x, y =0, 0
  build_movie(drifting_grating(Screen=Screen, Nrepeat=10, x=x, y=y),
              name='drifting-grating')

if 0:
  x, y =0, 0
  build_movie(drifting_surround(Screen=Screen, Nrepeat=10, x=x, y=y),
              name='drifting-surround')

if 0:
  build_movie(moving_dots(Screen=Screen, Nrepeat=10),
              name='moving-dots')

if 0:
  build_movie(grey_10min(Screen=Screen),
              name='grey-10min')

if 0:
  build_movie(black_2min(Screen=Screen),
              name='black-2min')

if 0:
  build_movie(quick_spatial_mapping(Screen=Screen, Nrepeat=4),
              name='quick-spatial-mapping')

if 0:
  # 8 orientations (0 -> 157.5, every 22.5 deg) x 2 contrasts (0.5, 1)
  build_movie(ff_gratings(angles=(0, 157.5, 8), contrasts=(0.5, 1.0, 2),
                          Nrepeat=2, Screen=Screen, shuffling='full', jitter=2),
              name='ff-gratings-8orientation-2contrasts-15repeats',demo=True)

if 0:
  # 2 orientations (0, 90) x 8 contrasts (0 -> 1, every 1/7)
  build_movie(ff_gratings(angles=(0, 90, 2), contrasts=(0, 1.0, 8),
                          Nrepeat=2, Screen=Screen, shuffling='full', jitter=2),
              name='ff-gratings-2orientation-8contrasts-15repeats',demo=True)

if 0:
  # 10 natural images (OneDrive NI_bank) x 2 contrasts (0.5, 1)
  build_movie(natural_images_contrasts(contrasts=(0.5, 1.0, 2),
                                       Nrepeat=1, Screen=Screen,
                                       jitter=2),
              name='10NaturalImages-2contrasts-10repeats',demo=True)

if 0:
  # 2 natural images x 8 contrasts (0.05 -> 1), shuffled
  build_movie(natural_images_contrasts(images=['2.png', '3.png'],
                                       contrasts=(0.05, 1.0, 8),
                                       Nrepeat=1, Screen=Screen,
                                       interstim=4, jitter=2,
                                       shuffling='full'),
              name='2NaturalImages-8contrasts-15repeats',demo=True)

if 1:
  x, y =0, 0
  build_movie(MultiProtocol(Screen=Screen, Nrepeat=1, x=x, y=y, jitter=2),
              name='vision-survey', demo=True)
