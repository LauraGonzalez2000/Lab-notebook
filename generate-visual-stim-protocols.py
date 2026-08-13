import os, subprocess, pathlib
import itertools

Screen="Dell-2020"

# stimulus-center position:
Saccade_Amplitude="200.0"

x=45.
y=20.

def build_movie(json_protocol, name='temp', rm=True):

    with open('%s.json' % name, 'w') as f:
        f.write(json_protocol)
        
    cmd = 'python -m physion.visual_stim.build ../../%s.json' % name

    p = subprocess.Popen(cmd,
            cwd = os.path.join(pathlib.Path(__file__).resolve(), 'physion', 'src'),
            shell=True)

    # clean up generated json afterwards
    if rm:
        os.remove('%s.json' % name)


###########################################
###      center patch                   ###
###########################################

def center_patch(Screen='', Nrepeat=1, x=0, y=0):
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
        "N-repeat": %i
}  """ % (Screen, x, y, Nrepeat)


###########################################
###      natural images                 ###
###########################################

def natural_images(Screen='', Nrepeat=1, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "natural_image",
        "Screen": "%s",
        "units": "lin-deg",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 2.0,
        "-----------------------------------------------------------------------2":0,
        "Image-ID-1": 1.0,  "Image-ID-2": 5.0, "N-Image-ID": 5,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i 
}  """ % (Screen, Nrepeat)



def MultiProtocol(Screen='', Nrepeat=1, 
                  interstim=6.0,
                  jitter=2.0,
                  x=0, y=0):
   
    multiprotocol = """{
        "Presentation": "multiprotocol",
        "shuffling" :"full-with-alternate-even-odd-repeats",
        "shuffling-seed" :34,
        "movie_refresh_freq":30.0,
        "units":"cm",
        "Screen": "%s",
        "presentation-prestim-period": 10.0,
        "presentation-poststim-period": 10.0,
        "presentation-interstim-period": %.1f,
        "presentation-interstim-jitter": %.1f,
        "presentation-blank-screen-color": 0.5,
    """ % (Screen, interstim, jitter)

    i = 1 # protocol counter
    for protocol_func, protocol_name in zip(
                    [center_patch, natural_images],
                    ['center-patch', 'natural-images']):

        protocol = protocol_func(Screen=Screen, Nrepeat=Nrepeat, x=x, y=y)

        with open('protocol-%s.json' % protocol_name, 'w') as f:
            f.write(protocol)

        multiprotocol += '  "Protocol-%i": "protocol-%s.json",\n' % (i, protocol_name)

        i+=1

    multiprotocol = multiprotocol[:-2]+'}'

    return multiprotocol

if 0:
  x, y =0, 0
  build_movie(center_patch(Screen=Screen, Nrepeat=1, x=x, y=y),
              name='center-patch')
if 1:
  build_movie(MultiProtocol(Screen=Screen, Nrepeat=4),
              name='vision-survey-short+1sPrePostOpto',
              rm=True)

if 0:
    for x, y in itertools.product(
                [-25, 0, 25], [-15, 0, 15]):
        build_movie(MultiProtocol(Screen=Screen, Nrepeat=2, x=x, y=y),
                    name='vision-survey-short-x=%.0f-y=%.0f+1sPrePostOpto' % (x,y),
                    rm=True)