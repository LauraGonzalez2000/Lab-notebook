import os, subprocess, pathlib
import itertools

Screen="Dell-2020"

x=0.
y=0.

def build_movie(json_protocol, name='temp', rm=True):

    with open('%s.json' % name, 'w') as f:
        f.write(json_protocol)
        
    if 'posix' in os.name:
        cmd = 'python -m physion.visual_stim.build ../../%s.json' % name
    else:
        cmd = 'python -m physion.visual_stim.build ..\\..\\%s.json' % name
    
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

def center_patch(Screen='', Nrepeat=1, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "grating",
        "movie_refresh_freq":30.0,
        "Screen": "%s",
        "units":"cm",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 1,
        "speed": 0,
        "phase": 0,
        "spatial-freq": 0.04,
        "x-center": %.1f,
        "y-center": %.1f,
        "angle": 0.0,
        "-----------------------------------------------------------------------2":0,
        "radius-1": 20, "radius-2": 50, "N-radius": 2,
        "contrast-1": 0.2, "contrast-2": 1.0, "N-contrast": 3,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i
}  """ % (Screen, x, y, Nrepeat)


###########################################
###      full-field drifting grating     ##
###########################################

def ff_drifting_grating(Screen='', Nrepeat=1, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "grating",
        "movie_refresh_freq":30.0,
        "Screen": "%s",
        "units":"cm",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 2,
        "speed": 2,
        "phase": 0,
        "spatial-freq": 0.04,
        "x-center": %.1f,
        "y-center": %.1f,
        "radius": 200.0,
        "angle": 0.0,
        "-----------------------------------------------------------------------2":0,
        "contrast-1": 0.2, "contrast-2": 1.0, "N-contrast": 3,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i
}  """ % (Screen, x, y, Nrepeat)


###########################################
###      optogenetic - only              ##
###########################################

def grey_screen(Screen='', Nrepeat=1, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "uniform_bg",
        "movie_refresh_freq":30.0,
        "Screen": "%s",
        "units":"cm",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 2,
        "screen-color": 0.5,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i
}  """ % (Screen, Nrepeat)


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
        "Image-ID": 1,
        "-----------------------------------------------------------------------2":0,
        "contrast-1": 0.2, "contrast-2": 1.0, "N-contrast": 3,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i 
}  """ % (Screen, Nrepeat)


###########################################
###      looming                        ###
###########################################

def looming(Screen='', Nrepeat=1, x=0, y=0):
  return """{
        "Presentation": "Stimuli-Sequence",
        "Stimulus": "looming_stim",
        "Screen": "%s",
        "units": "lin-deg",
        "-----------------------------------------------------------------------1":0,
        "presentation-duration": 5.0,
        "looming-duration": 1.0,
        "end-duration": 4.0,
        "looming-nonlinearity": 3,
        "radius-start":0.5,
        "radius-end":120,
        "x-center":%.1f,
        "y-center":%.1f,
        "-----------------------------------------------------------------------3":0,
        "N-repeat": %i 
} """ % (Screen, x, y, Nrepeat)


def MultiProtocol(Screen='', 
                  Nrepeat=None, 
                  interstim=6.0,
                  #   jitter=2.0,
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
        "presentation-blank-screen-color": 0.5,
    """ % (Screen, interstim)

    i = 1 # protocol counter
    for protocol_func, protocol_name, nrepeats in zip(
                    [center_patch, ff_drifting_grating, grey_screen, natural_images, looming],
                    ['center-patch', 'natural-images', 'grey-screen', 'natural-images', 'looming'],
                    [20, 20, 20, 20, 10]):

        if Nrepeat is not None:
            n=Nrepeat
        else:
            n=nrepeats

        protocol = protocol_func(Screen=Screen, Nrepeat=n, x=x, y=y)

        with open('protocol-%s.json' % protocol_name, 'w') as f:
            f.write(protocol)

        multiprotocol += '  "Protocol-%i": "protocol-%s.json",\n' % (i, protocol_name)

        i+=1

    multiprotocol = multiprotocol[:-2]+'}'

    return multiprotocol


if 0:
  build_movie(looming(Screen=Screen, Nrepeat=1, x=x, y=y),
              name='looming', rm=False)
if 0:
  x, y =0, 0
  build_movie(center_patch(Screen=Screen, Nrepeat=1, x=x, y=y),
              name='center-patch')
if 1:
  build_movie(MultiProtocol(Screen=Screen, Nrepeat=None), # Nrepeat=None to have the desired ones
              name='vision-survey+1sPrePostOpto',
              rm=False)