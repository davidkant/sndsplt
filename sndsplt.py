"""the module."""

import bregwoman
from bregwoman.suite import *
import numpy as np

"""

    Source

    - filename
    - num_components
    - name
    - sr
    - audiofile[]
      - x
    - F[]
      - Features
    - train[]
      - w, z, h, n, r, l
    - fit[]
      - w, z, h, n, r, l

dk

"""

#-----------------------------------------------------------------------------#

def train(sources_info, iters=50, feature='stft', nfft=8096, wfft=8096, 
    nhop=4096, window='hann', verbose=True):

    """Train.

    blah blah blah

    """

    # list of sources
    sources = []

    #-------------------------------------------#

    """init sources."""

    class Source: pass

    for filename, num_components, name in sources_info:

        s = Source()

        s.filename = filename
        s.num_components = num_components
        s.name = name
        sources += [s]

    #-------------------------------------------#

    """load source audiofiles."""

    class Audiofile: pass

    for s in sources:

        # read audiofile
        x, sr, pcm = scikits.audiolab.wavread(s.filename)

        # init channels
        s.audiofile = []

        # stuff channels
        for ci in range(x.shape[1]):

            s.audiofile.append(Audiofile())

            s.audiofile[ci].x = x[:,ci] 
            s.sr = sr

    #-------------------------------------------#

    """"extract features."""

    if verbose: print '\nextracting source features...'

    # init feature params
    p = Features.default_params()
    p['feature']=feature
    p['nfft']=nfft
    p['wfft']=wfft
    p['nhop']=nhop
    p['window']=window
    p['verbosity']=1
    p['nbpo']=96
    # p['intensify']=True

    for si,s in enumerate(sources):

        # init channels
        s.F = []

        # stuff channels
        for ci,channel in enumerate(s.audiofile):

            s.F.append(None)

            s.F[ci] = Features(s.audiofile[ci].x, p)

    #-------------------------------------------#

    """"learn source models."""

    if verbose: print '\nlearning source models...'

    class Train: pass

    for si,s in enumerate(sources):

        if verbose: print '#{0} n={2} {1}'.format(si, s.name, s.num_components)

        # init channels
        s.train = []

        # stuff channels
        for ci in range(len(s.F)):

            if (s.num_components <= 0): continue

            s.train.append(Train())

            s.train[ci].w, s.train[ci].z, s.train[ci].h, \
            s.train[ci].n, s.train[ci].r, s.train[ci].l = \
                PLCA.analyze(s.F[ci].X, s.num_components, niter=iters)
    
    #-------------------------------------------#

    """return."""

    return sources


#-----------------------------------------------------------------------------#

def fit(mix_info, sources, iters=50, feature='stft', nfft=8096, wfft=8096, 
    nhop=4096, window='hann', verbose=True):

    """Fit.

    blah blah blah

    """

    #-------------------------------------------#

    """init mix."""

    class Mix: pass

    filename, num_components, name = mix_info

    mix = Mix()

    mix.filename = filename
    mix.num_components = num_components
    mix.name = name

    #-------------------------------------------#

    """load mix audiofile."""

    class Audiofile: pass

    # read audiofile
    x, sr, pcm = scikits.audiolab.wavread(mix.filename)

    # init channels
    mix.audiofile = []

    # stuff channels
    for ci in range(x.shape[1]):

        mix.audiofile.append(Audiofile())

        mix.audiofile[ci].x = x[:,ci] 
        mix.sr = sr

    #-------------------------------------------#

    """"extract features."""

    if verbose: print '\nextracting mix features...'

    # init feature params
    p = Features.default_params()
    p['feature']=feature
    p['nfft']=nfft
    p['wfft']=wfft
    p['nhop']=nhop
    p['window']=window
    p['verbosity']=1
    p['nbpo']=96
    # p['intensify']=True

    # init channels
    mix.F = []

    # stuff channels
    for ci,channel in enumerate(mix.audiofile):

        mix.F.append(None)

        mix.F[ci] = Features(mix.audiofile[ci].x, p)

    #-------------------------------------------#

    """"fit source models."""

    if verbose: print '\nfitting source models...'

    class Fit: pass

    # init channels
    mix.fit = []

    # stuff channels
    for ci in range(len(mix.F)):

        if verbose: print 'channel {0}...'.format(ci)

        # sum of all source num_components + mix num_compnents
        total_num_components = sum([s.num_components for s in sources]) + \
            mix.num_components

        # all source w's concatenated
        W_trained = np.concatenate([s.train[ci].w for s in sources if 
            s.num_components > 0],1)

        # do not update the source components but do update the mix components
        W_update = [False] * sum([s.num_components for s in sources]) + \
            [True] * mix.num_components

        # okay, do it
        mix.fit.append(Fit())

        mix.fit[ci].w, mix.fit[ci].z, mix.fit[ci].h, \
            mix.fit[ci].n, mix.fit[ci].r, mix.fit[ci].l = \
            PLCA.analyze(mix.F[ci].X, total_num_components, niter=iters, 
                initW=W_trained, updateW=W_update)

    #-------------------------------------------#

    """parse fitted source data + residual."""

    # residual is what's left over
    class Residual: pass
    residual = Residual()
    residual.name = mix.name
    residual.num_components = mix.num_components
    residual.sr = mix.sr

    # mix is now entire mix so let's call it that   
    mix.name = 'mix'

    for si,s in enumerate(sources + [residual]):

        # range of components this source
        i = sum([s.num_components for s in (sources+[residual])[0:si]])
        j = sum([s.num_components for s in (sources+[residual])[0:si+1]])

        # init channels
        s.fit = []

        # stuff channels
        for ci in range(len(mix.fit)):

            s.fit.append(Fit())

            s.fit[ci].w = mix.fit[ci].w[:,i:j]
            s.fit[ci].z = mix.fit[ci].z[i:j]
            s.fit[ci].h = mix.fit[ci].h[i:j,:]

    #-------------------------------------------#

    """return."""

    return sources, residual, mix

#-----------------------------------------------------------------------------#

def resynth_source(s, mix=None):

    # resynthesized signal goes here
    y = []

    # for each channel
    for ci in range(len(s.fit)):

        W = s.fit[ci].w
        Z = s.fit[ci].z
        H = s.fit[ci].h

        if mix:
            fn = mix.F[ci].X / (mix.fit[ci].w.dot(
                np.diag(mix.fit[ci].z)).dot(mix.fit[ci].h))
            tf = (W.dot(np.diag(Z)).dot(H)) * fn

        else:
            tf = (W.dot(np.diag(Z)).dot(H))

        sig = mix.F[ci].inverse(tf, pvoc=False)  # note: use phase from original
        sig = np.atleast_1d(sig / (sig.max() + .005))  # normalize

        y += [sig]

    return y

def write(sources, mix=None):

    # create save folder
    foldername = 'test'
    if not os.path.exists(foldername): os.makedirs(foldername)

    # resynthesize Smaragdis style - the band
    for si,s in enumerate(sources):

        # tell me about it
        print '\nresynthesizing s{0}, {1}...'.format(si, s.name)

        # call resynth on source
        y = resynth_source(s, mix=mix)

        # write to disk
        filename = foldername + '/s{0}-{1}.wav'.format(si, s.name)
        wavwrite((np.array([y[0],y[1]])).transpose(), filename, s.sr) # FUCKING: hardcoded channels

#-----------------------------------------------------------------------------#

def write_info(sources_info, mix_info, params):

    # create save folder
    foldername = 'test'
    if not os.path.exists(foldername): os.makedirs(foldername)

    # write info file
    f = open(foldername + '/info.txt', 'w')
    f.write(str(sources_info) + '\n\n')
    f.write(str(mix_info) + '\n\n')
    f.write(str(params) + '\n\n')

    f.close
