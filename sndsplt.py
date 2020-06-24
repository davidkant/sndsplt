"""the module."""

import bregman
from bregman.suite import *
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

def train(sources_info, classifier='PLCA', win=(5,12), iters=25,
    feature='stft', nfft=8096, wfft=8096, nhop=4096, window='hann',
    priors=None, verbose=True):

    """Train.

    blah blah blah

    """

    #-------------------------------------------#

    """initialize."""

    # init classifier and window
    if classifier == 'PLCA':
        classifier = PLCA
        win = None
    elif classifier == 'SIPLCA':
        classifier = SIPLCA
        win = win[0]
    elif classifier == 'SIPLCA2':
        classifier = SIPLCA2
        win = win
    else:
        raise NameError('So sorry but I didn\'t recognize that classifier. ' \
            'Supported classifiers are \'PLCA\', \'SIPLCA\', and \'SIPLCA2\'.')

    # tell me about it
    if verbose:
        print '\nTRAINING!'
        print 'classifier: {0}'.format(classifier)
        print 'window: {0}'.format(win)
        print 'iters: {0}'.format(iters)
        print 'feature: {0}'.format(feature)

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
                classifier.analyze(s.F[ci].X, s.num_components, niter=iters,
                    win=win, **priors)

    #-------------------------------------------#

    """return."""

    return sources

def fit(mix_info, sources, classifier='PLCA', win=(5,12), iters=25,
    feature='stft', nfft=8096, wfft=8096, nhop=4096, window='hann',
    priors=None, verbose=True):

    """Fit.

    blah blah blah

    """

    #-------------------------------------------#

    """initialize."""

    # init classifier and window
    if classifier == 'PLCA':
        classifier = PLCA
        win = None
    elif classifier == 'SIPLCA':
        classifier = SIPLCA
        win = win[0]
    elif classifier == 'SIPLCA2':
        classifier = SIPLCA2
        win = win
    else:
        raise NameError('So sorry but I didn\'t recognize that classifier. ' \
            'Supported classifiers are \'PLCA\', \'SIPLCA\', and \'SIPLCA2\'.')

    # tell me about it
    if verbose:
        print '\nFITTING!'
        print 'classifier: {0}'.format(classifier)
        print 'window: {0}'.format(win)
        print 'iters: {0}'.format(iters)
        print 'feature: {0}'.format(feature)

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
            s.num_components > 0], axis=1)

        # do not update the source components but do update the mix components
        W_update = [False] * sum([s.num_components for s in sources]) + \
            [True] * mix.num_components

        # okay, do it
        mix.fit.append(Fit())

        mix.fit[ci].w, mix.fit[ci].z, mix.fit[ci].h, \
            mix.fit[ci].n, mix.fit[ci].r, mix.fit[ci].l = \
            classifier.analyze(mix.F[ci].X, total_num_components, niter=iters,
                initW=W_trained, updateW=W_update, win=win, **priors)

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

            # w and h will have extra dims for SIPLA and
            # SIPLCA2 but we don'thave to change the code

    #-------------------------------------------#

    """return."""

    return sources, residual, mix

def resynth_source(s, classifier='PLCA', mix=None):

    # init classifier and window
    if classifier == 'PLCA':
        classifier = PLCA
    elif classifier == 'SIPLCA':
        classifier = SIPLCA
    elif classifier == 'SIPLCA2':
        classifier = SIPLCA2
    else:
        raise NameError('So sorry but I didn\'t recognize that classifier. ' \
            'Supported classifiers are \'PLCA\', \'SIPLCA\', and \'SIPLCA2\'.')

    # resynthesized signal goes here
    y = []

    # for each channel
    for ci in range(len(s.fit)):

        # reconstruct full mix
        WZH_mix = classifier.reconstruct(mix.fit[ci].w, mix.fit[ci].z, mix.fit[ci].h)

        # reconstruct source
        WZH = classifier.reconstruct(s.fit[ci].w, s.fit[ci].z, s.fit[ci].h)

        if mix:
            fn = mix.F[ci].X / WZH_mix
            tf = WZH * fn

        else:
            tf = WZH

        sig = mix.F[ci].inverse(tf, pvoc=False)  # note: use phase from original
        sig = np.atleast_1d(sig / (sig.max() * 0.95))  # normalize

        y += [sig]

    return y

def write(sources, classifier='PLCA', mix=None, regionname='', foldername='test'):

    # create save folder
    if not os.path.exists(foldername): os.makedirs(foldername)

    # resynthesize Smaragdis style - the band
    for si,s in enumerate(sources):

        # tell me about it
        print '\nresynthesizing s{0}, {1}...'.format(si, s.name)

        # call resynth on source
        y = resynth_source(s, classifier=classifier, mix=mix)

        # write to disk
        filename = foldername + '/' + regionname + '_s{0}-{1}.wav'.format(si, s.name)
        wavwrite((np.array([y[0],y[1]])).transpose(), filename, s.sr)
        # FUCKING: hardcoded channels

def write_components(sources, classifier='PLCA', mix=None, regionname='', foldername='test'):

    # create save folder
    if not os.path.exists(foldername): os.makedirs(foldername)

    # resynthesize Smaragdis style - the band
    for si,s in enumerate(sources):

        # tell me about it
        print '\nresynthesizing s{0}, {1} component-wise...'.format(si, s.name)

        # loop through components
        for ni in range(s.num_components):

            # call resynth on source
            y = resynth_source(s, classifier=classifier, mix=mix, component=ni)

            # # write to disk
            # filename = foldername + '/s{0}-{1}-{2}.wav'.format(si, s.name, ni)
            # wavwrite((np.array(y)).transpose(), filename, s.sr)

            # write to disk
            filename = foldername + '/' + regionname + '_s{0}-{1}-{2}-L.wav'.format(si, s.name, ni)
            wavwrite((np.array(y[0])).transpose(), filename, s.sr)
            filename = foldername + '/' + regionname + '_s{0}-{1}-{2}-R.wav'.format(si, s.name, ni)
            wavwrite((np.array(y[1])).transpose(), filename, s.sr)

def write_info(sources_info, mix_info, params, regionname='', foldername='test'):

    # create save folder
    if not os.path.exists(foldername): os.makedirs(foldername)

    # write info file
    f = open(foldername + '/' + regionname + '_info.txt', 'w')
    f.write(str(sources_info) + '\n\n')
    f.write(str(mix_info) + '\n\n')
    f.write(str(params) + '\n\n')

    f.close
