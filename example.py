"""purple rain.

example of a foreground/background type model. we train on the band and then
extract that from the mix leaving the vocal. 

clean results take a while. this takes about 30 minutes on my machine. a good
way to cut down on the time is to reduce the number of iters, but it also 
makes the results less independent.

"""

import sndsplt as splt
import time as time
reload(splt)

#-----------------------------------------------------------------------------#

sources_info = [ ('clips/voice_muted.wav', 50, 'band') ]
mix_info     =   ('clips/full_mix.wav',    20, 'prince')

classifier = 'PLCA'

params1 = {
    'iters':50, 
    'win':(10,24),
    'feature':'stft', 
    'nfft':4096, 
    'wfft':4096, 
    'nhop':1024,
    'window':'hann'
}

params2 = {
    'iters':50, 
    'win':(10,24),
    'feature':'stft', 
    'nfft':4096, 
    'wfft':4096, 
    'nhop':1024, 
    'window':'hann'
}

priors = {
    'alphaW':0, 
    'alphaZ':0, 
    'alphaH':0,
    'betaW':0, 
    'betaZ':0, 
    'betaH':0
}

#-----------------------------------------------#

# init time it
start = time.time()

# train model
sources = splt.train(sources_info, classifier=classifier, priors=priors, **params1)

# fit to model
sources, residual, mix = splt.fit(mix_info, sources, classifier=classifier, priors=priors, **params2)

# write audio
splt.write(sources+[residual], classifier=classifier, mix=mix)

# write info
splt.write_info(sources_info, mix_info, params1)

# print time it
dur = time.time() - start
print '\nin {0} min, {1} sec\n'.format(int(dur/60.0), dur%60.0)
