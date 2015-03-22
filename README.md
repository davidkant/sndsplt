#sndsplt

happy valley source separation

###what
* really just a few conveniences for handling PLCA sound decomposition (all of the heavy lifting is done by Bregman Toolkit). __sndsplit__ manages training and selective fitting of components. it allows you to train a collection of sources and decompose a mixture in terms of these sources + unknown components. this is particularly effective for sound separation when components can be isolated for training.

###depends on
* my fork of Bregman Toolkit modified to allow selective updating of PLCA components
