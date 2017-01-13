import numpy as np

parameters = {
    'data' : 'biling',
    'folder name' : None,
    'subfolder name' : None,
    'target language' : ['113','111'], # '111' for color, should correspond to TXT and TXT if 'input sampling
    'moment of onset' : [0,0],
    'language share' : [.5,.5],
    # responses' ==
    # 'corpus'
    #
    'fold' : None,
    'test interval' : 100,
    'n simulations' : 5,
    'length simulation' : 30000,
    #
    'conceptual data' : 'dm_perc',
    'input sampling responses' : 'corpus', # uniform, corpus, siutation
    'frequency data' : 'frequencies',
    #
    'pca threshold' : 0.999,
    'leave-one-out' : False,
    # {True,False} TODO: works for GCM and GNB, but for ALCOVE (given dim. weights)??
    'leave target language out' : True,
    'classifier' : 'som', # {gnb, gcm, alcove, som}
    'distance metric' : 'euclidean',
    # {'euclidean', 'manhattan'} only if 'pca preprocessing matrix' = 'distance matrix'
    #
    # GCM PARAMETERS
    'gcm prior' : 'uniform', # or input
    'gcm c' : 1.0,
    'gcm r' : 2.0,
    'gcm p' : 2.0,
    # ALCOVE PARAMETERS
    'alcove c' : 1.0,
    'alcove q' : 1.0,
    'alcove r' : 2.0,
    'alcove phi' : 0.5,
    'alcove lambda_a' : 0.001,
    'alcove lambda_w' : 0.2,
    'alcove attention weight initialization' : 'eigenvalues',
    # {'uniform','eigenvalues'}
    #
    # SOM PARAMETERS
    'som neighborhood function' : 'gaussian',
    'som alpha' : 0.25,
    'som a' : 0.2,
    'som c' : 1.0,
    'som size' : 6,
    'som sigma_0' : 5.0,
    'som lambda_sigma' : 20000,
    'som n pretrain' : 0,
    'som initialization bandwidth' : 0.01,
    'som neighborhood' : 'euclidean', # 'vonneuman'
    'som delta sigma' : 0.1,
    #
    # DISCRIMINATION EXPERIMENTS
    'discrimination data' : 'winawer'
    }
