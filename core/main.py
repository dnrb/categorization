from classifiers import som
from data import data
import parameters
import sys
import multiprocessing
import os
#
import dill as pickle
from subprocess import call

##
# SHELL FOR EXPERIMENT
##
def experiment(params, folder_name, pool_size = 1):
    params['folder name'] = folder_name
    # YM: changed TARGET class, added onset and share
    param_combos = [[sam, target, cd, size, a, alpha, onset, share]
                    for cd in ['dm_biling_test']
                    for target in [['italian', 'english']]
                    for sam in ['corpus','uniform']
                    for size in [7,10,12]
                    for a in [0.1,0.3,0.5]
                    for alpha in [0.1,0.3,0.5]
                    for onset in [[0, 0], [0, 0.5], [0, 1]]
                    for share in [[1, 0], [0.75, 0.25], [0.5, 0.5], [0, 1]]
                    ]
    print('n parameter combos: %d' % len(param_combos))
    features = ['input sampling responses', 'target language', 'conceptual data', 'som size',
                'som a', 'som alpha', 'moment of onset', 'language share']
    shorthands = ['sam', 'target', 'language', 'cd', 'size', 'a', 'alpha', 'onset', 'share']
    #
    arguments = [(param_combo,params,features,shorthands,ix)
                 for ix,param_combo in enumerate(param_combos)]
    for a in arguments:
        train_and_test_and_discriminate(a)
    return

###
# POOL FUNCTIONS
###
def train_and_test_and_discriminate(arguments):
    param_combo, params, features, shorthands, ix = arguments
    print('train and test %d' % ix)
    params = { k:v for k,v in params.items() }
    params['subfolder name'] = '_'.join(['%s_%s'%(sh,pv) for sh,pv in 
                                         zip(shorthands,param_combo)])
    for pv, fe in zip(param_combo, features): params[fe] = pv
    d = data(params)
    # #YM: not needed for this experiment
    # d.read_discrimination_data(params)
    d.dirname = '%s/%s' % (params['folder name'], params['subfolder name'])
    print(os.path.isdir(d.dirname))
    if os.path.isdir(d.dirname): 
        return
        # doesn't run experiments if this parameter setting has been run before
    os.makedirs(d.dirname)
    with open('%s/parameters.p' % d.dirname, 'wb') as fh:  
        pickle.dump(params, fh)
    dump = True
    for simulation in range(params['n simulations']):
        print(simulation)
        c = som(d, params, simulation)
        c.trainM(dump = dump, test = True)
        # trains the model incrementally, testing it on all items regularly
        # c.discriminate()
        # replicates the color discrimination experiments of Winawer et al.
        # YM: not applicable

###
# MAIN
###
def main():
    params = parameters.parameters
    experiment(params, sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    main()
