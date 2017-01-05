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
def experiment(params, folder_name, pool_size = 3):
    params['folder name'] =  folder_name
    param_combos = [[sam, target, cd, size, a, alpha]
                    for cd in ['dm_percconc']
                    for target in ['112', '111']
                    for sam in ['corpus','uniform']
                    for size in [7,10,12]
                    for a in [0.1,0.3,0.5]
                    for alpha in [0.1,0.3,0.5]]
    print('n parameter combos: %d' % len(param_combos))
    features = ['input sampling responses', 'target language', 
                'conceptual data', 'som size', 'som a', 'som alpha']
    shorthands = ['sam', 'language', 'cd', 'size', 'a', 'alpha']
    #
    arguments = [(param_combo,params,features,shorthands,ix)
                 for ix,param_combo in enumerate(param_combos)]
    pool = multiprocessing.Pool(processes = pool_size)
    As = pool.map(train_and_test_and_discriminate, arguments)
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
    d.read_discrimination_data(params)
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
        c.train(dump = dump, test = True)
        # trains the model incrementally, testing it on all items regularly
        c.discriminate() 
        # replicates the color discrimination experiments of Winawer et al.

###
# MAIN
###
def main():
    params = parameters.parameters
    experiment(params, sys.argv[1], int(sys.argv[2]))

if __name__ == "__main__":
    main()
