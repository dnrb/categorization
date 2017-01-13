from sklearn.naive_bayes import GaussianNB
# DELEER from scipy.stats import entropy as kl
# DELEER from scipy.stats import pearsonr
import numpy as np
from collections import Counter
from collections import defaultdict as dd
# DELEER import math
# MIGRATE from sklearn.metrics import confusion_matrix as cm
# DELEER import csv
#from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
import pickle as pickle
from sklearn.metrics import pairwise_distances
import os

A = np.array
P = np.power
S = np.sum

class classifier:
    # classifier; contains shared functions between the various categorization 
    # models
    
    def __init__(self, data, parameters, simulation):
        # set central variables and parameters
        self.time = 0
        self.data = data
        self.parameters = parameters
        self.simulation = simulation
        self.target_language = parameters['target language']
        self.input_sampling_responses = parameters['input sampling responses']
        self.n_iterations = parameters['length simulation']
        self.len_interval = parameters['test interval']
        #
        self.initialize_model(parameters)
        #
        self.test_set = self.init_testset()
        # model-specific initialization

    def init_testset(self):
        n_folds = 211 if self.parameters['target language'] == '111' else 49
        if self.parameters['fold'] == None: return []
        f = self.parameters['fold']
        s_per_t = [[] for i in range(self.data.nT)]
        for s,v in enumerate(self.data.P_s_given_t.T):
            if v.sum() == 0: continue                                     
            else: s_per_t[v.argmax()].append(s)   
        testset = []
        folds = [[] for i in range(n_folds)]
        current = 0
        for i in sorted(range(len(s_per_t)), key = lambda t : -len(s_per_t)):
            for j in s_per_t[i]:
                folds[current % n_folds].append(j)
                current += 1
        print(sum(len(fi) for fi in folds))
        return folds[f]


    def sample_input_item(self):
        # returns a sampled vector of feature-values (reals) for a situation 
        # and a term (integer)
        situation = None
        if (self.input_sampling_responses == 'corpus' or 
            self.input_sampling_responses == 'uniform'):
            # sampling on the basis of corpus/uniform term frequencies and 
            # P(s|t)
            while (situation == None or situation in self.test_set):
                term = np.random.choice(self.data.nT, 1, p = self.data.P_t)[0]
                p_s_given_t = self.data.P_s_given_t[term]
                situation = np.random.choice(self.data.nS, 1, p=p_s_given_t)[0]
        elif self.input_sampling_responses == 'situation':
            # sampling on the basis of a uniform distribution over situations 
            # and P(t|s)
            while (situation == None or 
                   self.data.max_P_t_given_s[situation] == -1):
                situation = np.random.choice(self.data.nS)[0]
            p_t_given_s = self.data.P_t_given_s[situation]
            term = np.random.choice(self.data.nT, 1, p = p_t_given_s)[0]
        #
        coordinates = self.data.situations[situation]
        return coordinates, term

    def sample_input_itemM(self):
        # returns a sampled vector of feature-values (reals) for a situation
        # and a term (integer)
        situation = None
        if (self.input_sampling_responses == 'corpus' or
            self.input_sampling_responses == 'uniform'):
            # sampling on the basis of corpus/uniform term frequencies and
            # P(s|t)
            while (situation == None or situation in self.test_set or situation not in self.data.situations):
                term = np.random.choice(self.data.nT[self.lang], 1, p = self.data.P_t[self.lang])[0]
                p_s_given_t = self.data.P_s_given_t[self.lang][term]
                #situation = np.random.choice(self.data.nS, 1, p=p_s_given_t)[0]
                situation = np.random.choice(self.data.nS, 1, p=p_s_given_t)[0]
        elif self.input_sampling_responses == 'situation':
            # sampling on the basis of a uniform distribution over situations
            # and P(t|s)
            while (situation == None or
                   self.data.max_P_t_given_s[situation] == -1):
                situation = np.random.choice(self.data.nS)[0]
            p_t_given_s = self.data.P_t_given_s[situation]
            term = np.random.choice(self.data.nT[self.lang], 1, p = p_t_given_s)[0]
        #
        coordinates = self.data.situations[situation]
        return coordinates, term

    def train(self, test = False, dump = False):
        # training the model for n_iterations iterations, writing away the 
        # state of the model
        # every len_interval iterations if dump == True. Runs test() if test 
        # == true.
        d = self.data
        while self.time < self.n_iterations:
            #print(self.time)
            self.time += self.len_interval
            self.fit()
            if dump and self.time > 29950: self.dump()
            if test: self.test()
        return

    def trainM(self, test = False, dump = False):
        # training the model for n_iterations iterations, writing away the
        # state of the model
        # every len_interval iterations if dump == True. Runs test() if test
        # == true.
        d = self.data
        while self.time < self.n_iterations:
            #print(self.time)
            self.time += self.len_interval
            self.fitM()
            print(self.get_term_mapM())
            if dump and self.time > 29950: self.dump()

            if test: self.testM()
        return

    def test(self):
        # tests the whole set of situations and writes both convergence with
        # adult modal naming behavior as well as the distributions over terms
        # per situation to output files.
        development_fn = '%s/development.csv' % self.data.dirname
        convergence_fn = '%s/convergence.csv' % self.data.dirname
        self.development_fh = open(development_fn, 'a')
        self.convergence_fh = open(convergence_fn, 'a')
        if os.path.getsize(development_fn) == 0:
            self.development_fh.write('simulation,time,situation,%s\n' %
                                        ','.join(self.data.terms))
        if os.path.getsize(convergence_fn) == 0:
            self.convergence_fh.write('simulation,time,score\n')
        #
        posterior = self.predict_termsM(self.data.situations)
        predicted_best_term = posterior.argmax(1)
        mpt = self.data.max_P_t_given_s
        if self.test_set == []:
            predictions = (predicted_best_term == mpt)[mpt != -1]
        else:
            predictions = (predicted_best_term == mpt)[self.test_set]
        self.convergence_fh.write('%d,%d,%.3f\n' %
                                  (self.simulation, self.time,
                                   np.mean(predictions)))
        for i in range(self.data.nS):
            if i not in self.test_set and len(self.test_set) > 0: continue
            self.development_fh.write('%d,%d,%d,%s\n' %
                                      (self.simulation, self.time, i,
                                        ','.join(['%.3f' % p
                                                  for p in posterior[i]])))
        self.development_fh.close()
        self.convergence_fh.close()
        return

    def testM(self):
        # tests the whole set of situations and writes both convergence with 
        # adult modal naming behavior as well as the distributions over terms 
        # per situation to output files.
        development_fn = '%s/development.csv' % self.data.dirname
        convergence_fn = '%s/convergence.csv' % self.data.dirname
        self.development_fh = open(development_fn, 'a')
        self.convergence_fh = open(convergence_fn, 'a')
        if os.path.getsize(development_fn) == 0: 
            self.development_fh.write('simulation,time,situation,%s\n' % 
                                        ','.join(t for terms in self.data.terms.values() for t in terms))
        if os.path.getsize(convergence_fn) == 0: 
            self.convergence_fh.write('simulation,time,%s\n' % ','.join(['score-'+l for l in self.target_language]))
        #
        posterior = self.predict_termsM(self.data.situations)
        predictions = dd(lambda: dd(int))
        for i in range(len(self.target_language)):
            l = self.target_language[i]
            predictions[l]
            for situation in self.data.situations.keys():
                predictions[l][situation]
            if self.current_exposure[i]:
                predicted_best_term = dd()
                for situation in sorted(posterior[l].keys()):
                    predicted_best_term[situation] = posterior[l][situation].argmax(1)[0]
                mpt_full = self.data.max_P_t_given_s[l]
                mpt = dict([(i,mpt_full[i]) for i in sorted(self.data.situations.keys())])
                if self.test_set == []:
                    # a1 = A(predicted_best_term)
                    # a2 = A(mpt)
                    # x = list((a1 == a2)[a2 != -1])
                    # predictions[l] = x
                    # predictions[l] = dd([(situation, mpt[situation]==predicted_best_term[situation])
                    #                       for situation in self.data.situations.keys()])
                    for situation in self.data.situations.keys():
                        predictions[l][situation] = (mpt[situation] == predicted_best_term[situation])
            # else:
            #     predictions.append((A(predicted_best_term) == A(mpt))[self.test_set])
        self.convergence_fh.write('%d,%d,' % (self.simulation, self.time))
        for l in self.target_language:
            self.convergence_fh.write('%.3f,' % (np.mean(list(predictions[l].values()))))
        self.convergence_fh.write('\n')
        #for i in range(self.data.nS):
        for i in self.data.situations.keys():
            if i not in self.test_set and len(self.test_set) > 0: continue
            self.development_fh.write('%d,%d,%d,' % (self.simulation, self.time, i))
            for l in self.target_language:
                self.development_fh.write('%s,' % ','.join(['%.3f' % p
                                                    for p in posterior[l][i][0]]))
            self.development_fh.write('\n')
        self.development_fh.close()
        self.convergence_fh.close()
        self.calculate_centroids()
        return
        #

    def calculate_centroids(self):
        # BB added 170113 for bilingualism experiment
        # pickles a dictionary of (language,term) pairs to arrays of average feature values for that term
        # (as given by a weighted mean over the map)
        start_features = np.sum([self.data.nT[l] for l in self.data.target_language])
        global_t_ix = 0
        
        centroids = {}
        for l in self.data.target_language:
            p_t_given_c = self.map[:,:,global_t_ix : global_t_ix+self.data.nT[l]].copy()
            for i in range(self.parameters['som size']):
                for j in range(self.parameters['som size']):
                    p_t_given_c[i,j] /= self.map[i,j,global_t_ix : global_t_ix+self.data.nT[l]].sum()
            for t_ix, t in enumerate(self.data.terms[l]):
                weighted_features = []
                for f in range(start_features,self.map.shape[2]):
                    weighted_features.append((p_t_given_c[:,:,t_ix] * self.map[:,:,f]).sum()/p_t_given_c[:,:,t_ix].sum())
                print(l,t,' '.join(['%.2f' % v for v in weighted_features[:10]]))
                centroids[(l,t)] = np.array(weighted_features)
                
            global_t_ix += self.data.nT[l]
        pickle.dump(centroids, open('%s/centroids_%d_%d.p' % (self.data.dirname, self.simulation, self.time),'wb'))

class gnb(classifier):
    # Gaussian Naive Bayes
    def initialize_model(self, parameters): 
        self.X, self.Y = [], []
        return

    def fit(self):
        new_X, new_Y = zip(*[self.sample_input_item() 
                             for i in range(self.len_interval)])
        self.X.extend(new_X)
        self.Y.extend(new_Y)
        self.classifier = GaussianNB()
        self.classifier.fit(self.X, self.Y)
        return

    def dump(self):
        # pickles the sklearn GaussianNB classifier
        fn = '%s/model_%d_%d.p' % (self.data.dirname,self.simulation,self.time)
        with open(fn,'wb') as fh: pickle.dump((self.classifier, self.Y), fh)
        return

    def load(self, simulation, time):
        # loads the pickled sklearn GaussianNB classifier
        self.simulation = simulation
        self.time = time
        fn = '%s/model_%d_%d.p' % (self.data.dirname, simulation, time)
        with open(fn, 'rb') as fh: self.classifier, self.Y = pickle.load(fh)
        return

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test 
        # item i in I from a matrix I x F for I test items each with F 
        # features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        posterior_incomplete = self.classifier.predict_proba(test_items)
        posterior = np.zeros((test_items.shape[0], self.data.nT))
        for y_ix, y in enumerate(sorted(set(self.Y))):
            posterior[:,y] += posterior_incomplete[:,y_ix]
        return posterior
        

class gcm(classifier):
    # Generalized Context Model
    # REFERENCE: Nosofsky, R. M. (1987). Attention and learning processes in 
    #  the identification and categorization of integral stimuli. Journal of 
    #  Experimental Psychology: Learning, Memory, and Cognition, 13(1), 87-108.
    # TODO: test and check if rightly implemented ~ don't use before doing this

    def initialize_model(self, parameters): 
        self.prior_type = parameters['gcm prior']
        self.c = parameters['gcm c']
        self.r = parameters['gcm r']
        self.p = parameters['gcm p']
        self.X, self.Y = [], []#np.zeros((0,self.data.nF)), np.zeros((0))
        #
        self.memorized_summed_eta = np.zeros((self.data.nS, self.data.nT))
        return
    
    def fit(self): 
        new_X, new_Y = zip(*[self.sample_input_item() 
                             for i in range(self.len_interval)])
        self.X.extend(new_X)
        self.Y.extend(new_Y)
        return

    def dump(self):
        # pickles the X and Y vectors (situations and terms).
        fn = '%s/model_%d_%d.p' % (self.data.dirname,self.simulation,self.time)
        with open(fn,'wb') as fh:
            pickle.dump((self.X,self.Y), fh)

    def load(self, simulation, time):
        self.simulation = simulation
        self.time = time
        #
        fn = '%s/model_%d_%d.p' % (self.data.dirname, simulation, time)
        with open(fn, 'rb') as fh:
            self.X, self.Y = pickle.load(fh) 
        return

    def predict_terms(self, test_items):
        # returns an IT matrix of posterior probabilities over T per test 
        # item i in I from a matrix IF for I test items each with F features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        #
        if test_items.shape[0] == self.data.nS:
            X = A(self.X[-self.len_interval:])
            Y = A(self.Y[-self.len_interval:])
        else: X,Y = A(self.X), A(self.Y)
        #
        if self.prior_type == 'input':
            counts = np.zeros((test_items.shape[0], self.data.nT))
            counts_incomplete = Counter(Y)
            for y,v in counts_incomplete.items(): counts[:,y] = v
            
        elif self.prior_type == 'uniform':
            counts = np.ones((test_items.shape[0], self.data.nT))
        b = normalize(counts, norm = 'l1', axis = 1)
        #
        self.a_dim = np.ones((X.shape[0], self.data.nF))
        #
        # d_abs = np.abs( self.X  - test_item )
        # TODO this is suboptimal -- X and Y shd be np arrays from getgo
        # distances = self.c * P(S(P(self.a_dim * d_abs,
        #  self.r), 1), 1.0/self.r)
        distances = A([self.c*np.linalg.norm(self.a_dim * (X-test_items[i]),
                                             ord=self.r,axis=1)
                       for i in range(test_items.shape[0])])
                       # TODO X is subopt, shd be np.A
        etas = np.exp(-P(distances, self.p))
        #
        if test_items.shape[0] == self.data.nS:
            summed_eta = self.memorized_summed_eta.copy()
        else: summed_eta = np.zeros((test_items.shape[0], self.data.nT))
        for i in range(test_items.shape[0]):
            for y,a in zip(Y.astype('int'),etas[i]): summed_eta[i][y] += a
        if test_items.shape[0] == self.data.nS:
            self.memorized_summed_eta = summed_eta.copy()
        #
        return normalize(b * summed_eta, norm = 'l1', axis = 1)
        #return (b * summed_eta) / S(b * summed_eta)
 
class alcove(classifier):
    # ALCOVE
    # REFERENCE: Kruschke, J. K. (1992). ALCOVE: An exemplar-based 
    #  connectionist model of category learning. Psychological Review, 99(1),
    #  22-44. 

    def initialize_model(self, parameters):
        # parameters
        self.a_dim_initialization = parameters['alcove attention weight initialization']
        self.lambda_w = parameters['alcove lambda_w']
        self.lambda_a = parameters['alcove lambda_a']
        self.c = parameters['alcove c']
        self.q = parameters['alcove q']
        self.r = parameters['alcove r']
        self.phi = parameters['alcove phi']
        # initializes dimension weights a_dim, either as uniform or as set to 
        # the relative importance of the features in the data.
        if self.a_dim_initialization == 'uniform':
            self.a_dim = (np.ones(self.data.situations.shape[1])/
                                 self.data.situations.shape[1])
        elif self.a_dim_initialization == 'eigenvalues':
            self.a_dim = self.data.dim_weights.copy()
        self.hidden = A([])
        self.w = A([])
        self.out = A([])
        # here I deviate from Kruschke's formulation -- my implementation 
        # starts with incrementally growing network, because otherwise weights
        # self.w between hidden layer self.hidden and output layer self.out 
        # will be negatively affected without the elements in self.hidden and
        # self.out being observed. The network `grows' incrementally with the 
        # self.update_nodes() function. 
        # TODO: implement original formulation and parametrize this choice.   
    
    def fit(self):
        # trains the model for len_interval input items
        for i in range(self.len_interval):
            a_in, k = self.sample_input_item()
            self.update_nodes(a_in, k)
            a_hid, a_out = self.get_activations(a_in)
            self.backprop(a_in, a_hid, a_out, k)
        return

    def update_nodes(self, a_in, k):
        # makes the network's hidden and output layer grow. See comment in 
        # initialize_model()
        w_init = 0.0
        new_k = k not in self.out
        new_a_in = not any(np.all(a_in == x) for x in self.hidden)
        no_a_in = self.hidden.shape[0] == 0
        #
        if no_a_in:
            self.hidden = A([a_in])
            self.out = np.concatenate((self.out,A([k])), axis = 0)
            self.w = A([A([w_init])])
        else:
            if new_k:
                new_row = A([A([np.random.uniform(-0,0)]) for k_ in 
                                range(self.w.shape[1])]).T
                self.out = np.concatenate((self.out,A([k])), axis = 0)
                self.w = np.concatenate((self.w, new_row), axis = 0)
            if new_a_in:
                new_col = A([A([np.random.uniform(-0,0) for h_ in 
                                range(self.out.shape[0])])]).T
                self.hidden = np.concatenate((self.hidden, [a_in]), axis = 0)
                self.w = np.concatenate((self.w, new_col), axis = 1)
        return
        
    def get_activations(self, a_in):
        # calculates the activation values for the hidden layer and the output 
        # layer of ALCOVE given an input signal a_in
        a_hid = np.exp(-self.c * 
                        P(S(P(self.a_dim * 
                                                 (self.hidden-a_in), self.r),
                                                  axis = 1), 
                                 self.q/self.r))
        # a_hid = e^(-c(SUM_i(a_dim_i * |h_ji - a_in_i|)^r)^(q/r)
        a_out = S(a_hid * self.w, axis = 1)
        # a_out = SUM_j(w_kj * a_hid_j)
        return a_hid, a_out

    def backprop(self, a_in, a_hid, a_out, k):
        # backpropagates the error signal on the basis of the activation 
        # values over the hidden layer and the output layer, given a correct 
        # category k and an input signal a_in
        t_k = A([(max(1, a_out[ix]) if k_ == k else min(-1, a_out[ix]))
                 for ix,k_ in enumerate(self.out)])
        # teacher values
        error = t_k - a_out
        delta_w = self.lambda_w * np.outer(error, a_hid)
        delta_a_dim = (-self.lambda_a) * S((S(self.w.T * error, 
                                                        axis = 1) * a_hid)
                                                * (self.c * np.abs(self.hidden-
                                                    a_in)).T, 1)
        # calculate deltas
        self.w += np.nan_to_num(delta_w)
        # update w
        self.a_dim += np.nan_to_num(delta_a_dim)
        self.a_dim[self.a_dim < 0.0] = 0.0
        # update a_dim and clip negative values

    def dump(self):
        # pickles the layers of the neural network
        fn = '%s/model_%d_%d.p' % (self.data.dirname,self.simulation,self.time)
        with open(fn,'wb') as fh:
            pickle.dump((self.a_dim, self.hidden, self.out, self.w), fh)
        return

    def load(self, simulation, time):
        # loads the pickled NN layers
        self.simulation = simulation
        self.time = time
        fn = '%s/model_%d_%d.p' % (self.data.dirname, simulation, time)
        with open(fn, 'rb') as fh:
            self.a_dim, self.hidden, self.out, self.w = pickle.load(fh) 
        return

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test 
        # item i in I from a matrix I x F for I test items each with F 
        # features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        e_a_outs = []
        for a_in in test_items:
            a_hid, a_out = self.get_activations(a_in)
            e_a_out = np.nan_to_num(np.exp(self.phi * a_out))
            e_a_outs.append(e_a_out)
        normalized = normalize(A(e_a_outs), norm = 'l1', axis = 1)
        normalized_all_terms = np.zeros((test_items.shape[0], self.data.nT))
        for i, k in enumerate(self.out):
            normalized_all_terms[:,k] = normalized[:,i]
        return normalized_all_terms
    
class som(classifier):
    # Self-Organizing Map
    # REFERENCE: H Ritter, T Kohonen (1989). Self-organizing semantic maps. 
    #  Biological cybernetics 61(4), 241-254
    # REFERENCE: T. Kohonen (2001): Self-Organizing Maps. Third Edition. 
    #  Springer

    def initialize_model(self, parameters):
        # parameters
        self.init_bandwidth = parameters['som initialization bandwidth']
        self.size = parameters['som size']
        self.alpha = parameters['som alpha']
        self.a = parameters['som a']
        self.c = parameters['som c']
        self.lambda_sigma = parameters['som lambda_sigma']
        self.sigma_0 = parameters['som sigma_0']
        self.n_pretrain = parameters['som n pretrain']
        self.neighborhood = parameters['som neighborhood']
        self.delta_sigma = parameters['som delta sigma']
        # YM: added parameters
        if parameters['moment of onset']:
            self.onset = [x*self.n_iterations for x in parameters['moment of onset']]
            self.share = parameters['language share']
        #
        # initialize MAP
        self.size_y = self.size_x = self.size
        term_map = np.zeros((self.size_x, self.size_y, sum(self.data.nT.values())))#[:,:,:]
        property_map = (((np.random.rand(self.size_x, self.size_y, 
                                         self.data.nF) - 0.5 ) *
                         self.init_bandwidth) + 0.5) #[:,:,:]
        self.map = np.concatenate((term_map, property_map), axis = 2)
        self.indices = A([A([A([i,j]) for j in range(self.size_y)]) 
                            for i in range(self.size_x)])
        # FUTURE: rectangular and growing maps
        self.sigma_0 = self.size / 3
        self.quantization_errors = []
        self.current_quantization_error = 0
        #
        # pretrain on just property features
        self.pretrain()
        #
        return

    def pretrain(self):
        # train the SOM on just property features
        for i in range(self.n_pretrain):
            x, _y = self.sample_input_item()
            self.time += 1
            input_item = self.get_input_item(features = x, term = None)
            self.update_map(input_item, self.time)
        self.time = 0
        return

    def fit(self):
        # train the SOM on input items
        for i in range(self.len_interval):
            x, y = self.sample_input_item()
            input_item = self.get_input_item(features = x, term = y)
            self.update_map(input_item, (self.n_pretrain + self.time - 
                                         self.len_interval + i ))
        f = 1
        if self.time % 500 == 0:
            if (self.quantization_errors != [] and 
                self.current_quantization_error < 
                  (np.mean(self.quantization_errors[-10:]) * f)):
                self.sigma_0 = np.max([self.sigma_0 - self.delta_sigma, 0.001])

            self.quantization_errors.append(self.current_quantization_error)
            self.current_quantization_error = 0

    def fitM(self):
        # train the SOM on input items
        for i in range(self.len_interval):
            self.current_exposure = [self.time > self.onset[i]
                               for i in range(len(self.target_language))]
            # YM: the languages available at this moment.
            if sum(self.current_exposure) == 1:
                self.lang = self.target_language[self.current_exposure.index(1)]
            elif sum(self.current_exposure) > 1:
                ps = A([float(x) for x in A(self.current_exposure) * A(self.share)])
                ps /= sum(ps)
                self.lang = np.random.choice(self.target_language, 1, p=ps)[0]
            # YM: choosing a language from the distribution given by 'share'.
            x, y = self.sample_input_itemM()
            input_item = self.get_input_itemM(features = x, term = y)
            self.update_mapM(input_item, (self.n_pretrain + self.time -
                                         self.len_interval + i ))
        f = 1
        if self.time % 500 == 0:
            if (self.quantization_errors != [] and
                self.current_quantization_error <
                  (np.mean(self.quantization_errors[-10:]) * f)):
                self.sigma_0 = np.max([self.sigma_0 - self.delta_sigma, 0.001])

            self.quantization_errors.append(self.current_quantization_error)
            self.current_quantization_error = 0

    def dump(self):
        # pickles the SOM
        fn = '%s/model_%d_%d.p' % (self.data.dirname,self.simulation,self.time)
        with open(fn,'wb') as fh:
            pickle.dump(self.map.astype('float16'), fh)

    def load(self, simulation, time):
        # loads a pickled SOM
        self.simulation = simulation
        self.time = time
        fn = '%s/model_%d_%d.p' % (self.data.dirname, simulation, time)
        with open(fn, 'rb') as fh:
            self.map = pickle.load(fh,encoding='latin1') 

    def get_input_item(self, features, term = None):
        # on the basis of a string of features and a term, returns one vector,
        # combining a one-hot distribution (with the hot bit set to a) and the
        # feature string
        term_str = self.data.terms[term] if term != None else ''
        return np.hstack(( self.a * (self.data.terms == term_str), features))

    def get_input_itemM(self, features, term = None):
        # on the basis of a string of features and a term, returns one vector,
        # combining a one-hot distribution (with the hot bit set to a) and the
        # feature string
        term_str = self.data.terms[self.lang][term] if term != None else ''
        all_terms = A([i for sublist in list(self.data.terms.values()) for i in sublist])
        return np.hstack(( self.a * (all_terms == term_str), features))

    def update_map(self, input_item, time):
        # updates the map with an input item
        # sigma = self.sigma_0 * np.exp(-(time / self.lambda_sigma))
        sigma = self.sigma_0
        # BB 2906 sigma decreases as a function of the quantization error 
        # improvement
        bmu_ix = self.get_bmu_ix(input_item)
        h = np.exp(-self.get_grid_distance(bmu_ix) / (2 * P(sigma, 2) ))
        # the formulation with (2*sigma^2) a.o.t. sigma^2 comes from Kohonen 
        # (2001), p. 111
        h = h[..., None] * np.ones((self.data.nT + self.data.nF))
        self.map = self.map + self.alpha * h * (-self.map + input_item)
        self.current_quantization_error += P(S(P(self.map[bmu_ix]-input_item,
                                                 2)),0.5)

    def update_mapM(self, input_item, time):
        # updates the map with an input item
        # sigma = self.sigma_0 * np.exp(-(time / self.lambda_sigma))
        sigma = self.sigma_0
        # BB 2906 sigma decreases as a function of the quantization error
        # improvement
        bmu_ix = self.get_bmu_ixM(input_item)
        h = np.exp(-self.get_grid_distance(bmu_ix) / (2 * P(sigma, 2) ))
        # the formulation with (2*sigma^2) a.o.t. sigma^2 comes from Kohonen
        # (2001), p. 111
        h = h[..., None] * np.ones((sum(self.data.nT.values()) + self.data.nF))
        self.map = self.map + self.alpha * h * (-self.map + input_item)
        self.current_quantization_error += P(S(P(self.map[bmu_ix]-input_item,
                                                 2)),0.5)

    def get_bmu_ix(self, input_item, ignore_terms = False):
        # gets the map index of the best matching unit
        f = self.data.nT * ignore_terms
        D = np.linalg.norm(self.map[:,:,f:] - input_item[f:],ord = 2,axis = 2)
        return np.unravel_index(D.argmin(), self.map.shape[:2])

    def get_bmu_ixM(self, input_item, ignore_terms = False):
        # gets the map index of the best matching unit
        f = sum(self.data.nT.values()) * ignore_terms
        D = np.linalg.norm(self.map[:,:,f:] - input_item[f:],ord = 2,axis = 2)
        return np.unravel_index(D.argmin(), self.map.shape[:2])

    def get_grid_distance(self, bmu_ix, neighborhood = 'euclidean'):
        # gets values for each cell of the map, given a center of activation 
        # at bmu_ix
        if neighborhood == 'euclidean': 
            return S(P(self.indices - bmu_ix, 2),2) 
        elif neighborhood == 'vonneuman':
            h = np.zeros(self.indices.shape[:2])
            h[bmu_ix] = 1
            h[bmu_ix[0]-1, bmu_ix[1]] = h[bmu_ix[0]+1, bmu_ix[1]] = 1
            h[bmu_ix[0], bmu_ix[1]-1] = h[bmu_ix[0], bmu_ix[1]+1] = 1 
            return h          

    def predict_terms(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test 
        # item i in I from a matrix I x F for I test items each with F 
        # features.
        if len(test_items.shape) == 1: test_items = A([test_items])
        bmu_ixx = []
        for test_item in test_items:
            input_item = self.get_input_itemM(features = test_item, term = None)
            bmu_ix = self.get_bmu_ix(input_item, True)
            bmu_ixx.append(bmu_ix)
        term_distributions = A([self.map[i,j,:self.data.nT] for i,j in bmu_ixx])
        return normalize(term_distributions, norm = 'l1', axis = 1)

    def predict_termsM(self, test_items):
        # returns a I x T matrix of posterior probabilities over T per test
        # item i in I from a matrix I x F for I test items each with F
        # features.
        #if len(test_items.shape) == 1: test_items = A([test_items])
        # YM: this function receives a dictionary now.
        term_distributions = dict()
        bmu_ixx = dict()
        for idx,test_item in test_items.items():
            input_item = self.get_input_itemM(features = test_item, term = None)
            bmu_ix = self.get_bmu_ixM(input_item, True)
            bmu_ixx[idx] = bmu_ix
        x = 0
        for l in self.target_language:
            term_distributions[l] = dict()
            for idx, (i,j) in bmu_ixx.items():
                term_distributions[l][idx] = A(self.map[i,j,x:x+self.data.nT[l]])
                term_distributions[l][idx] = normalize(A([term_distributions[l][idx]]), norm = 'l1', axis = 1)
            x += self.data.nT[l]
        return term_distributions

    def get_dominant_term(self, i, j):
        return self.data.terms[np.argmax(self.map[i][j][:self.data.nT])]

    def get_dominant_termM(self, i, j, l, start):
        return self.data.terms[l][np.argmax(self.map[i][j][start:start+self.data.nT[l]])]

    def get_term_mapM(self):
        # returns a matrix the size of the SOM with the most likely term in
        # every cell
        result = list()
        start = 0
        for l in self.target_language:
            result.append(A([A([('%s   ' % (self.get_dominant_termM(i,j, l, start)
                                    if S(self.map[i,j,start:start+self.data.nT[l]]) > 0
                                    else '[]'))[:3]
                         for j in range(self.map.shape[1])])
                        for i in range(self.map.shape[0])]))
            start += self.data.nT[l]
        return result

    def get_term_map(self):
        # returns a matrix the size of the SOM with the most likely term in 
        # every cell
        return A([A([('%s   ' % (self.get_dominant_term(i,j)
                                if S(self.map[i,j,:self.data.nT]) > 0 
                                else '[]'))[:3]
                     for j in range(self.map.shape[1])]) 
                    for i in range(self.map.shape[0])])

    def discriminate(self):
        # calculates the between-cell map distance for the BMUs of an array of 
        # stimuli used in Beekhuizen & Stevenson 2016.
        # only uses sinij/goluboj or blue
        t_set = { '111' : ['BU'], '112' : ['sinij', 'goluboj'] }
        lg = self.parameters['target language']
        t_ix = [i for i,t in enumerate(self.data.terms) if t in t_set[lg]]
        testset = A([ self.get_input_item(t) for t in 
                      self.data.discrimination_stimuli ])
        positions = A([ A(self.get_bmu_ix(t)) for t in testset ])
        distances = pairwise_distances(positions, metric = 'euclidean')
        dn, tn = self.data.dirname, self.data.discrimination_data
        pt = self.predict_terms(self.data.discrimination_stimuli)
        terms = [t_set[lg][t] for t in pt[:,t_ix].argmax(1)]
        #print(terms)
        d_fn = '%s/discrimination_terms_%s_bo.csv' % (dn, tn)
        dc_fn = '%s/discrimination_confusability_%s_bo.csv' % (dn, tn)
        with open(d_fn, 'a') as o:
            if os.path.getsize(d_fn) == 0: 
                o.write('simulation,time,stimulus,term\n')
            for i,t in enumerate(terms):
                o.write('%d,%d,%d,%s\n' % (self.simulation, self.time, i, t))
        with open(dc_fn, 'a') as o:
            if os.path.getsize(dc_fn) == 0: 
                o.write('simulation,time,stimulus.1,stimulus.2,')
                o.write('term.1,term.2,distance\n')
            for i in range(distances.shape[0]):
                for j in range(i+1, distances.shape[0]):
                    o.write('%d,%d,%d,%d,%s,%s,%.3f\n' %
                            (self.simulation, self.time, i, j, terms[i], 
                             terms[j], distances[i,j]))
        return

    def discriminate_activation_distance(self):
        testset = A([ self.get_input_item(t) for t in 
                      self.data.discrimination_stimuli ])
        f = self.data.nT
        activation = np.array([np.linalg.norm(self.map[:,:,f:] - t[f:], 
                                            ord = 2, axis = 2).reshape(1,-1)[0]
                             for t in testset])
        distances = pairwise_distances(activation, metric = 'euclidean')
        dn, tn = self.data.dirname, self.data.discrimination_data
        tz = self.predict_terms(self.data.discrimination_stimuli).argmax(1)
        terms = self.data.terms[tz]

        d_fn = '%s/AD_discrimination_terms_%s.csv' % (dn, tn)
        dc_fn = '%s/AD_discrimination_confusability_%s.csv' % (dn, tn)
        with open(d_fn, 'a') as o:
            if os.path.getsize(d_fn) == 0: 
                o.write('simulation,time,stimulus,term\n')
            for i,t in enumerate(terms):
                o.write('%d,%d,%d,%s\n' % (self.simulation, self.time, i, t))
        with open(dc_fn, 'a') as o:
            if os.path.getsize(dc_fn) == 0: 
                o.write('simulation,time,stimulus.1,stimulus.2,')
                o.write('term.1,term.2,distance\n')
            for i in range(distances.shape[0]):
                for j in range(i+1, distances.shape[0]):
                    o.write('%d,%d,%d,%d,%s,%s,%.3f\n' %
                            (self.simulation, self.time, i, j, terms[i], 
                             terms[j], distances[i,j]))
        return
