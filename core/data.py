import re
import numpy as np
import csv
from collections import defaultdict as dd
#
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#
A = np.array

class data:

    def __init__(self, parameters):
        # set local parameters
        self.dirname = '%s/%s' % (parameters['folder name'], 
                                  parameters['subfolder name'])
        self.target_language = parameters['target language']
        self.leave_out_language = (None if 
                                   parameters['leave target language out'] 
                                   else self.target_language)
        self.distance_metric = parameters['distance metric']
        self.pca_threshold = parameters['pca threshold']
        self.data_folder = parameters['data']
        self.input_sampling_responses = parameters['input sampling responses']
        self.frequency_data = (parameters['frequency data'] if 'frequency data'
                               in parameters else 'frequencies')
        self.conceptual_data = parameters['conceptual data']
        #      
        fn = 'data/%s/elicited_features.csv' % parameters['data']
        with open(fn, 'r') as fh: elicited_features = list(csv.reader(fh))
        #
        self.initialize_data(elicited_features)
        self.dim_weights, self.situations = self.initialize_features()
        # initializes several variables; reads features from file.
        # set initial dimension weights (for ALCOVE)
        self.nF = self.situations.shape[1]
        # set number of features
        self.P_t = self.initialize_P_t()
        # set corpus probabilities of terms
        self.nT = self.P_t.shape[0]
        # set number of terms for target language
        self.P_s_given_t = normalize(self.CMs[self.target_language].T, 
                                     norm = 'l1', axis = 1)
        # set conditional probabilities of situations given terms for target 
        # language
        self.max_P_t_given_s = self.CMs[self.target_language].argmax(1)
        self.max_P_t_given_s[self.CMs[self.target_language].sum(1) == 0] = -1
        # set the most likely term given every situation; for unseen 
        # situations set to -1
        self.terms = A(sorted(self.term_indices[self.target_language].keys(),
                       key = lambda k : 
                                self.term_indices[self.target_language][k]))
        # creates list of terms for target language for quick access.
        # creates a csv file with the representations used by the model in it

    def initialize_data(self, elicited_features):
        # sets term_indices (per language, languages (set of languages, nS 
        # (number of situations) as wel as CMs (count matrices for every 
        # language; { language : nS x nT_language }
        #fn = 'data/%s/elicited_features.csv' % self.data_folder
        #with open(fn, 'r') as fh:
        #    self.elicited_features = list(csv.reader(fh))
        self.elicited_features = elicited_features
        self.term_indices = dd(lambda : {})
        self.languages = set()
        self.nS = 0
        CM_constructor = dd(lambda : dd(lambda : dd(float)))
        #
        for language, subject, situation, word in self.elicited_features:
            self.languages.add(language)
            self.nS = np.max([self.nS, int(situation)+1])
            try:
                word_ix = self.term_indices[language][word]
            except KeyError:
                lt = len(self.term_indices[language])
                word_ix = self.term_indices[language][word] = lt
            CM_constructor[language][int(situation)][word_ix] += 1.0
        self.CMs = { language : 
                        np.zeros((self.nS, len(self.term_indices[language]))) 
                     for language in self.languages}
        for language, v1 in CM_constructor.items():
            for situation, v2 in v1.items():
                for term, count in v2.items():
                    self.CMs[language][situation,term] = count
        return

    def initialize_features(self):
        fn = ('data/%s/feature_spaces/%s.csv' % 
              (self.data_folder, self.conceptual_data))
        with open(fn, 'r') as fh:
            features =A([A([float(c) for c in row]) for row in csv.reader(fh)])
            range_ = features.max(0) - features.min(0)
            dim_weights = range_ / range_.max()
            features_centered=(features-features.mean(0))/range_.max()+0.5
            # centers the values s.t. the mean per feature is 0.5, the 
            # feature with the highest original range now has a range of 1, and
            # the other features have proportional ranges according to their 
            # original ranges.
        print('situation shape', features_centered.shape, self.conceptual_data)
        return dim_weights, features_centered

    def initialize_P_t(self):
        # returns the probabilities of the terms as read off from a frequencies
        # .csv file
        count = np.ones(len(self.term_indices[self.target_language]))
        if self.input_sampling_responses == 'corpus':
            fn = ('data/%s/frequencies/%s.csv' % 
                    (self.data_folder, self.frequency_data))
            with open(fn,'r') as fh:
                freqs = [f for f in csv.reader(fh) 
                         if f[0] == self.target_language]
            for language, word, freq in freqs:
                try:
                    word_ix = self.term_indices[language][word]
                    count[word_ix] = float(freq)+1
                    # added smoothing here to deal with low counts (e.g. for 
                    # Russian color)
                except KeyError: pass
        return normalize([count], norm = 'l1')[0]

    def read_discrimination_data(self, parameters):
        # creates an array of 20 stimuli for discrimination experiments.
        self.discrimination_data = parameters['discrimination data']
        if self.discrimination_data == 'winawer':
            start, end = 278, 187
            # takes situations whose Lab values are closest to 20 Winawer 
            # stimuli and that have require maxPt|s to be sin/gol; end = 195 
            # if this last constraint is not in place
        start_v, end_v = self.situations[start], self.situations[end]
        self.discrimination_stimuli = A([start_v - i*(start_v-end_v)/19 
                                         for i in range(20)])
        return
