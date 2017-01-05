import csv
from collections import defaultdict as dd
import numpy as np


def get_dist(e1, e2):
    return pow(pow(float(e1['L*']) - float(e2['L*']), 2) +
               pow(float(e1['a*']) - float(e2['a*']), 2) +
               pow(float(e1['b*']) - float(e2['b*']), 2), 0.5)

def get_nn(davies_item, munsell):
    min_dist, min_munsell = None, None
    for m in munsell:
        dist = get_dist(davies_item, m)
        if min_dist == None or dist < min_dist:
            min_dist = dist
            min_munsell = m
    return min_munsell['#cnum']

davies = list(csv.DictReader(open('davies_stimuli.csv')))
munsell = list(csv.DictReader(open('cnum-vhcm-lab-new.csv'),
                              delimiter = '\t'))
davies_elic = list(csv.reader(open('russian_adult_elicitation_raw.csv')))
primary = set(['chernyj', 'belyj', 'seryj', 'krasnyj',
               'zheltyj', 'sinij', 'goluboj', 'fioletovyj',
               'zelenyj', 'korichnevyj', 'oranzhevyj', 'rozovyj'])


terms = davies_elic[0][1:-1]
for line in davies_elic[1:]:
    color = next(d for d in davies if d['daviescode'] == line[0])
    munsell_nn = get_nn(color, munsell)
    for term, frequency in zip(terms, line[1:-1]):
        #if term not in primary: continue
        for f in range(int(frequency)):
            print '%d,%d,%d,\"%s\"' % (111, 1, int(munsell_nn), term)
