import re
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from collections import defaultdict as dd
import numpy as np
N_CHIPS = 330

def cielab_to_xyz(L, a, b):
    print L,a,b
    P = (L + 16) / 116
    Xn, Yn, Zn = 0.9642, 1.0000, 0.8249
    X = Xn * pow( P + a / 500, 3)
    Y = Yn * pow( P, 3 )
    Z = Zn * pow( P - b / 200, 3)
    return L, a, b #X, Y, Z

    
def read_chip_mundell_values():
    values = [None for i in range(N_CHIPS)]
    with open('cnum-vhcm-lab-new.txt') as fh:
        lines = [re.split('\t', line.strip('\n')) for line in fh.readlines()]
        for l in lines[1:]:
            values[int(l[0])-1] = "%s %s/24" % (l[4], int(float(l[5])))
            L = float(l[6])
            a = float(l[7])
            b = float(l[8])
            xyz = ','.join(['%.2f' % c for c in cielab_to_xyz(L, a, b)])
            values[int(l[0])-1] = xyz
            print values[int(l[0])-1]
    return values

def get_neighbors():
    neighbor_ixs = [None for i in range(N_CHIPS)]
    with open('chip.txt') as fh:
        lines = [re.split('\t', l.strip('\n')) for l in fh.readlines()]
        for i in range(N_CHIPS):
            line = lines[i]
            if line[2] == '0': continue
            row = chr(ord(line[1][0])-1), line[1], chr(ord(line[1][0])+1)
            col = ((int(line[2])-1 if int(line[2]) > 1 else 40),
                   int(line[2]),
                   (int(line[2])+1 if int(line[2]) < 40 else 1))
            neighbors = ['%s%d' % (row[ix], col[jx]) for ix,jx in
                         [(0,1), (2,1), (1,0), (1,2)]]
            neighbor_ixs[i] = ((ord(line[1][0]), int(line[2])),
                               tuple([j for j in range(N_CHIPS)
                                      if lines[j][3] in neighbors]))
            print i, line, neighbors, neighbor_ixs[i]
    return neighbor_ixs
            
        

def read_counts_per_language():
    language_x, language_y = dd(lambda: []), dd(lambda: [])
    terms_per_language = dd(lambda: {})
    count_matrix = {}
    fh = open('term.txt')
    for line in fh.readlines():
        language, speaker, chip, term = re.split('\t', line.strip('\n'))
        #if language != '1' or speaker != '1': continue
        if term not in terms_per_language[language]:
            terms_per_language[language][term] = len(terms_per_language[language])
        language_x[language].append(int(chip)-1)
        language_y[language].append(terms_per_language[language][term])
    for language in language_x:
        count_matrix[language] = csr_matrix((np.array([1 for i in language_x[language]]),
                                            (np.array(language_x[language]),
                                             np.array(language_y[language])))).todense()# sparse matrix
        #print language, count_matrix[language]
    return terms_per_language, count_matrix

def get_dm(count_matrix):
    dm = np.zeros(shape=(N_CHIPS,N_CHIPS))
    for language in count_matrix:
        #print language
        dm_l =  pairwise_distances(count_matrix[language], metric = 'euclidean')
        dm += dm_l
    return dm

def get_pca_coordinates(dm):
    colors = read_chip_mundell_values()
    out = open('pca_coordinates.txt', 'wb')
    model = PCA()
    coordinates = model.fit_transform(dm)
    for color, coordinate in zip(colors, coordinates):
        out.write('%s,%s\n' % (color, ','.join(['%.3f' % c for c in coordinate])))
    out.close()

def get_boundaries(neighbors, dm):
    out = open('boundaries.txt', 'wb')
    for i,n in enumerate(neighbors):
        if n == None: continue
        distances = [dm[i][j] for j in n[1]]
        mean = np.mean(distances)
        print i,chr(n[0][0]), n[0][1], n[1], distances
        out.write('%d,%d,%.1f\n' % (n[0][0], n[0][1], mean))
    out.close()
        
neighbors = get_neighbors()
terms, count_matrix = read_counts_per_language()
dm = get_dm(count_matrix)
print dm
print count_matrix
get_boundaries(neighbors, dm)
#get_pca_coordinates(dm)
