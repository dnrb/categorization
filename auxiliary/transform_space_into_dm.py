import sys
import numpy as np
import re
from sklearn.metrics.pairwise import pairwise_distances as pd

np.set_printoptions(precision = 2, linewidth= 300)
values = []
ffn = sys.argv[1]
fn = re.split('/', ffn)[-1]
values = np.array([np.array([float(f) for f in re.split(',', line.strip('\n'))]) for line in open(ffn)])
dm = pd(values)
dm /= dm.max()
dm -= (dm.max(0) - dm.min(0)) / 2
dm += 0.5
np.savetxt('dm_%s' % fn, dm, delimiter = ',')
    
