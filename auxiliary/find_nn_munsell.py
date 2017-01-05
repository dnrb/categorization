import sys, csv, re

def distance(X, Y):
    return pow(sum(pow(abs(x-y),2) for x,y in zip(X,Y)),0.5)

t1 = [ [re.split(',', l)[0]] + [float(i) for i in re.split(',', l)[1:]]
       for l in open('cie_munsell.txt').readlines()[1:] ]
c = [float(i) for i in sys.argv[1:4]]

min_d = None
min_munsell = None

for t in t1:
    d = distance(c, [t[5], t[3], t[4]])
    if min_d == None or d < min_d:
        min_d = d
        min_munsell = t[0:3]
        
print c, min_munsell, min_d
