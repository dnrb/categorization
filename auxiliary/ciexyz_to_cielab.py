import sys, re
xn, yn, zn = (0.95047, 1.00000, 1.08333)
# Illuminant D65

def yxy_to_xyz(Y, x, y):
    X = Y * (x/y) #x * ( Y / y )
    Y = Y
    Z = (X/x) - X - Y #(1.0 - x - y ) * ( Y / y )
    return X,Y,Z
    
def trns(t):
    if t > 0.008856: return pow(t, 1/3.0)
    else: return (7.797*t) + (4/29.0)

def xyz_to_cielab(X,Y,Z):
    L = (116 * trns(Y/yn)) - 16
    a = 500 * (trns(X/xn) - trns(Y/yn))
    b = 200 * (trns(Y/yn) - trns(Z/zn))
    return L,a,b

f = open('davies_corbett_1994_adult_naming_data_russian.csv')
for line in f.readlines()[1:]:
    Y, x, y = [float(i) for i in re.split(',',line)[1:4]]
    X, Y, Z = yxy_to_xyz(Y,x,y)
    print Y, x, y, '|',
    sum_xyz = sum([X,Y,Z])
    x = X/sum_xyz
    y = Y/sum_xyz
    z = Z/sum_xyz
    print x,y,z, '|', xyz_to_cielab(X,Y,Z)
