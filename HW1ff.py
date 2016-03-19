from numpy import array, zeros
import numpy as np
from pylab import *
import time
import math
One = """
..XXX...
.XXXX...
XXXXX...
...XX...
...XX...
...XX...
...XX...
...XX...
"""

Five = """
XXXXXXXX
XX......
XX......
XX......
XXXXXXXX
......XX
......XX
XXXXXXXX
"""

Three = """
XXXXXXXX
......XX
......XX
...XXXXX
......XX
......XX
......XX
XXXXXXXX
"""

Four = """
.....XX.
....XX..
...XX...
.XX.....
XX......
XX...XX.
XXXXXXXX
.....XX.
"""

def samplearray(letter):
    return array([+1 if c=='X' else -1 for c in letter.replace('\n','')])


sample1 = samplearray(One)
sample2 = samplearray(Five)
sample3 = samplearray(Three)
sample4 = samplearray(Four)
T = zeros((64,64))

for i in range(64):
    for j in range(64):
        if i==j:
            T[i,j] = 0
        else:
            T[i,j] = sample1[i]*sample1[j]+sample2[i]*sample2[j]+sample3[i]*sample3[j]+sample4[i]*sample4[j]    

def engine(T, newpat):
    while 1:
        x = newpat
        from matplotlib import pyplot as plt
        plt.ion()
        for j in range(64):
            m = 0
            for i in range(64):
                m = m + T[i,j]*x[i]
            if m>=0:
                newpat[j] = 1
            else:
                newpat[j] = -1
            
            plt.imshow(newpat.reshape((8,8)),cmap=cm.binary, interpolation='nearest')
            pause(0.1)
        if(x.all()==newpat.all()):
            break
            
def noiser(numb,var):
    dev = math.sqrt(var)
    patnumb = samplearray(numb)
    for i in range(64):
        if patnumb[i] == 1:
            noise = np.random.normal(0,dev,1)
            patnumb[i] = patnumb[i] + noise[0]
            if patnumb[i] >= 0:
                patnumb[i] = 1
            else:
                patnumb[i] = -1
    return patnumb
    
engine(T,noiser(Four,100))