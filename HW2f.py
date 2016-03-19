import numpy as np
from numpy import array, zeros
from pylab import rand, plot, show,figure
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt

weights = rand(3)*2 - 1
coef = 0.6

def samplemachine():
    r = rand(1)
    if r[0]>=0.5:
        x1 = rand(1)
        x2 = rand(1)
        x3 = rand(1)
        d = 1
    else:
        x1 = -rand(1)
        x2 = -rand(1)
        x3 = -rand(1)
        d = -1
    s = []
    s.append(x1[0])
    s.append(x2[0])
    s.append(x3[0])
    s.append(d)
    return s

def output(s,w):
    u = s[0]*w[0] + s[1]*w[1] + s[2]*w[2]
    if u>=0:
        y = 1
    else:
        y = -1
    return y
    
def newweight(wold,coef,error,x):
    wnew = wold + coef*error*x
    return wnew
       
    
def train():
    sx1 = []
    sx2 = []
    sx3 = []
    sx4 = []
    sx5 = []
    for i in range (35):
        s = samplemachine()
        sx1.append(s[0])
        sx2.append(s[1])
        sx3.append(s[2])
        y = output(s,weights)
        if s[3] == 1:
            sx4.append("red")
        else:
            sx4.append("blue")
        if y == 1:
            sx5.append("red")
        else:
            sx5.append("blue")
        if y != s[3]:
            e = s[3] - y
            weights[0] = newweight(weights[0],coef,e,s[0])
            weights[1] = newweight(weights[1],coef,e,s[1])
            weights[2] = newweight(weights[2],coef,e,s[2])
        else:
            e = 0
    fig1 = figure()
    fig2 = figure()
    ax = Axes3D(fig1)        
    ax.scatter(sx1,sx2,sx3,c=sx4,marker='o')
    ax2 = Axes3D(fig2)        
    ax2.scatter(sx1,sx2,sx3,c=sx5,marker='x')
    show()
    
    point  = np.array([0, 0, 0])
    normal = weights
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(-1,2), range(-1,2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    plt.show()

def test(n):
    tx1 = []
    tx2 = []
    tx3 = []
    tx4 = []
    tx5 = []
    for i in range (n):
        s = samplemachine()
        print s
        tx1.append(s[0])
        tx2.append(s[1])
        tx3.append(s[2])
        if s[3] == 1:
            tx4.append("red")
        else:
            tx4.append("blue")
        y = output(s,weights)
        if y == 1:
            tx5.append("red")
        else:
            tx5.append("blue")
        print y
        #if y != s[3]:
        #    e = s[3] - y
        #    weights[0] = newweight(weights[0],coef,e,s[0])
        #    weights[1] = newweight(weights[1],coef,e,s[1])
        #    weights[2] = newweight(weights[2],coef,e,s[2])
        #else:
        #    e = 0  
    fig3 = figure()
    fig4 = figure()
    ax = Axes3D(fig3)        
    ax.scatter(tx1,tx2,tx3,c=tx4,marker='o')
    ax2 = Axes3D(fig4)        
    ax2.scatter(tx1,tx2,tx3,c=tx5,marker='x')
    show()      

    
def execute(n):
    train()
    test(n)
    
    
execute(5)