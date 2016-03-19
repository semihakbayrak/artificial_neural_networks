# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

#Nonlinear function sigmoid and its derivative
def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))

def drvsigmoid(x):
    x = np.array(x)
    return sigmoid(x)*(1-sigmoid(x))

    
#MLP class which includes several functions
class MLP:
    #structurel function of MLP which is used to to create new MLP object
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        #weight array starts to be created by for loops
        for i in range(len(self.layers)-1):
            self.weights.append([])
            for j in range(self.layers[i]):
                self.weights[i].append([])
                for k in range(self.layers[i+1]):
                    r = (2*np.random.random()-1)
                    self.weights[i][j].append(r) #initial weights specified
                            
    
    #starts from input layer and propagate to output layer
    def propagate_forward(self,X):
        self.outputs = []
        self.inputs = []
        self.inputs.append(X[0])
        self.outputs.append(self.inputs[0]) #output of first layer, simply equal to input
        for i in range(len(self.layers)-1):
            self.inputs.append([])
            for j in range(self.layers[i+1]):
                add = 0
                for k in range(self.layers[i]):
                    add = add + self.outputs[i][k]*self.weights[i][k][j]
                self.inputs[i+1].append(add) #net input to a neuron
            if i == len(self.layers)-1:
                self.outputs.append(self.inputs[i+1]) #output of last layer, which is a linear function
            else:
                self.outputs.append(sigmoid(self.inputs[i+1])) #output of hidden layers acquired by using nonlinear sigmoid
        
        
    #This is where learning starts and weights update themselves
    def propagate_backward(self,X,nuu=0.4):
        self.expected = np.array(X[1])
        self.out = np.array(self.outputs[len(self.layers)-1])
        drv = np.array(drvsigmoid(self.inputs[len(self.layers)-1]))
        self.error = (self.expected - self.out)
        sigma = self.error*drv #sigma belongs to last layer
        self.sigmas = []
        self.sigmas.append(sigma)
        #loop to create sigma array
        for i in range(len(self.layers)-1):
            self.sigmas.append([])
            p = len(self.layers)-(i+2)
            for j in range(self.layers[p]):
                add = 0
                for k in range(self.layers[len(self.layers)-(i+1)]):
                    add = add + self.sigmas[i][k]*self.weights[p][j][k]
                add = add*drvsigmoid(self.inputs[p][j])
                self.sigmas[i+1].append(add)
        self.deltaw = []
        #loop to create deltaw array, change in weights
        for i in range(len(self.layers)-1):
            j = len(self.layers)-(i+2)
            s = np.array(self.sigmas[i])
            sm = np.mat(s)
            o = np.array(self.outputs[j])
            om = np.mat(o)
            wm = om.T*sm
            wm = np.array(wm)
            wm = nuu*wm
            self.deltaw.append(wm)
        for i in range(len(self.layers)-1):
            p = len(self.layers)-(i+2)
            for j in range(self.layers[p]):
                self.weights[p][j] = self.weights[p][j] + self.deltaw[i][j] 
        
        
    #function for training
    def train(self,X,epochs=10000,nuu=0.4,muu=0.2,error=0.01):
        self.errorlist = []
        for i in range(epochs):
            print i
            self.errorfunc = 0
            for j in range(len(X)):
                self.propagate_forward(X[j])
                self.propagate_backward(X[j])
                for k in range(len(self.layers)-1):
                    for l in range(self.layers[len(self.layers)-(k+2)]):
                        self.deltaw[k][l] = self.deltaw[k][l] + muu*self.deltaw[k][l] #momentum term
                self.errorfunc = (self.errorfunc + (np.array(self.error))**2)
            self.errorlist.append(self.errorfunc)
            print self.errorlist[i]
            count = 0
            zerocount = 0
            if i>0:
                for j in range(self.layers[len(self.layers)-1]):
                    if (self.errorlist[i]-self.errorlist[i-1])[j]<0:
                        count = count + 1
                    elif (self.errorlist[i]-self.errorlist[i-1])[j]==0:
                        zerocount = zerocount + 1
                #Adaptive parameters
                if count==self.layers[len(self.layers)-1]:
                    nuu = nuu + 0.005
                else:
                    nuu = nuu - 0.01*nuu
                if zerocount==self.layers[len(self.layers)-1]:
                    break
            errorcount = 0
            for j in range(self.layers[len(self.layers)-1]):
                    if self.errorfunc[j]<error:
                        errorcount = errorcount + 1
            if errorcount == self.layers[len(self.layers)-1]:
                plt.figure()
                plt.plot(self.errorlist)
                plt.show()
                break
                
            
    #function for testing, very similar to forward propagating        
    def test(self,X):
        self.outputs = []
        self.inputs = []
        self.inputs.append(X[0])
        self.outputs.append(self.inputs[0])
        for i in range(len(self.layers)-1):
            self.inputs.append([])
            for j in range(self.layers[i+1]):
                add = 0
                for k in range(self.layers[i]):
                    add = add + self.outputs[i][k]*self.weights[i][k][j]
                self.inputs[i+1].append(add)
            if i == len(self.layers)-1:
                self.outputs.append(self.inputs[i+1])
            else:
                self.outputs.append(sigmoid(self.inputs[i+1]))
        print self.inputs[0]
        print self.outputs[len(self.layers)-1]
                
        
####################

#mlp = MLP([2,8,9,1])
#X = [[[0,0],[0]],
#[[0,1],[1]],
#[[1,0],[1]],
#[[1,1],[0]]]
#mlp.train(X)
#print 'Training is over'
#mlp.test([[0,0]])
#mlp.test([[0,1]])
#mlp.test([[1,0]])
#mlp.test([[1,1]])


#mlp2 = MLP([1,10,1])
#X = []
#my_list = list(xrange(1,101))
#random.shuffle(my_list)
#for i in range (85):
#    X.append([])
#    for j in range(2):
#        X[i].append([])
#        if j ==0:
#            X[i][j].append(my_list[i])
#        else:
#            X[i][j].append(1.0/my_list[i])
#mlp2.train(X)
#A = []
#for i in range (100):
#    A.append([])
#    for j in range(2):
#        A[i].append([])
#        if j ==0:
#            A[i][j].append(my_list[i])
#        else:
#            A[i][j].append(1.0/my_list[i])
#print 'Training is over'
#for i in range (15):
#    j = i+85
#    mlp2.test(A[j])
    

mlp3 = MLP([4,9,10,3])
X = [[[5.1,3.5,1.4,0.2],[1,0,0]],
[[4.9,3.0,1.4,0.2],[1,0,0]],
[[4.7,3.2,1.3,0.2],[1,0,0]],
[[4.6,3.1,1.5,0.2],[1,0,0]],
[[5.0,3.6,1.4,0.2],[1,0,0]],
[[5.4,3.9,1.7,0.4],[1,0,0]],
[[4.6,3.4,1.4,0.3],[1,0,0]],
[[5.0,3.4,1.5,0.2],[1,0,0]],
[[4.4,2.9,1.4,0.2],[1,0,0]],
[[4.9,3.1,1.5,0.1],[1,0,0]],
[[5.4,3.7,1.5,0.2],[1,0,0]],
[[4.8,3.4,1.6,0.2],[1,0,0]],
[[4.8,3.0,1.4,0.1],[1,0,0]],
[[4.3,3.0,1.1,0.1],[1,0,0]],
[[5.8,4.0,1.2,0.2],[1,0,0]],
[[5.7,4.4,1.5,0.4],[1,0,0]],
[[5.4,3.9,1.3,0.4],[1,0,0]],
[[5.1,3.5,1.4,0.3],[1,0,0]],
[[5.7,3.8,1.7,0.3],[1,0,0]],
[[5.1,3.8,1.5,0.3],[1,0,0]],
[[5.4,3.4,1.7,0.2],[1,0,0]],
[[5.1,3.7,1.5,0.4],[1,0,0]],
[[4.6,3.6,1.0,0.2],[1,0,0]],
[[5.1,3.3,1.7,0.5],[1,0,0]],
[[4.8,3.4,1.9,0.2],[1,0,0]],
[[5.0,3.0,1.6,0.2],[1,0,0]],
[[5.0,3.4,1.6,0.4],[1,0,0]],
[[5.2,3.5,1.5,0.2],[1,0,0]],
[[5.2,3.4,1.4,0.2],[1,0,0]],
[[4.7,3.2,1.6,0.2],[1,0,0]],
[[4.8,3.1,1.6,0.2],[1,0,0]],
[[5.4,3.4,1.5,0.4],[1,0,0]],
[[5.2,4.1,1.5,0.1],[1,0,0]],
[[5.5,4.2,1.4,0.2],[1,0,0]],
[[4.9,3.1,1.5,0.1],[1,0,0]],
[[5.0,3.2,1.2,0.2],[1,0,0]],
[[5.5,3.5,1.3,0.2],[1,0,0]],
[[4.9,3.1,1.5,0.1],[1,0,0]],
[[4.4,3.0,1.3,0.2],[1,0,0]],
[[5.1,3.4,1.5,0.2],[1,0,0]],
[[5.0,3.5,1.3,0.3],[1,0,0]],
[[4.5,2.3,1.3,0.3],[1,0,0]],
[[4.4,3.2,1.3,0.2],[1,0,0]],
[[5.0,3.5,1.6,0.6],[1,0,0]],
[[5.1,3.8,1.9,0.4],[1,0,0]],
[[4.8,3.0,1.4,0.3],[1,0,0]],
[[5.1,3.8,1.6,0.2],[1,0,0]],
[[4.6,3.2,1.4,0.2],[1,0,0]],
[[5.3,3.7,1.5,0.2],[1,0,0]],
[[5.0,3.3,1.4,0.2],[1,0,0]],
[[7.0,3.2,4.7,1.4],[0,1,0]],
[[6.4,3.2,4.5,1.5],[0,1,0]],
[[6.9,3.1,4.9,1.5],[0,1,0]],
[[5.5,2.3,4.0,1.3],[0,1,0]],
[[6.5,2.8,4.6,1.5],[0,1,0]],
[[5.7,2.8,4.5,1.3],[0,1,0]],
[[6.3,3.3,4.7,1.6],[0,1,0]],
[[4.9,2.4,3.3,1.0],[0,1,0]],
[[6.6,2.9,4.6,1.3],[0,1,0]],
[[5.2,2.7,3.9,1.4],[0,1,0]],
[[5.0,2.0,3.5,1.0],[0,1,0]],
[[5.9,3.0,4.2,1.5],[0,1,0]],
[[6.0,2.2,4.0,1.0],[0,1,0]],
[[6.1,2.9,4.7,1.4],[0,1,0]],
[[5.6,2.9,3.6,1.3],[0,1,0]],
[[6.7,3.1,4.4,1.4],[0,1,0]],
[[5.6,3.0,4.5,1.5],[0,1,0]],
[[5.8,2.7,4.1,1.0],[0,1,0]],
[[6.2,2.2,4.5,1.5],[0,1,0]],
[[5.6,2.5,3.9,1.1],[0,1,0]],
[[5.9,3.2,4.8,1.8],[0,1,0]],
[[6.1,2.8,4.0,1.3],[0,1,0]],
[[6.3,2.5,4.9,1.5],[0,1,0]],
[[6.1,2.8,4.7,1.2],[0,1,0]],
[[6.4,2.9,4.3,1.3],[0,1,0]],
[[6.6,3.0,4.4,1.4],[0,1,0]],
[[6.8,2.8,4.8,1.4],[0,1,0]],
[[6.7,3.0,5.0,1.7],[0,1,0]],
[[6.0,2.9,4.5,1.5],[0,1,0]],
[[5.7,2.6,3.5,1.0],[0,1,0]],
[[5.5,2.4,3.8,1.1],[0,1,0]],
[[5.5,2.4,3.7,1.0],[0,1,0]],
[[5.8,2.7,3.9,1.2],[0,1,0]],
[[6.0,2.7,5.1,1.6],[0,1,0]],
[[5.4,3.0,4.5,1.5],[0,1,0]],
[[6.0,3.4,4.5,1.6],[0,1,0]],
[[6.7,3.1,4.7,1.5],[0,1,0]],
[[6.3,2.3,4.4,1.3],[0,1,0]],
[[5.6,3.0,4.1,1.3],[0,1,0]],
[[5.5,2.5,4.0,1.3],[0,1,0]],
[[5.5,2.6,4.4,1.2],[0,1,0]],
[[6.1,3.0,4.6,1.4],[0,1,0]],
[[5.8,2.6,4.0,1.2],[0,1,0]],
[[5.0,2.3,3.3,1.0],[0,1,0]],
[[5.6,2.7,4.2,1.3],[0,1,0]],
[[5.7,3.0,4.2,1.2],[0,1,0]],
[[5.7,2.9,4.2,1.3],[0,1,0]],
[[6.2,2.9,4.3,1.3],[0,1,0]],
[[5.1,2.5,3.0,1.1],[0,1,0]],
[[5.7,2.8,4.1,1.3],[0,1,0]],
[[6.3,3.3,6.0,2.5],[0,0,1]],
[[5.8,2.7,5.1,1.9],[0,0,1]],
[[7.1,3.0,5.9,2.1],[0,0,1]],
[[6.3,2.9,5.6,1.8],[0,0,1]],
[[6.5,3.0,5.8,2.2],[0,0,1]],
[[7.6,3.0,6.6,2.1],[0,0,1]],
[[4.9,2.5,4.5,1.7],[0,0,1]],
[[7.3,2.9,6.3,1.8],[0,0,1]],
[[6.7,2.5,5.8,1.8],[0,0,1]],
[[7.2,3.6,6.1,2.5],[0,0,1]],
[[6.5,3.2,5.1,2.0],[0,0,1]],
[[6.4,2.7,5.3,1.9],[0,0,1]],
[[6.8,3.0,5.5,2.1],[0,0,1]],
[[5.7,2.5,5.0,2.0],[0,0,1]],
[[5.8,2.8,5.1,2.4],[0,0,1]],
[[6.4,3.2,5.3,2.3],[0,0,1]],
[[6.5,3.0,5.5,1.8],[0,0,1]],
[[7.7,3.8,6.7,2.2],[0,0,1]],
[[7.7,2.6,6.9,2.3],[0,0,1]],
[[6.0,2.2,5.0,1.5],[0,0,1]],
[[6.9,3.2,5.7,2.3],[0,0,1]],
[[5.6,2.8,4.9,2.0],[0,0,1]],
[[7.7,2.8,6.7,2.0],[0,0,1]],
[[6.3,2.7,4.9,1.8],[0,0,1]],
[[6.7,3.3,5.7,2.1],[0,0,1]],
[[7.2,3.2,6.0,1.8],[0,0,1]],
[[6.2,2.8,4.8,1.8],[0,0,1]],
[[6.1,3.0,4.9,1.8],[0,0,1]],
[[6.4,2.8,5.6,2.1],[0,0,1]],
[[7.2,3.0,5.8,1.6],[0,0,1]],
[[7.4,2.8,6.1,1.9],[0,0,1]],
[[7.9,3.8,6.4,2.0],[0,0,1]],
[[6.4,2.8,5.6,2.2],[0,0,1]],
[[6.3,2.8,5.1,1.5],[0,0,1]],
[[6.1,2.6,5.6,1.4],[0,0,1]],
[[7.7,3.0,6.1,2.3],[0,0,1]],
[[6.3,3.4,5.6,2.4],[0,0,1]],
[[6.4,3.1,5.5,1.8],[0,0,1]],
[[6.0,3.0,4.8,1.8],[0,0,1]],
[[6.9,3.1,5.4,2.1],[0,0,1]],
[[6.7,3.1,5.6,2.4],[0,0,1]],
[[6.9,3.1,5.1,2.3],[0,0,1]],
[[5.8,2.7,5.1,1.9],[0,0,1]],
[[6.8,3.2,5.9,2.3],[0,0,1]],
[[6.7,3.3,5.7,2.5],[0,0,1]],
[[6.7,3.0,5.2,2.3],[0,0,1]],
[[6.3,2.5,5.0,1.9],[0,0,1]],
[[6.5,3.0,5.2,2.0],[0,0,1]],
[[6.2,3.4,5.4,2.3],[0,0,1]],
[[5.9,3.0,5.1,1.8],[0,0,1]]]

my_list = list(xrange(0,150))
random.shuffle(my_list)
TS = []
for i in range(142):
    TS.append(X[my_list[i]])

mlp3.train(TS,10000,0.4,0.2,2)
print 'Training is over'
A = []
for i in range(8):
    A .append(X[my_list[i+142]])
    mlp3.test(A[i])