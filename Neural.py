import numpy as np

x = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
y = np.array(([92], [86], [89]), dtype=float) # output

# scale units
x = x/np.amax(x, axis=0) 
y = y/100 

# split data
traindata = np.split(x, [3])[0] # training data
testdata = np.split(x, [3])[1] # testing data

class Neural_class(object):
  def __init__(self):
  #parameters
    self.inputSize = 3#2
    self.outputSize = 1#1
    self.hiddenSize = 2#3

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def forward(self, traindata):
   
    self.z = np.dot(traindata, self.W1)
    self.z2 = self.sigmoid(self.z) 
    self.z3 = np.dot(self.z2, self.W2) 
    r = self.sigmoid(self.z3)
    return r

  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)

  def backward(self, traindata, y, r):

    self.r_error = y - r # error in output
    self.r_delta = self.r_error*self.sigmoidPrime(r)

    self.z2_error = self.r_delta.dot(self.W2.T) 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 

    self.W1 += traindata.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.r_delta) 

  def train(self, traindata, y):
    r = self.forward(traindata)
    self.backward(traindata, y, r)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

