from Neural import  Neural_class
import numpy as np

#x = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
#y = np.array(([92], [86], [89]), dtype=float) # output

x=np.array(([1,0,1],[0,0,1]), dtype=float)
y=np.array([[1,1]]).T

# scale units
#x = x/np.amax(x, axis=0) 
#y = y/100 

# split data
#traindata = np.split(x, [3])[0] # training data
#testdata = np.split(x, [3])[1] # testing dat

traindata=x
testdata=np.array(([1,1,1]), dtype=float)

NN = Neural_class()

for i in range(1000): 
    print("# " + str(i) + "\n")
    print("Input (scaled): \n" + str(traindata))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.forward(traindata)))
    #print("Loss: \n" + str(np.mean(np.square(y - NN.forward(traindata)))))
    print("\n")
    NN.train(traindata, y)

NN.saveWeights()

print("Predicted data based on trained weights: ")
print("Input (scaled): \n" + str(testdata))
print("Output: \n" + str(NN.forward(testdata)))


