from sklearn import preprocessing
import numpy as np


# parameters

n = 10000
no_of_iterations = 10
alpha = 0.001

# prepare data 
# random data

x = np.linspace(1, 10, n)
np.random.shuffle(x)
y = 1120.021*x + 122

#x = preprocessing.scale(x)



def f(x, a, b):
    return a*x + b


def cost(x, a, b, y):
    return np.sum(np.square((f(x,a,b)-y)))

def SGD(x, y):
    
    a = np.random.random(1)
    b = np.random.random(1)


    noi =  0

    while noi != no_of_iterations:

        for i,j in enumerate(x):
            tempa = a - alpha * np.sum(((f(j, a, b) - y[i])*j))
            tempb = b - alpha * np.sum(((f(j, a, b)-y[i])))

            a, b = tempa, tempb

            #print("cost = {}".format(cost(x,a,b,y)))
            #print("a = {}, b = {}".format(a, b))


        print("cost after {} iteration {}".format(noi+1,cost(x, a, b, y)))
        print("a = {}, b = {}".format(a, b))

        noi+=1

SGD(x, y)
