"""
Written By 
Chaitya Shah
"""
#Linear Regression 
import numpy as np

class LinearRegression:
    """
        alpha : learning rate
        max_iter: maximum iteration for regression
    """
    def __init__(self,alpha=0.04,max_iter=10000):
        self.theta = np.random.random((1,1)) 
        self.X = np.random.random((1,1))
        self.Y =np.random.random((1,1))
        self.m = self.Y.size
        self.alpha = alpha
        self.max_iter = max_iter

    
    def predict(self,X):    
        return np.dot(self.theta,X.T)

    

    def costFunction(self,Yp,Yr):
        if(Yp.size != Yr.size):
            raise "Values must have same dimentions"
        return 0.5*np.sum((Yp-Yr)**2)/(Yr.size)
    
    
    def _dcostFunction(self): #derivative of Costfunction

        error =  (self.predict(self.X)-self.Y)  
        correction = (np.dot(self.X.T,error)/self.m)
        return correction
     


    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        self.m = self.Y.size
        
        #initailizing random values of theta
        self.theta = np.random.random(self.X.shape[1])
        for i in range(self.max_iter):
            self.theta = self.theta - self.alpha * self._dcostFunction()
        return self.theta

         

if __name__ == "__main__":

       linear = LinearRegression(alpha=0.4,max_iter=100)
       X = np.random.random((1000,5))
       Y = 2*X[:,0]+X[:,1]+2.012*X[:,2]+0.31*X[:,3]+ 9*X[:,4]
       Y.reshape((Y.size,))
       theta = linear.fit(X,Y)
       print(theta) #expected values [2, 1, 2.012, 0.31, 9]

