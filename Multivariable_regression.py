import numpy as np
import matplotlib.pyplot as plt
features = 5
def Hypothesis(theta,x):
	return np.dot(x,theta)

def CostFunction(theta,x,y):
	m = y.size
	j = np.sum(Hypothesis(theta,x) - y)/(2*m)
	return j

def GradientDescent(theta,x,y,iterations=1000,alpha=0.05):
	count = 0
	while(count!=iterations):	
		#print(CostFunction(theta,x,y))
		a = Hypothesis(theta,x)-y
		temp = np.dot(x.T,a)/y.size
		theta = theta - alpha*temp
		count+=1
	return theta

x = np.random.random(1000)
x1 = x;
#print(x)
x = np.column_stack((x**0,x,x**2,x**3,x**4))
y = 2 + 3*x.T[0] - 1*x.T[1] + 20*x.T[2]+0.1*x.T[3]
#print(y)
theta = np.random.random(5) 
theta = GradientDescent(theta,x,y)
print(theta)
print(y.shape)
print(x.shape)
plt.plot(x1,y,'r')
plt.plot(x1,Hypothesis(theta,x),'g')
plt.show()
