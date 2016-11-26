import numpy as np
import random

#linear regression

#Define hyposthesis
# H(x) = theta0 + theta1*x
def Hypothesis(theta0,theta1,x):
	return theta0 + theta1*x



#define Cost Function
#J(theta0,theta1) = 1/2m*(sum((H(xi)-yi)**2))
def CostFunction(theta0,theta1,x,y):
	m = x.size
	J = (Hypothesis(theta0,theta1,x)-y)**2
	J1 = np.sum(J)/(2*m)
	return J1

#minimize Cost Function	
def GradientDescent(theta1,theta0,x,y,max_iterations=10000,alpha= 0.04):
	count = 0
	m = x.size
	while(count!=max_iterations):
		temp0 =theta0 - alpha * (np.sum((Hypothesis(theta0,theta1,x)-y))/m)
		temp1 = theta1- alpha * (np.sum((Hypothesis(theta0,theta1,x)-y)*(x))/m)
		theta0 = temp0
		theta1 = temp1
		count+=1
	return theta0,theta1


a= int(input())
x = np.array([1,2,3,4])
y = 1000.23*x + 0.93
theta0 = random.randrange(1,100)
theta1 = random.randrange(0,2)
theta1,theta2 = GradientDescent(theta1,theta0,x,y,max_iterations=a,alpha=0.05)
print(theta1)
print(theta2)
