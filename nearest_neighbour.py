#nearest neighbour classifier
from sklearn import datasets
import numpy as np

#utility function to calculate distance
def dist(x,y):
	return np.sqrt(np.sum((x-y)**2))


#load data from a to b
def load_data(a,b):
	x_train = (datasets.load_digits())['data'][a:b]
	y_train = (datasets.load_digits())['target'][a:b]
	return x_train,y_train

#predict the digit using training data
def predict_digit(x_train,y_train,x_test):
	distance = []
	for i in x_train:
		distance.append(dist(i,x_test))
	min_distance = min(distance)
	ans_index = distance.index(min_distance)
	return y_train[ans_index]

def accuracy(x_train,y_train,a,b):
	no_error = 0;
	x_test = (datasets.load_digits())['data'][1700:]
	y_test = (datasets.load_digits())['target'][1700:]
	for i in range(b-a):
		
		if predict_digit(x_train,y_train,x_test[i]) != y_test[i]:	
			no_error+=1
	return no_error

def main():
	#load the training data
	x_train,y_train = load_data(0,1699)

	#load the test data
	x_test = (datasets.load_digits())['data'][1750] 
	ans = (datasets.load_digits())['target'][1750]

	#predict
	prediction = predict_digit(x_train,y_train,x_test)

	#print result
	print("The prediction is "+str(prediction))
	print("The answer is " + str(ans))
	print("The errors of algorithms is "+str(accuracy(x_train,y_train,1700,1797)))

if __name__ == "__main__":
	main()	
