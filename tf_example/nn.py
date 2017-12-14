import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nodes_layer1 = 500
nodes_layer2 = 500

layer_nodes = [500, 500, 10];


n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def nn_model(data):

	hidden_layers = [];

	for i, nodes in enumerate(layer_nodes):
		layer = {'weights': None, 'biases': None}
		if i == 0:
			weights = tf.Variable(tf.random_normal([784, nodes]))
			layer['weights'] = weights
		else:
			weights = tf.Variable(tf.random_normal([layer_nodes[i-1], nodes]))
			layer['weights'] = weights
		biases = tf.Variable(tf.random_normal([nodes]))
	
		layer['biases'] = biases
		hidden_layers.append(layer)

	

	layers = []
	layers.append(data)

	for i in range(len(hidden_layers)):
		li = tf.add(tf.matmul(layers[i], hidden_layers[i]['weights']), hidden_layers[i]['biases'])
		if(i != len(hidden_layers)-1):
			li = tf.nn.relu(li)
		layers.append(li)



	return layers[-1]


def network_model(data):
    hidden_1_layer =  {
                        'weights':tf.Variable(tf.random_normal([784,nodes_layer1])), 
                        'biases': tf.Variable(tf.random_normal([nodes_layer1]))
                      }
    
    hidden_2_layer = {
                        'weights':tf.Variable(tf.random_normal([nodes_layer1, nodes_layer2])), 
                        'biases': tf.Variable(tf.random_normal([nodes_layer2])) 
                     }
    
    output_layer = {
                        'weights': tf.Variable(tf.random_normal([nodes_layer2, n_classes])), 
                        'biases': tf.Variable(tf.random_normal([n_classes]))
                        
                    }   

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'] )

    return output


def train_nn(x, y):
    prediction  = nn_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, ' completed out of ', hm_epochs, 'loss',epoch_loss)



        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



train_nn(x, y)

"""
layer1 -> activation -> layer2 -> activation -> output

"""


