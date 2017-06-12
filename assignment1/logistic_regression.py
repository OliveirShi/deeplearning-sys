import numpy as np
import autodiff as ad



x = ad.Variable(name = "x")
w = ad.Variable(name = "w")
b = ad.Variable(name = "b")
labels = ad.Variable(name = "lables")


# Define Computation graph

p = 1.0 / (1.0 + ad.exp_op((-1.0 * ad.matmul_op(w, x))))

loss = -1.0 * ad.reduce_sum_op(labels * ad.log_op(p) + (1.0 - labels) * ad.log_op(1.0 - p), axis = 1)

grad_y_w, = ad.gradients(loss, [w])



num_features = 2
num_points = 500
num_iterations = 1000
learning_rate = 0.001

# The dummy dataset consists of two classes.
# The classes are modelled as a random normal variables with different means.

class_1 = np.random.normal(2, 0.1, (num_points / 2, num_features))
class_2 = np.random.normal(4, 0.1, (num_points / 2, num_features))
x_val = np.concatenate((class_1, class_2), axis = 0).T

x_val = np.concatenate((x_val, np.ones((1, num_points))), axis = 0)
w_val = np.random.normal(size = (1, num_features + 1))
print x_val.shape

labels_val = np.concatenate((np.zeros((class_1.shape[0], 1)), np.ones((class_2.shape[0], 1))), axis=0).T
print labels_val.shape
executor = ad.Executor([loss, grad_y_w])

for i in xrange(100000):
	# evaluate the graph
	loss_val, grad_y_w_val =  executor.run(feed_dict={x:x_val, w:w_val, labels:labels_val})
	# update the parameters using SGD
	w_val = w_val - learning_rate * grad_y_w_val
	if i % 1000 == 0:
		print loss_val
		print grad_y_w_val