import numpy as np
import autodiff as ad

def linear_regression():
    x = ad.Variable(name='x')
    y = ad.Variable(name='y')
    W = ad.Variable(name='W')
    output = ad.matmul_op(x, W)
    # loss function
    cost = 0.5 * ad.reduce_sum_op((y - output) * (y - output), axis=0)
    # cost = 0.5 * ad.matmul_op((y - output), (y - output), True, False)
    # gradient
    grad_cost_w, = ad.gradients(cost, [W])
    # construct data set
    # y = x
    num_point = 10
    x_data = np.array(range(num_point)).reshape((num_point, 1))
    y_data = x_data + np.random.uniform(-0.1, 0.1, (num_point, 1))
    x_data = np.concatenate([x_data, np.ones((num_point, 1))], axis=1)
    # initialize the parameters
    w_val = np.array([[0.5],[0.1]])
    excutor = ad.Executor([cost, grad_cost_w])
    # train
    n_epoch = 1000
    lr = 0.001
    cost_list = []
    print "training..."
    for i in range(n_epoch):
        # evaluate the graph
        cost_val, grad_cost_w_val = excutor.run(feed_dict={x: x_data, W: w_val, y: y_data})
        # update the parameters using GD
        print "cost: ", cost_val
        print "grad: ", grad_cost_w_val
        w_val = w_val - lr * grad_cost_w_val
        print "weight: ", w_val
        cost_list.append(cost_val)
    # numpy
    # cost_np = 0.5*np.sum((y_data-x_data.dot(w_val))**2, axis=0)
    # grad_np = x_data.T.dot(y_data-x_data.dot(w_val))
    # print "numpy cost: ", cost_np
    # print "numpy cost: ", grad_np

linear_regression()
