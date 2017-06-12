import numpy as np

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        """subtracting other from self return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_byconst_op(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(other, self)
        else:
            new_node = sub_fromconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """TODO: Your code here"""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_byconst_op(self, other)
        return new_node

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(other, self)
        else:
            new_node = div_fromconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__
    __div__  = __truediv__
    __rdiv__ = __rtruediv__

    def __str__(self):
        """Allow print to display node name.""" 
        return self.name

def Variable(name):
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node
        
    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """

        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]

class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]

class SubOp(Op):
    """Op to element-wise subtract two nodes"""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise subtraction."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        '''Y = A - B
        dA = dY
        dB = -dY'''
        return [output_grad, -1.0 * output_grad]

class SubByConstOp(Op):
    """Op to element-wise subtract a node by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of an input node and a constant, return result of element-wise subtraction."""
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]

class SubFromConstOp(Op):
    """Op to element-wise subtract a node from a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of an input node and a constant, return result of element-wise subtraction."""
        assert len(input_vals) == 1
        return node.const_attr - input_vals[0]

    def gradient(self, node, output_grad):
        return [-1.0 * output_grad]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
        Y = A * B
        """
        """TODO: Your code here"""
        return [node.inputs[1]*output_grad, node.inputs[0]*output_grad]

class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 1
        return node.const_attr * input_vals[0]

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        """TODO: Your code here"""
        return [node.const_attr * output_grad]

class DivOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
        Y = A / B
        """
        # problem
        return [1.0 / node.inputs[1] * output_grad,
                -1.0 * (node.inputs[0] / (node.inputs[1] * node.inputs[1])) * output_grad]

class DivByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "(%s/%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad / node.const_attr]

class DivFromConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "(%s/%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr / input_vals[0]

    def gradient(self, node, output_grad):
        return [(-1.0 * node.const_attr / (node.inputs[0] * node.inputs[0])) * output_grad]

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 2
        if node.matmul_attr_trans_A:
            input_vals[0] = input_vals[0].T
        if node.matmul_attr_trans_B:
            input_vals[1] = input_vals[1].T
        return input_vals[0].dot(input_vals[1])


    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        """TODO: Your code here"""
        if node.matmul_attr_trans_A and node.matmul_attr_trans_B:
            # dY/dX1 = (YX2)^T = X2^T Y^T; (Y, X2, F, F).T = (X2, Y, T, T)
            # dY/dX2 = (X1Y)^T = Y^T X1^T; (X1, Y, F, F).T = (Y, X1, T, T)
            return [matmul_op(node.inputs[1], output_grad, trans_A=True, trans_B=True),
                    matmul_op(output_grad, node.inputs[0], trans_A=True, trans_B=True)]
        elif node.matmul_attr_trans_A and not node.matmul_attr_trans_B:
            # dY/dX1 = (YX2^T)^T = X2 Y^T; (Y, X2, F, T)^T = (X2, Y, F, T)
            # dY/dX2 = X1 Y; (X1, Y, F, F)
            return [matmul_op(node.inputs[1], output_grad, False, True),
                    matmul_op(node.inputs[0], output_grad, False, False)]
        elif not node.matmul_attr_trans_A and node.matmul_attr_trans_B:
            # dY/dX1 = (Y, X2, F, F)
            # dY/dX2 = (X1^T Y)^T = Y^T X1; (X1, Y, T, F)^T = (Y, X1, T, F)
            return [matmul_op(output_grad, node.inputs[1], False, False),
                    matmul_op(output_grad, node.inputs[0], True, False)]
        else:
            # dY/dX1 = (Y, X2, F, T)
            # dY/dX2 = (X1, Y, T, F)
            return [matmul_op(output_grad, node.inputs[1], trans_A=False, trans_B=True),
                    matmul_op(node.inputs[0], output_grad, trans_A=True, trans_B=False)]

class ReduceSumOp(Op):
    """summarizing a node in an axis"""
    def __call__(self, node_A, axis=0):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.axis = axis
        new_node.name = "reduce_sum(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=node.axis)

    def gradient(self, node, output_grad):
        return [output_grad]

class LogOp(Op):
    """Op to element-wise and represent np.log operator"""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "log(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert (isinstance(input_vals[0], np.ndarray))
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [1.0 / node.inputs[0] * output_grad]

class ExpOp(Op):
    """Op to element-wise and represent np.exp operator"""
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "exp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(input_vals[0], np.ndarray)
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [exp_op(node.inputs[0]) * output_grad]

class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
add_byconst_op = AddByConstOp()
sub_byconst_op = SubByConstOp()
sub_fromconst_op = SubFromConstOp()
mul_byconst_op = MulByConstOp()
div_byconst_op = DivByConstOp()
div_fromconst_op = DivFromConstOp()
matmul_op = MatMulOp()
exp_op = ExpOp()
log_op = LogOp()
reduce_sum_op = ReduceSumOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

class Executor:
    """Executor computes values for a given subset of nodes in a computation graph.""" 
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        """TODO: Your code here"""

        for node in topo_order:
            if node in node_to_val_map:
                continue
            else:
                node_to_val_map[node] = node.op.compute(node, [node_to_val_map[in_node] for in_node in node.inputs])


        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    """TODO: Your code here"""
    for node in reverse_topo_order:
        node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
        grads = node.op.gradient(node, node_to_output_grad[node])
        for idx, in_node in enumerate(node.inputs):
            if in_node in node_to_output_grads_list:
                node_to_output_grads_list[in_node].append(grads[idx])
            else:
                node_to_output_grads_list[in_node] = [grads[idx]]

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods ####### 
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
