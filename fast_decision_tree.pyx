cimport cython
cimport numpy as np
import numpy as np
#cimport sklearn.tree._tree
#from sklearn.tree._tree import Tree
#cimport numpy.tree._tree


ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

DEF TREE_LEAF = -1
DEF TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples # Weighted number of samples at the node

NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})
    
cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes


cpdef np.ndarray eval_tree(object o, object X):
    """Finds the terminal region (=leaf node) for each sample in X."""
    cdef Tree t = <Tree>o;

    # Check input
    if not isinstance(X, np.ndarray):
        raise ValueError("X should be in np.ndarray format, got %s"
                         % type(X))

    if X.dtype != DTYPE:
        raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    # Extract input
    cdef np.ndarray X_ndarray = X
    cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
    cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
    cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
    cdef SIZE_t n_samples = X.shape[0]

    # Initialize output
    cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
    cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    # Initialize auxiliary data-structure
    cdef Node* nodes = t.nodes
    cdef Node* node = NULL
    cdef SIZE_t i = 0
    
    print("beginning eval...")

    for i in range(n_samples):
        print("i = {}".format(i))
        node = nodes
        print("Address of node: {0:x}".format(<unsigned int>&node))
        print("node.left_child == {}".format(node.left_child))
        # While node not a leaf
        while node.left_child != _TREE_LEAF:
            # ... and node.right_child != _TREE_LEAF:
            if X_ptr[X_sample_stride * i +
                     X_fx_stride * node.feature] <= node.threshold:
                node = &nodes[node.left_child]
            else:
                node = &nodes[node.right_child]

        out_ptr[i] = <SIZE_t>(node - nodes)  # node offset
    return out


#cpdef apply(self, np.ndarray[DTYPE_t, ndim=2] X):
#    """Finds the terminal region (=leaf node) for each sample in X."""
#    cdef int i = 0
#    cdef int n_samples = X.shape[0]
#    cdef int node_id = 0
#
#    cdef np.ndarray[np.int32_t, ndim=1] out
#    out = np_zeros((n_samples, ), dtype=np.int32)
#
#    for i from 0 <= i < n_samples:
#        node_id = 0
#
#        # While node_id not a leaf
#        while self.children_left[node_id] != _TREE_LEAF: # and self.children_right[node_id] != _TREE_LEAF:
#            if X[i, self.feature[node_id]] <= self.threshold[node_id]:
#                node_id = self.children_left[node_id]
#            else:
#                node_id = self.children_right[node_id]
#
#        out[i] = node_id
#    return out