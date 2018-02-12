cimport cython
cimport numpy as np
import numpy as np

def eval_tree(object tree, np.ndarray[double] xs):
#    cdef np.ndarray[np.int64] children_left = tree.children_left
#    cdef np.ndarray[np.int64] children_right = tree.children_right
    
    node_id = 0
    while tree.children_left[node_id] >= 0:
        if xs[tree.feature[node_id]] <= tree.threshold[node_id]:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]
    return node_id
    