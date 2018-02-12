cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int apply_tree(object tree, np.ndarray[double, ndim=1, mode="c"] xs):
    cdef np.ndarray[np.int64_t] children_left = tree.children_left
    cdef np.ndarray[np.int64_t] children_right = tree.children_right
    cdef np.ndarray[np.int64_t] feature = tree.feature
    cdef np.ndarray[np.float64_t] threshold = tree.threshold
    cdef int node_id = 0
    while children_left[node_id] >= 0:
        if xs[feature[node_id]] <= threshold[node_id]:
            node_id = children_left[node_id]
        else:
            node_id = children_right[node_id]
    return node_id


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
cdef eval_tree(object tree, np.ndarray[double, ndim=1, mode="c"] xs):
    cdef int node_id = apply_tree(tree, xs)
    cdef double left = tree.value[node_id][0][0]
    cdef double right = tree.value[node_id][0][1]
    return 1 if right > left else 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef eval_proba(object tree, np.ndarray[double, ndim=1, mode="c"] xs):
    cdef int node_id = apply_tree(tree, xs)
    cdef double left = tree.value[node_id][0][0]
    cdef double right = tree.value[node_id][0][1]
    # normalize
    cdef double total = left + right
    if total == 0.0:
        total = 1.0
    right /= total
    return right


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def eval_forest(object forest, np.ndarray[double, ndim=1, mode="c"] xs):
    cdef int n_estimators = forest.n_estimators
    cdef double result = 0.0
    cdef double proba
    cdef object tree
    for i in range(n_estimators):
        tree = forest.estimators_[i].tree_
        proba = eval_proba(tree, xs)
        result += proba
    result /= n_estimators
    return 1.0 if result > 0.5 else 0.0