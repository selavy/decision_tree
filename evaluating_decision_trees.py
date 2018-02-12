# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:03:26 2018

@author: Peter Lesslie
"""

# %%

import numpy as np
import pandas as pd
import patsy.highlevel
from scipy.sparse import issparse
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True, build_dir='.')
from fast_decision_tree import eval_tree

N = 42
xs = np.random.lognormal(mean=5.0, sigma=2.0, size=N)
ys = np.random.lognormal(mean=7.0, sigma=9.0, size=N)
zs = np.random.lognormal(mean=0.0, sigma=1.0, size=N)
ms = np.random.normal(loc=0.0, scale=1.0, size=N)

df = pd.DataFrame(data={
        'x': xs,
        'y': ys,
        'z': zs,
        'm': ms,
        })

df['c'] = np.where(df['m'] > 0.0, 1.0, 0.0)

# %%

formula = "c ~ x + y + z"
y, X = patsy.highlevel.dmatrices(formula, data=df)
y = y.flatten()

# %%

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1,
                             criterion="gini",
                             max_depth=5,
                             min_samples_leaf=2,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=1,
                             random_state=0)

clf.fit(X, y)
y_pred = clf.predict(X)

# %%

# Method #1: reproduce y_pred by using underlying estimator
est = clf.estimators_[0]
y_pred2 = est.predict(X)
result2 = np.allclose(y_pred, y_pred2)
print("y_pred =?= y_pred2: {}".format(result2))

# %%

# Method #2: reproduce y_pred by using predict_proba()
# N.B.: it seems to assign 50% probability to class 0
probs3 = est.predict_proba(X)
y_pred3 = np.where(probs3[:, 1] > 0.5, 1.0, 0.0)
result3 = np.allclose(y_pred, y_pred3)
print("y_pred =?= y_pred3: {}".format(result3))

# %%

tt = est.tree_
X32 = X.astype(np.float32)
proba = tt.predict(X32)
#classes = np.asarray([0., 1.])
#y_pred4 = classes.take(np.argmax(proba, axis=1), axis=0)
y_pred4 = np.argmax(proba, axis=1)
result4 = np.allclose(y_pred, y_pred4)
print("y_pred =?= y_pred4: {}".format(result4))

# %%

#cpdef np.ndarray predict(self, object X):
#    """Predict target for X."""
#    out = self._get_value_ndarray().take(self.apply(X), axis=0,
#                                         mode='clip')
#    if self.n_outputs == 1:
#        out = out.reshape(X.shape[0], self.max_n_classes)
#    return out

tree = est.tree_
X32 = X.astype(np.float32)
xx = tree.apply(X32)
#node_count = tree.node_count
#n_outputs = tree.n_outputs
#max_n_classes = tree.max_n_classes
#shape = (node_count, n_outputs, max_n_classes)
# Note: ‘clip’ mode means that all indices that are too large are replaced by the
# index that addresses the last element along that axis. Note that this
# disables indexing with negative numbers.
proba2 = tree.value.take(xx, axis=0) #, mode='clip')
proba3 = proba2[:, 0]
y_pred5 = np.argmax(proba3, axis=1)
result5 = np.allclose(y_pred, y_pred5)
print("y_pred =?= y_pred5: {}".format(result5))

#cdef np.ndarray _get_value_ndarray(self):
#    """Wraps value as a 3-d NumPy array.
#    The array keeps a reference to this Tree, which manages the underlying
#    memory.
#    """
#    cdef np.npy_intp shape[3]
#    shape[0] = <np.npy_intp> self.node_count
#    shape[1] = <np.npy_intp> self.n_outputs
#    shape[2] = <np.npy_intp> self.max_n_classes
#    cdef np.ndarray arr
#    arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
#    Py_INCREF(self)
#    arr.base = <PyObject*> self
#    return arr

# %%

assert(not issparse(X32))
assert(X32.dtype == np.float32)
n_samples = X32.shape[0]
out = np.zeros((n_samples, ), dtype=np.intp)

#eval_tree(tree, X32)

#for i in range(n_samples):
#    node = self.nodes
#    # While node not a leaf
#    while node.left_child != _TREE_LEAF:
#        # ... and node.right_child != _TREE_LEAF:
#        if X_ptr[X_sample_stride * i +
#                 X_fx_stride * node.feature] <= node.threshold:
#            node = &self.nodes[node.left_child]
#        else:
#            node = &self.nodes[node.right_child]

# %%

#X_sample_stride = X.strides[0] // X.itemsize
#X_fx_stride = X.strides[1] // X.itemsize
#n_samples = X.shape[0]

# %%

#cdef inline np.ndarray _apply_dense(self, object X):
#    """Finds the terminal region (=leaf node) for each sample in X."""
#
#    # Check input
#    if not isinstance(X, np.ndarray):
#        raise ValueError("X should be in np.ndarray format, got %s"
#                         % type(X))
#
#    if X.dtype != DTYPE:
#        raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
#
#    # Extract input
#    cdef np.ndarray X_ndarray = X
#    cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
#    cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
#    cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
#    cdef SIZE_t n_samples = X.shape[0]
#
#    # Initialize output
#    cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
#    cdef SIZE_t* out_ptr = <SIZE_t*> out.data
#
#    # Initialize auxiliary data-structure
#    cdef Node* node = NULL
#    cdef SIZE_t i = 0
#
#    with nogil:
#        for i in range(n_samples):
#            node = self.nodes
#            # While node not a leaf
#            while node.left_child != _TREE_LEAF:
#                # ... and node.right_child != _TREE_LEAF:
#                if X_ptr[X_sample_stride * i +
#                         X_fx_stride * node.feature] <= node.threshold:
#                    node = &self.nodes[node.left_child]
#                else:
#                    node = &self.nodes[node.right_child]
#
#            out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset
#    return out
