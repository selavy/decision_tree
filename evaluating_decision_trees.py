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
#import pyximport
#pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True, build_dir='.')
#from fast_decision_tree import eval_tree

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

tree = est.tree_
X32 = X.astype(np.float32)
xx = tree.apply(X32)
# Note: ‘clip’ mode means that all indices that are too large are replaced by the
# index that addresses the last element along that axis. Note that this
# disables indexing with negative numbers.
proba2 = tree.value.take(xx, axis=0) #, mode='clip')
proba3 = proba2[:, 0]
y_pred5 = np.argmax(proba3, axis=1)
result5 = np.allclose(y_pred, y_pred5)
print("y_pred =?= y_pred5: {}".format(result5))

# %%

assert(not issparse(X32))
assert(X32.dtype == np.float32)
n_samples = X32.shape[0]
out = np.zeros((n_samples, ), dtype=np.intp)

#for i in range(n_samples):
#    node_id = 0
#    while tree.children_left[node_id] >= 0:
#        if X[i, tree.feature[node_id]] <= tree.threshold[node_id]:
#            node_id = tree.children_left[node_id]
#        else:
#            node_id = tree.children_right[node_id]
#    out[i] = node_id

def eval_tree(tree, xs):
    node_id = 0
    while tree.children_left[node_id] >= 0:
        if xs[tree.feature[node_id]] <= tree.threshold[node_id]:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]
    return node_id
    
for i in range(n_samples):
    out[i] = eval_tree(tree, X32[i, :])

proba3 = tree.value.take(out, axis=0) #, mode='clip')
proba4 = proba3[:, 0]
y_pred6 = np.argmax(proba4, axis=1)
result6 = np.allclose(y_pred, y_pred6)
print("y_pred =?= y_pred6: {}".format(result6))
