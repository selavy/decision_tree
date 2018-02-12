# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:02:08 2018

@author: Peter Lesslie
"""

# %%

import time
import numpy as np
import pandas as pd
import patsy.highlevel
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True, build_dir='.')
from fast_decision_tree import eval_tree

# %%

N = 500
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

# %% ---- vectorized version

start = time.time()
y_pred2 = clf.predict(X)
end = time.time()
print("Took {:0.2f} seconds.".format(end - start))
result = np.allclose(y_pred, y_pred2)
print("Result #1: {}".format(result))
      
# %% ---- non-vectorized version

start = time.time()
y_pred2 = np.zeros(len(X), dtype=np.double)
for i in range(len(X)):
    y_pred2[i] = clf.predict(X[i, :].reshape(1, -1))
end = time.time()
print("Took {:0.2f} seconds.".format(end - start))
result = np.allclose(y_pred, y_pred2)
print("Result #2: {}".format(result))

# %% ---- non-vectorized version #2

start = time.time()
y_pred2 = np.zeros(len(X), dtype=np.double)
vals = np.zeros((1, X.shape[1]), dtype=np.double)
for i in range(len(X)):
    for j in range(len(vals)):
        vals[0][j] = X[i, j]
    y_pred2[i] = clf.predict(vals)
end = time.time()
print("Took {:0.2f} seconds.".format(end - start))
result = np.allclose(y_pred, y_pred2)
print("Result #2: {}".format(result))
      
# %% ---- 
      
      