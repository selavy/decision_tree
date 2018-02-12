# %%

import pstats, cProfile
import numpy as np
import pandas as pd
import patsy.highlevel
from sklearn.ensemble import RandomForestClassifier

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True, build_dir='.')
from fast_decision_tree import eval_forest

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

clf = RandomForestClassifier(n_estimators=100,
                             criterion="gini",
                             max_depth=5,
                             min_samples_leaf=2,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=1,
                             random_state=0)

clf.fit(X, y)

# %%

def run(X):
    y_pred2 = np.zeros(len(X), dtype=np.double)
    for i in range(len(X)):
        y_pred2[i] = eval_forest(clf, X[i, :])

cProfile.runctx("run(X)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()