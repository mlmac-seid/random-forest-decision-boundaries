# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns
import pandas as pd

# 3d figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# creating animations
import matplotlib.animation
from IPython.display import HTML

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

!pip install dtreeviz

import dtreeviz
from dtreeviz import decision_boundaries

from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data

X.shape

"""This dataset has 13 features:"""

wine.feature_names

X = X[:,[12,6]]
y = wine.target

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, n_jobs=-1)
rf.fit(X, y)

fig,axes = plt.subplots(1,1,dpi=300)
decision_boundaries(rf, X, y, ax=axes,
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'],
       feature_names=['proline', 'flavanoid']);

rf1 = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, max_depth=5, n_jobs=-1)
rf1.fit(X, y)

fig,axes = plt.subplots(1,1,dpi=300)
decision_boundaries(rf1, X, y, ax=axes,
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'],
       feature_names=['proline', 'flavanoid']);

rf2 = RandomForestClassifier(n_estimators=50, min_samples_leaf=50, max_depth=5, n_jobs=-1)
rf2.fit(X, y)

fig,axes = plt.subplots(1,1,dpi=300)
decision_boundaries(rf2, X, y, ax=axes,
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'],
       feature_names=['proline', 'flavanoid']);

rf3 = RandomForestClassifier(n_estimators=10, min_samples_leaf=50, max_depth=5, n_jobs=-1)
rf3.fit(X, y)

fig,axes = plt.subplots(1,1,dpi=300)
decision_boundaries(rf3, X, y, ax=axes,
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'],
       feature_names=['proline', 'flavanoid']);

rf4 = RandomForestClassifier(n_estimators=50, min_samples_leaf=50, max_depth=50, n_jobs=-1)
rf4.fit(X, y)

fig,axes = plt.subplots(1,1,dpi=300)
decision_boundaries(rf4, X, y, ax=axes,
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'],
       feature_names=['proline', 'flavanoid']);