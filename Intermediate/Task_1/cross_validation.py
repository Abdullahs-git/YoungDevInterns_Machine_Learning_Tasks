from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import numpy as np

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(accuracy_score))
f1 = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(f1_score, average='macro'))

print("Accuracy per fold:", np.round(accuracy, 4))
print("F1-score per fold:", np.round(f1, 4))
print("Mean Accuracy:", round(np.mean(accuracy), 4))
print("Mean F1-score:", round(np.mean(f1), 4))
