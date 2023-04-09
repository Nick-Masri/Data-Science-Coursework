import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from process import process

model = pickle.load(open("model.pickle", "rb"))

X = np.genfromtxt("KNN-sample-features.csv", delimiter = ",")
y = np.genfromtxt("KNN-sample-target.csv")

y_pred = model.predict(process(X))

print(confusion_matrix(y, y_pred))


