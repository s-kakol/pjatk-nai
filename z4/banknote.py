import numpy as np
import matplotlib.pyplot as plt
import json
with open("data_banknote.json") as banknote_data:
    banknote_data = json.load(banknote_data)
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, model_selection, decomposition, metrics

"""
Load banknote data and target outcomes to separate variables
"""
X = banknote_data['data']
y = np.array(banknote_data['target'])

"""
Create Train and Test splits of data and target variables
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

"""
Create the SVC model and start training
"""
svc = svm.SVC(kernel='poly')
svc.fit(X_train, y_train)

"""
Compare a section model result with desired outcomes
"""
y_model_outcome = svc.predict(X_test)
print(f"Model: {y_model_outcome[0:10]}")
print(f"Goal: {y_test[0:10]}")

"""
Reduce dimension whilist keeping the data structure with the goal of visualizing the data on a simple x,y chart
"""
pca = decomposition.PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
svc.fit(X_train2, y_train)
plot_decision_regions(X_train2, y_train, clf=svc, legend=2)
plt.show()

"""
Calculate and display the score for recognizing fakes and real banknotes
Higher the value, better the model
"""
score = metrics.f1_score(y_test, y_model_outcome, average=None)
print(score)
