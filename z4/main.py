import numpy as np
import matplotlib.pyplot as plt
import json
with open("data_banknote.json") as banknote:
    banknote = json.load(banknote)
with open("data_haberman.json") as haberman:
    haberman = json.load(haberman)
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, model_selection, decomposition, metrics


def classify_svc(dataset, split_size, kernel, C, gamma, title):
    """
    Splits the data and target into train and test parts. Also
    selects one of the two algorithms and allows modification of
    test-train proportion.
    Depending on the algorithm chosen uses either SVC or Decision Tree
    to classify given data.

    :param dataset: The dataset - with data and target values - to classify.
    :type dataset: dict
    :param split_size: Proportion of the dataset to include in the test split.
    :type split_size: float
    :param kernel: Linear model.
    :type kernel: str
    :param C: Regularization parameter. Higher the value, larger-margin separating hyperplane.
    :type C: int
    :param gamma: How far the influence reaches.
    Higher the value, only closest points to the separation line are considered.
    :type gamma: float
    :param title: Chart title
    :type title: str
    :return score: Value in range of 0-1 representing total score of the model.
    """

    """
    Create Train and Test splits of data and target variables
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset['data'], np.array(dataset['target']), test_size=split_size)

    """
    Create the SVC model and start training
    """
    svc = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    svc.fit(X_train, y_train)

    """
    Compare a section of model results with desired outcomes
    """
    y_model_outcome = svc.predict(X_test)
    print(f"First 10 results for {title}")
    print(f"Model: {y_model_outcome[0:10]}")
    print(f"Goal: {y_test[0:10]}")

    """
    Reduce dimension whilst keeping the data structure with the goal of visualizing the data on a simple x,y chart
    """
    pca = decomposition.PCA(n_components=2)
    X_train2 = pca.fit_transform(X_train)
    svc.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train, clf=svc, legend=2)
    plt.title(title)
    plt.show()

    """
    Calculate the score. Higher the value, better the model
    """
    score = metrics.accuracy_score(y_test, y_model_outcome)
    return score


if __name__ == "__main__":
    banknote_svc_result = classify_svc(banknote, 0.2, 'rbf', 10, 0.00001, 'Banknote Authenticity')
    haberman_svc_result = classify_svc(haberman, 0.2, 'rbf', 15000, 0.09, 'Haberman\'s Survival')
    print(f"Accuracy of SVC model:\nBanknote dataset: {banknote_svc_result}\nHaberman dataset: {haberman_svc_result}")
