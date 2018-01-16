// Most basic decision tree using sklearn.
// Stolen from https://medium.com/totvslabs/beginning-machine-learning-by-a-software-engineer-in-a-hurry-ad58412145b8

from sklearn import tree
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))
