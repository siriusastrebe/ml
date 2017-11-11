# ml
Following the tutorials in Python Machine Learning by Sebastian Raschka

OS X Python Virtual environments
===
http://www.marinamele.com/2014/07/install-python3-on-mac-os-x-and-use-virtualenv-and-virtualenvwrapper.html

mkvirtualenv ml
pip install pandas
pip install sklearn
pip install numpy
pip install scipy
pip install matplotlib

Matplotlib Backend Error
===
https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python



Chapter 3
===
Perceptron Classifier

Logistic Regression
* Actually a classifier despite its name
* Linear classifier, but converges unlike the perceptron
* Regularization
  * 10^C where C is the 

SVM
* Maximizes margin between classification
* Kernalizable
  * Linear kernal
  * Radial Basis Function kernel, Gaussian kernal (RBF kernel)
* Gamma function affects cutoff, decision boundary


Decision Trees
* information gain
* Gini Impurity
* Entropy
* Classification Error

Random Forests
* Ensemble learning
* Weak learners combined to create a strong learner


K-Nearest Neighbors
* Minkowski uses a generalized Euclidean and Manhattan Distance to determine high-dimensionality distance


Outside readings
===
Activation Funcions:
Sigmoid
Relu
Softmax Functions
