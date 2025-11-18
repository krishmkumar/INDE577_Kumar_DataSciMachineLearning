# Perceptron Classifier

In the notebook `perceptron_demo.ipynb`, we implement and experiment with one of the fundamental linear models in machine learning: the perceptron classifier.  
The perceptron is a binary linear classifier that updates its weights whenever it misclassifies an observation. Unlike logistic regression or SVMs, the perceptron uses a simple rule-based update and does not optimize a convex loss function.

The perceptron is best suited to linearly separable datasets, and its behavior depends on the learning rate, number of iterations, and proper feature scaling.

In this example, we implement the perceptron entirely from scratch inside the `rice2025.supervised_learning` module and apply it to a real-world digits dataset (8×8 pixel representation).  
To convert this into a binary classification task, we predict whether the digit is an “8”.

## In this project, we:

- Build a complete perceptron classifier from scratch, supporting `fit`, `predict`, `accuracy`, and training-loss tracking.
- Apply our custom `StandardScaler` and `train_test_split` utilities.
- Use a real digits dataset stored locally as `digits.csv`, converting it into a binary prediction task.
- Visualize the training loss over epochs to understand convergence.
- Evaluate the perceptron using predictive accuracy on held-out test data.
- Validate that our implementation behaves consistently with the classic perceptron update rule.

This notebook demonstrates the perceptron algorithm end-to-end and confirms the correctness of the implementation included in the `rice2025` package.
