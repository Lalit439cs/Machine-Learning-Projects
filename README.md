# Machine-Learning-Projects
Herer are many machine learning projects ranging from classical models to Neural Networks

## 1. Judging a Book by its Cover  
• Build Neural Network Architectures ,**CNNs and RNNs on image(Book cover image) and text(Book title)
dataset of Books**, to predict the book genre amongst 30 possible categories<br>
• Implemented own CNNs and RNNs models using **PyTorch** ,for this classification task.<br>
• Used **BERT (for text dataset)** and **EfficientNetB4 (for image dataset)**.For this Multimodal,combined (by
Stacking) probabilities obtained from these models and then used **SVM Classifier** to get accuracy of **63.4%**<br>

**Note:** This project was created collaboratively by Lalit Meena and Prakul Virdi during COL774, Machine Learning course(Fall 2022,Prof. Parag Singla).

## 2.  Image Classification
• Used Support Vector Machines (**SVMs**) to build models for **binary classification and Multi-Class
Classification (one-vs-one classifier setting) for CIFAR-10 image dataset** <br>
• Solved this SVM optimization problem using a general purpose convex optimization package(**CVXOPT**) as well
as using a **scikit-learn library function (based on LIBSVM)**<br>
• Compared their **accuracy and training time** for solving this SVM dual problem using **Linear & Gaussian
Kernel**.Accuracy obtained- **85.6% (Binary ) & 62.44% (Multi-Class
Classification)**<br>
