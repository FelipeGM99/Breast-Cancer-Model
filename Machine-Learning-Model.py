#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Data Analysis

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np


# ### Import Data

# In[38]:


dataset = load_breast_cancer()
X, y = dataset.data, dataset.target


# ### Splitting and test Data

# In[42]:


#X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=2.2).fit(X_train, y_train)
y_pred_svc = svm.predict(X_test)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_test)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_pred_svc)))
print('Recall: {:.2f}'.format(recall_score(y_test, y_pred_svc)))
print('F1: {:.2f}'.format(f1_score(y_test, y_pred_svc)))


# ### ML Model: Support Vector Machine

# In[31]:


C_values = np.arange(0.01, 5, 0.01)
Recall_values = np.array([])
Precision_values = np.array([])
F1_values = np.array([])

for i in C_values:
    svm = SVC(kernel='rbf', C=i).fit(X_train, y_train)
    y_pred_svc = svm.predict(X_test)
    Recall_values = np.append(Recall_values, recall_score(y_test, y_pred_svc))
    Precision_values = np.append(Precision_values, precision_score(y_test, y_pred_svc))    
    F1_values = np.append(F1_values, f1_score(y_test, y_pred_svc))    

plt.plot(C_values, Recall_values, label='Recall')
plt.plot(C_values, Precision_values, label='Precision')
plt.plot(C_values, F1_values, label='F1-score')
plt.legend()
plt.xlabel('C parameter')
plt.ylabel('Score')

plt.show()


# #### We choose C=2.2, because we are detecting breast cancer. We prefer a Recall-oriented model over a precision-oriented one. 

# ### Cross-validation

# In[49]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(svm, X, y, cv=5, scoring='recall')
print('Cross-validation Recall (5-fold): ', cv_scores)

