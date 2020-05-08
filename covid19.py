# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python [conda env:env]
#     language: python
#     name: conda-env-env-py
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier


# %%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %%
train.head(2)

# %%
sentences = train['text'].values
y = train['target'].values
sentences_submit = test['text'].values

# %%
sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, y, test_size=0.25, random_state=1000)

# %%
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

# %%
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_submit = vectorizer.transform(sentences_submit)

# %%
classifier = LogisticRegression()

# %%
classifier.fit(X_train, y_train)

# %%
score = classifier.score(X_test, y_test)

# %%
score


# %%
def rmse(y_true,y_preds,extras=None):
    return np.sqrt(mean_squared_error(y_true,y_preds))


# %%
print(rmse(classifier.predict(X_test),y_test))

# %%
y_pred = classifier.predict(X_submit)

# %%
y_pred = model.predict(X_submit)

# %% [markdown]
# ### CatBoost

# %%
model = CatBoostClassifier(learning_rate=0.03,
                           eval_metric='AUC')

# %% {"jupyter": {"outputs_hidden": true}}
model.fit(X_train, y_train)

# %%
print(rmse(model.predict(X_test),y_test))

# %%



# %%

# %%

# %%

# %% [markdown]
# ### Submission

# %%
submit = pd.read_csv("sample_sub.csv")

# %%
submit["target"]=y_pred

# %%
submit.to_csv("sub1.csv",index=False)

# %%

# %%
