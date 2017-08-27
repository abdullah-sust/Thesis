
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from numpy import array
import numpy as np
from sklearn import datasets, linear_model
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier

trainData =[]
trainTarget=[]
fileName="Facebook posts.csv"
file = open(fileName, "r",encoding="utf8")
data = csv.reader(file)
count =0
for col in data:
    if count==0:
        count=1
        continue
    count+=1
    if count==158:
        break
    post=col[5]+" "+col[6]+" "+col[7]+" "+col[8]+" "+col[9]+" "+col[10]+" "+col[11]+" "+col[12]+" "+col[13]+" "+col[14]
    trainData.append(post)
    if col[15]=="Muslim":
        trainTarget.append(0)
    elif col[15]=="Hindu":
        trainTarget.append(1)
    elif col[15]=="Buddha":
        trainTarget.append(2)
    elif col[15]=="Christian":
        trainTarget.append(3)

vectorizer=TfidfVectorizer(use_idf=True, token_pattern='[^ \n,".\':()ঃ‘?’।“”!;a-zA-Z0-9#০১২৩৪৫৬৭৮৯*&_><+=%$-`~|^·]+') #০১২৩৪৫৬৭৮৯
trainData=vectorizer.fit_transform(trainData)
features=vectorizer.get_feature_names()
trainData=trainData.toarray()
clf= MultinomialNB()
clf.fit(trainData,trainTarget)

for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(trainData, trainTarget, test_size=0.3)
    acuracy= clf.score(x_test,y_test)
    print("Acuracy is: ",acuracy,"\n")

# Acuracy is:  0.659574468085

# Acuracy is:  0.723404255319

# Acuracy is:  0.787234042553

# Acuracy is:  0.851063829787

# Acuracy is:  0.595744680851
