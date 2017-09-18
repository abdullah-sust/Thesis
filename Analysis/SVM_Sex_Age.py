#Individual Doc with Age & Sex
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
    if count==185:
        break
#     post=col[5]+" "+col[6]+" "+col[7]+" "+col[8]+" "+col[9]+" "+col[10]+" "+col[11]+" "+col[12]+" "+col[13]+" "+col[14]
#     trainData.append(post)
    for i in range(10):
        trainData.append(col[i+5])
        if col[3]=='Female':
            t=str(col[4])+"0"
        else:
            t=str(col[4])+"1"
        trainTarget.append(int(t))

vectorizer=TfidfVectorizer(use_idf=True, token_pattern='[^ \n,".\':()ঃ‘?’।“”!;a-zA-Z0-9#০১২৩৪৫৬৭৮৯*&_><+=%$-`~|^·]+') #০১২৩৪৫৬৭৮৯
trainData=vectorizer.fit_transform(trainData)
features=vectorizer.get_feature_names()
model = svm.SVC(kernel='linear', C=1, gamma=1)
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(trainData, trainTarget, test_size=0.3)
    model.fit(x_train, y_train)
    predicted2 = model.predict(x_test)
    count2 = 0
    for i in range(len(predicted2)):
        if (predicted2[i]-y_test[i])==0:
            count2 += 1
    print(float(count2)/float(len(predicted2)))

# Accuracy:
# 0.34615384615384615
# 0.3076923076923077
# 0.2948717948717949
# 0.24358974358974358
# 0.34615384615384615