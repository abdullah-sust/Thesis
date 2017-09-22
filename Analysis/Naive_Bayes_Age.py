#Analysis with Age using Naive Bayes
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

# Initializing Training and Target data array
trainData =[]
trainTarget=[]
fileName="Facebook posts.csv"
# Openning the data set file
file = open(fileName, "r",encoding="utf8")
# Reading the data set file
data = csv.reader(file)
count =0
for col in data:
    if count==0:
        count=1
        continue
    count+=1
    if count==185:
        break
    post=col[5]+" "+col[6]+" "+col[7]+" "+col[8]+" "+col[9]+" "+col[10]+" "+col[11]+" "+col[12]+" "+col[13]+" "+col[14]
    trainData.append(post)
    trainTarget.append(int(col[4]))
#     if col[3]=='Female':
#         trainTarget.append(0)
#     else:
#         trainTarget.append(1)

# Creating input feature vector using TfidfVectorizer
vectorizer=TfidfVectorizer(use_idf=True, token_pattern='[^ \n,".\':()ঃ‘?’।“”!;a-zA-Z0-9#০১২৩৪৫৬৭৮৯*&_><+=%$-`~|^·]+') #০১২৩৪৫৬৭৮৯
trainData=vectorizer.fit_transform(trainData)
features=vectorizer.get_feature_names()
trainData=trainData.toarray()

# Initializing the Multinomial Naive Bayes model
clf= MultinomialNB()

# Fitting  Multinomial Naive Bayes model with trainData and trainTarget
clf.fit(trainData,trainTarget)

# Analyzing with 5 iteration
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(trainData, trainTarget, test_size=0.3)
    acuracy= clf.score(x_test,y_test)
    print("Acuracy is: ",acuracy,"\n")

# Acuracy is:  76.36

# Decision: Due to lack of lowest dataset our accuracy was also low. If we can enlarge the data set and consider age as level wise only then our accuracy will increase. 