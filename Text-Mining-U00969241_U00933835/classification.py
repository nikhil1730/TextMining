from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
import warnings

tf_X, tf_Y = load_svmlight_file('training_data_file.TF')
idf_X, idf_Y = load_svmlight_file('training_data_file.IDF')
tfidf_X, tfidf_Y = load_svmlight_file('training_data_file.TFIDF')

tf_X_train, tf_X_test, tf_Y_train, tf_Y_test = train_test_split(tf_X, tf_Y, test_size=0.4, random_state=0)
idf_X_train, idf_X_test, idf_Y_train, idf_Y_test = train_test_split(idf_X, idf_Y, test_size=0.4, random_state=0)
tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test = train_test_split(tfidf_X, tfidf_Y, test_size=0.4, random_state=0)

model_MNB = MultinomialNB() # TF
model_BNB = BernoulliNB(alpha=0.001) #IDF
model_KNC = KNeighborsClassifier(weights='distance') #TFIDF
model_SVC = SVC(gamma='scale') #TFIDF

modelClassifiers = [model_MNB, model_BNB, model_KNC, model_SVC]
scoringMethod = ['f1_macro','precision_macro', 'recall_macro']

for model in modelClassifiers:
    warnings.filterwarnings("ignore")
    classfierName = ''
    
    if model == model_MNB:
        classfierName = 'MultinomialNB'
        train_data = tf_X
        test_data = tf_Y
    elif model == model_BNB:
        classfierName = 'BernoulliNB'
        train_data = idf_X
        test_data = idf_Y
    elif model == model_KNC:
        classfierName = 'KNeighborsClassifier'
        train_data = tfidf_X
        test_data = tfidf_Y
    else:
        classfierName = 'SVC'
        train_data = tfidf_X
        test_data = tfidf_Y
    print('======= ' + classfierName + ' =======')
    for crossValidation in scoringMethod:
        scores = cross_val_score(model, train_data, test_data, cv=5, scoring=crossValidation)
        print(crossValidation + "accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

