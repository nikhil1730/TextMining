from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt
import warnings
import time

start = time.time()
training_tf, test_tf = load_svmlight_file('training_data_file.TF')
training_idf, test_idf = load_svmlight_file('training_data_file.IDF')
training_tfidf, test_tfidf = load_svmlight_file('training_data_file.TFIDF')

print(training_tf.shape)

# topFeatures = [60, 370, 4500, 16500, 21000]
# topFeatures = [9, 111, 777, 1111, 5555, 11111, 17777, 21111]
# topFeatures = [100, 200, 400, 800, 1600, 2400, 3200, 6400, 8000, 10000, 12800, 15000, 17500, 19999]
# topFeatures = [100, 2400, 3200, 10000, 17500, 19999]
# topFeatures = [100, 3200, 17500]
topFeatures = [100, 200, 800, 2400, 6400, 8000, 10000, 15000, 19999]

model_MNB = MultinomialNB() # TF
model_BNB = BernoulliNB(alpha=0.001) #IDF
model_KNC = KNeighborsClassifier(weights='distance') #TFIDF
model_SVC = SVC(gamma='scale') #TFIDF
modelClassifiers = [model_MNB, model_BNB, model_KNC, model_SVC]
tests = ['chi2', 'mic']
scoresValidation = {}
chi2_MNB = []
chi2_BNB = []
chi2_KNC = []
chi2_SVC = []

mic_MNB = []
mic_BNB = []
mic_KNC = []
mic_SVC = []
def getScores(featureSize):
    for i in tests:
        scoresValidation[top][i] = {} 
        warnings.filterwarnings("ignore")
        for model in modelClassifiers:
            classfierName = ''
            if model == model_MNB:
                classfierName = 'MultinomialNB'
                if i == 'chi2':
                    train_data = training_tf_chi2
                else:
                    train_data = training_tf_mic
                test_data = test_tf
            elif model == model_BNB:
                classfierName = 'BernoulliNB'
                if i == 'chi2':
                    train_data = training_idf_chi2
                else:
                    train_data = training_idf_mic
                test_data = test_idf
            elif model == model_KNC:
                classfierName = 'KNeighborsClassifier'
                if i == 'chi2':
                    train_data = training_tfidf_chi2
                else:
                    train_data = training_tfidf_mic
                test_data = test_tfidf
            else:
                classfierName = 'SVC'
                if i == 'chi2':
                    train_data = training_tfidf_chi2
                else:
                    train_data = training_tfidf_mic
                test_data = test_tfidf
            scores = cross_val_score(model, train_data, test_data, cv=5, scoring='f1_macro')
            scoresValidation[top][i][classfierName] = scores.mean()

for top in topFeatures:
    scoresValidation[top] = {}
    training_tf_chi2 = SelectKBest(chi2, k=top).fit_transform(training_tf, test_tf)
    training_tf_mic = SelectKBest(mutual_info_classif, k=top).fit_transform(training_tf, test_tf)

    training_idf_chi2 = SelectKBest(chi2, k=top).fit_transform(training_idf, test_idf)
    training_idf_mic = SelectKBest(mutual_info_classif, k=top).fit_transform(training_idf, test_idf)

    training_tfidf_chi2 = SelectKBest(chi2, k=top).fit_transform(training_tfidf, test_tfidf)
    training_tfidf_mic = SelectKBest(mutual_info_classif, k=top).fit_transform(training_tfidf, test_tfidf)
    getScores(top)
    print(top)

for feature in scoresValidation.keys():
    for selection in scoresValidation[feature].keys():
        for model in scoresValidation[feature][selection].keys():
            value = scoresValidation[feature][selection][model]
            if selection == 'chi2':
                if model == 'MultinomialNB':
                    chi2_MNB.append(value)
                elif model == 'BernoulliNB':
                    chi2_BNB.append(value)
                elif model == 'KNeighborsClassifier':
                    chi2_KNC.append(value)
                elif model == 'SVC':
                    chi2_SVC.append(value)
                
            elif selection == 'mic':
                if model == 'MultinomialNB':
                    mic_MNB.append(value)
                elif model == 'BernoulliNB':
                    mic_BNB.append(value)
                elif model == 'KNeighborsClassifier':
                    mic_KNC.append(value)
                elif model == 'SVC':
                    mic_SVC.append(value)
print('Number of Features:  ' + str(topFeatures))
print('Chi2 Cross Validation Scores')
print('MultinomialNB:       ' + str(chi2_MNB))
print('BernoulliNB:         ' + str(chi2_BNB))
print('KNeighborsClassifier:' + str(chi2_KNC))
print('SVC:                 ' + str(chi2_SVC))
print()
print('Mutual Information Cross Validation Scores')
print('MultinomialNB:       ' + str(mic_MNB))
print('BernoulliNB:         ' + str(mic_BNB))
print('KNeighborsClassifier:' + str(mic_KNC))
print('SVC:                 ' + str(mic_SVC))

plt.xlabel('K feature Size')
plt.ylabel('Scores')

plt.subplot(2,1,1)
plt.title('Chi2 Cross Validation Scores')
plt.plot(topFeatures,chi2_MNB, label='MultinomialNB')
plt.plot(topFeatures,chi2_BNB, label='BernoulliNB')
plt.plot(topFeatures,chi2_KNC, label='KNeighbors')
plt.plot(topFeatures,chi2_SVC, label='SVC')
plt.legend(loc='best', shadow=True, fontsize=6)
plt.tight_layout()
plt.subplot(2,1,2)
plt.title('Mic Cross Validation Scores')
plt.plot(topFeatures,mic_MNB, label='MultinomialNB')
plt.plot(topFeatures,mic_BNB, label='BernoulliNB')
plt.plot(topFeatures,mic_KNC, label='KNeighbors')
plt.plot(topFeatures,mic_SVC, label='SVC')
plt.legend(loc='best', shadow=True, fontsize=6)
plt.tight_layout()

filename = 'chi2mic' + str(len(topFeatures))+'.png'
plt.savefig(filename)
print('Saved the plot in the' + filename)
elapsed_time_fl = (time.time() - start) 
print(elapsed_time_fl)



    