import os
import sys
import re
import json
import time
import math
import nltk
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English


class InvertedDocs:
  def __init__(self, classLabel, documentId, data):
    self.classLabel = classLabelling[classLabel]
    self.documentId = documentId
    self.data = data
    self.size = len(data.split(' '))
    
def tokenizer(text):
    text = re.sub("[^a-zA-Z]+", " ", text)
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

def preprocessing_txt(text):
    tokens = tokenizer(text)
    stemmer = EnglishStemmer()
    processedText = ""
    stopWords = nltk.corpus.stopwords.words('english')
    for token in tokens:
        token = token.lower()
        if token not in stopWords:
            processedText += stemmer.stem(token)
            processedText += " "
    return processedText

def test_preprocessing_txt(text):
    tokens = tokenizer(text)
    stemmer = EnglishStemmer()
    processedText = ""
    stopWordsList = []
    stopWords = nltk.corpus.stopwords.words('english')
    for token in tokens:
        token = token.lower()
        if token not in stopWords:
            processedText += stemmer.stem(token)
            processedText += " "
        else:
            stopWordsList.append(token)
    print('Removed Stopwords are: ' + str(stopWordsList))
    return processedText

# To find the normalized term location in the processed query
def wordPositions(word, processedText):
    pos = 1
    positions = []
    for data in processedText.split():
        if word == data:
            positions.append(pos)
        pos += 1
    return positions

def indexDoc(document):
      processedText = document['data']
      for word in processedText.split():
         position = wordPositions(word, processedText)
         if word in features.keys(): 
               if not int(document['documentId']) in list(features[word].keys()):
                  features[word][int(document['documentId'])] = position
         else:
               features[word] = {}
               features[word][int(document['documentId'])] = position

# Storing Class labels for the directories in the Dataset
def storeClassLabels():
   for dir in directory:
      labelClass.append(str(classLabelling[dir]) + ' ' + dir)
   with open(classFileName, 'w') as data:
      for item in labelClass:
         data.write("%s\n" % item)
   print('class_defination_file.TF file is generated')

# Finding and storing the inverted Index for the above all the datasets
def featureProcessing():
   print('Feature Id processing Started')
   for doc in document_JSON:
      indexDoc(doc)
   with open('inverted_index_file','w') as invertedIndex:
      json.dump(features, invertedIndex)

# Storing the (FeatureId,Term) Pair in feature defination file
def featureFileCreation():
   count = 0
   featureIdTerm = ''
   print('Feature Id and Term pair processing started')
   for term in features.keys():
      count += 1
      featureIdTerm += str(count) + ' ' + str(term) + '\n'
   with open(featureDefinationFile,'w') as featureFile:
         featureFile.write(featureIdTerm)
   print('feature_definition_file file is generated')

# Finding TF, IDF, and TF-IDF values
def createlibsvmFiles_TF_IDF_TFIDF():
   featureKeys = list(features.keys()) #===== Use featureKeys.index('term') to get the featureId
   totalDocs = len(document_JSON)
   tf_classLabel = ''
   idf_classLabel = ''
   tf_idf_classLabel = ''
   training_tf_classLabel = ''
   training_idf_classLabel = ''
   training_tf_idf_classLabel = ''
   print('Calculating and Storing of TF, IDF and TFIDF for each term has started')
   for doc in document_JSON:
      totalData = doc['size']
      tf_classLabel = ''
      idf_classLabel = ''
      tf_idf_classLabel = ''
      orderFreq = {}
      for text in doc['data'].split(' '):
         if len(text) > 0:
            freq = len(features[text][int(doc['documentId'])])
            featureId = ' ' + str(featureKeys.index(text)  + 1) + ':'
            tf = float(freq/totalData)
            idf = float( 1 + math.log10(totalDocs/len(features[text].keys())))
            tfidf = float (tf * idf)
            orderFreq[featureKeys.index(text)] = featureId + ',' + str(freq) + ',' + str(idf) + ',' + str(tfidf)

      for id in sorted(orderFreq.keys()):
         dataFreq = orderFreq[id].split(',')
         tf_classLabel += dataFreq[0] + dataFreq[1]
         idf_classLabel += dataFreq[0] + dataFreq[2]
         tf_idf_classLabel += dataFreq[0] + dataFreq[3]
      training_tf_classLabel += str(doc['classLabel']) + str(tf_classLabel) + '\n'
      with open(tfFileName,'w') as tfdoc:
         tfdoc.write(training_tf_classLabel)

      training_idf_classLabel += str(doc['classLabel']) + str(idf_classLabel) + '\n'
      with open(idfFileName,'w') as idfdoc:
         idfdoc.write(training_idf_classLabel)
      
      training_tf_idf_classLabel += str(doc['classLabel']) + str(tf_idf_classLabel) + '\n'
      with open(tfidfFileName,'w') as tfidfdoc:
         tfidfdoc.write(training_tf_idf_classLabel)

   print('training_data_file.TF file is generated')
   print('training_data_file.IDF file is generated')
   print('training_data_file.TFIDF file is generated')


def extractData():
   insertLines = False
   for dir in os.listdir(dataSetLocation):
      directory.append(dir)
      newdir = os.path.join(dataSetLocation, dir)
      for filename in os.listdir(newdir):
         with open(os.path.join(newdir, filename), 'r') as f:
            data = ''
            for line in f:
               if 'Subject:' in line:
                  data = data + line.replace('Subject: ','').replace('Re: ', '').replace('\n', ' ')
               elif 'Lines: ' in line:
                  insertLines = True
               elif '--' in line:
                  insertLines = False
                  break
               elif insertLines and (len(line) > 1):
                  data = data + line.split('\n')[0] + ' '
            data = preprocessing_txt(data)
            if len(data) > 0:
               documentList.append(InvertedDocs(dir, filename, data))
      print('Number of Directories processed for feature Selection are  ' + str(len(directory)))

def test():
   test_tf_data = ''
   test_idf_data = ''
   test_tfidf_data = ''
   manual_indexing = {"game": {"1": [1, 3]}, "life": {"1": [2], "2": [2]}, "everlast": {"1": [4]}, "learn": {"1": [5], "3": [3]}, "unexamin": {"2": [1]}, "worth": {"2": [3]}, "live": {"2": [4]}, "never": {"3": [1]}, "stop": {"3": [2]}}
   manual_tfidf = '1 1:2.0 2:0.6989700043360187 3:2.0 4:1.0 5:0.6989700043360187 \n2 1:1.0 2:0.6989700043360187 3:1.0 4:1.0 \n3 1:1.0 2:1.0 3:0.6989700043360187 \n'
   print()
   print('===== Working on Test results =====')
   for testfileName in os.listdir('test_data'):
      testinsertLines = False
      with open(os.path.join('test_data',testfileName), 'r') as f:
            test_data = ''
            for line in f:
               if 'Subject:' in line:
                  test_data = test_data + line.replace('Subject: ','').replace('Re: ', '').replace('\n', ' ')
               elif 'Lines: ' in line:
                  testinsertLines = True
               elif '--' in line:
                  testinsertLines = False
                  break
               elif testinsertLines and (len(line) > 1):
                  test_data = test_data + line.split('\n')[0] + ' '
            
            print('Data extracted from DocId ' + str(testfileName) + ' considering Subject and Lines in test_data directory is')
            print('Data extracted before processing is --- ' + str(test_data))
            test_data = test_preprocessing_txt(test_data)
            print('Data extracted after processing is --- ' + str(test_data))

      test_tf = ''
      test_idf = ''
      test_tfidf = ''
      docId = str(testfileName)
      # count = 0
      for term in test_data.split(' '):
         if term in manual_indexing.keys():
            count = list(manual_indexing.keys()).index(term)
            freq = len(manual_indexing[term][docId])
            tf = float(freq)
            idf = float( 1 + math.log10(1/len(manual_indexing[term].keys())))
            tfidf = float (tf * idf)
            test_tf += str(count) + ':' + str(tf) + ' '
            test_idf += str(count) + ':' + str(idf) + ' '
            test_tfidf += str(count) + ':' + str(tfidf) + ' '
      test_tf_data += docId + ' ' + test_tf + '\n'
      test_idf_data += docId + ' ' + test_idf + '\n'
      test_tfidf_data += docId + ' ' + test_tfidf + '\n'
   with open('test_tf','w') as tfdoc:
         tfdoc.write(test_tf_data)
   
   with open('test_idf','w') as idfdoc:
         idfdoc.write(test_idf_data)

   with open('test_tfidf','w') as tfidfdoc:
         tfidfdoc.write(test_tfidf_data)    

   print('Test TF File saved successfully')
   print('Test IDF File saved successfully')
   print('Test TFIDF File saved successfully')

   print('Loaded Data from test_tf file in libsvm format is:')
   print(test_tf_data)
   print('Loaded Data from test_idf file in libsvm format is:')
   print(test_idf_data)
   print('Loaded Data from test_tfidf file in libsvm format is:')
   print(test_tfidf_data)

   if (test_tfidf_data == manual_tfidf):
      print('Data Loaded successfully')

   
   
if __name__ == '__main__':
   start = time.time()
   # python feature-extract.py mini_newsgroups new_feature_definition_file new_class_definition_file new_training_data_file
   dataSetLocation = str(sys.argv[1])
   classFileName = str(sys.argv[3])
   fileName = str(sys.argv[4])
   tfFileName = fileName + '.TF'
   idfFileName = fileName + '.IDF'
   tfidfFileName = fileName + '.TFIDF'
   featureDefinationFile = str(sys.argv[2])
   features = {}
   labelClass = []
   classLabelling = {
      'alt.atheism':5,
      'soc.religion.christian':5,
      'talk.religion.misc':5,
      'comp.graphics':0,
      'comp.os.ms-windows.misc':0,
      'comp.sys.ibm.pc.hardware':0,
      'comp.sys.mac.hardware':0,
      'comp.windows.x':0,
      'misc.forsale':3,
      'rec.autos':1,
      'rec.motorcycles':1,
      'rec.sport.baseball':1,
      'rec.sport.hockey':1,
      'sci.crypt':2,
      'sci.electronics':2,
      'sci.med':2,
      'sci.space':2,
      'talk.politics.guns':4,
      'talk.politics.mideast':4,
      'talk.politics.misc':4,
   }
   i = 0
   documentList = []
   directory = []
   tfList = []

   extractData()
   # Storing Documents with its data, classlabel and its id 
   document_JSON = eval(json.dumps([ob.__dict__ for ob in documentList]))
   with open('document_file','w') as doc:
         json.dump(document_JSON, doc)
   storeClassLabels()
   featureProcessing()
   featureFileCreation()
   createlibsvmFiles_TF_IDF_TFIDF()

   elapsed_time_fl = (time.time() - start) 
   print('Calculating and Storing of TF, IDF and TFIDF for each term in respective files took ' + str(elapsed_time_fl) + 'seconds')
   test()
