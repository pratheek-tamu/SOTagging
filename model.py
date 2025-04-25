from nltk import text
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import joblib


#for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stop = set(stopwords.words('english'))
snow = SnowballStemmer('english')

#clean text of any uneeded characters and remove stopwords
def clean_data(text):
  text = str(text)
  text = text.lower()
  html = re.compile('<.*?>')
  text = re.sub(html, " ", text)
  text = re.sub("[^A-Za-z#+-]", " ", text)
  text = re.sub("[\s]+", " ", text)
  return text

def stem(text):
  tlist = text.split()
  oplist = []
  for word in tlist:
    if word not in stop:
      word = snow.stem(word)
      oplist.append(word)
  text = " ".join(oplist)
  return text

#load pickle files
mlb=joblib.load('tags.pkl')
text_classifier = joblib.load('pipeline_text_classifer.pkl')
title_classifier=joblib.load('pipeline_title_classifier.pkl')

# combine results of both text and title classfiers
def combine_results(prediction_text,prediction_title):
      combined_pred=list()
      for row in range(len(prediction_text)):
          row_pred=(prediction_text[row]+prediction_title[row])/2
          combined_pred.append(row_pred)

      result=list()
      for row in combined_pred:
        row_pred = np.where(row >= 0.5, 1, 0)
        result.append(list(row_pred))
      return result

if __name__ == '__main__' :
    #dummy data
    bbody = """'I'm trying to get a Tv show to display on multiple cards. I'm just using one Tv show before I start adding any more. So it should basically display one tv show on all the cards.
                The error is coming from the tvList.js.
                tvList.js'"""


    btitle = "TypeError: Cannot read properties of undefined (reading 'map') react"

    #preprocess body and title to send to model
    prebody = clean_data(bbody)
    prebody = stem(prebody)
    pretitle = clean_data(btitle)
    pretitle = stem(pretitle)

    #predict the tags based on title and body of the preprocssed parts
    prediction_text = text_classifier.predict([prebody])
    prediction_title=text_classifier.predict([pretitle])

    #combine results
    result=combine_results(prediction_text,prediction_title)

    #print the results
    print(mlb.inverse_transform(np.array(result)))

