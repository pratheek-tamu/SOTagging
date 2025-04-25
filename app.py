from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import joblib

# load the model from disk - 
text_classifier = joblib.load('pipeline_text_classifer.pkl')
title_classifier=joblib.load('pipeline_title_classifier.pkl')
mlb=joblib.load('tags.pkl')
app = Flask(__name__)

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

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		form_title = request.form['title']
		form_body = request.form['question']
		title = [form_title]
		body = [form_body]

		cleaned_title=clean_data(title)
		cleaned_text=clean_data(body)

		stemmed_text=stem(cleaned_title)
		stemmed_title=stem(cleaned_text)

		title_prediction = title_classifier.predict([stemmed_title])
		text_prediction=text_classifier.predict([stemmed_text])

		result=combine_results(text_prediction,title_prediction)

		predicted_result = mlb.inverse_transform(np.array(result))
		pred = list(predicted_result[0])
		return render_template('result.html', prediction = pred, tag_len = len(pred) )

if __name__ == '__main__':
	app.run(debug=True)