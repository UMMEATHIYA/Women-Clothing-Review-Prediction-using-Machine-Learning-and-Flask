from flask import Flask, render_template,request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
import re
nltk.download('wordnet')

app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def index():
    return render_template("index_reviews.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    df= pd.read_csv("clothing.csv", encoding="latin-1")
    df['review'] = df['review'].fillna('')
    
    def clean_and_tokenize(review):
        text = review.lower()
    
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokens = tokenizer.tokenize(text)
    
        stemmer = nltk.stem.WordNetLemmatizer()
        text = " ".join(stemmer.lemmatize(token) for token in tokens)
        text = re.sub("[^a-z']"," ", text)
        return text
    df["Clean_Review"] = df["review"].apply(clean_and_tokenize)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['recommend']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5320)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    lr_pred=lr.predict(X_test)

        
    if request.method == 'POST':
       # message = request.form['message']
        text=request.form.get('text')
        data = [text]
        vect = vectorizer.transform(data).toarray()
        my_prediction = lr.predict(vect)
    return render_template('prediction_reviews.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run()