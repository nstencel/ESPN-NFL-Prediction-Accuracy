from flask import Flask, render_template, redirect, flash, url_for, request

import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords



import pandas as pd

import pickle

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'tepiglover'




@app.route("/", methods=["GET"])
def goHome():
    return render_template('home.html')

@app.route('/high_or_low')
def high_or_low():
    return render_template('high_or_low.html')

@app.route('/predict_high_or_low', methods=["POST"])
def predict_high_or_low():
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    model = pickle.load(open("multinomialNB_model_pkl", 'rb'))
    prediction = model.predict(loaded_vectorizer.transform([request.form['w3review']]))
    # model returns a numpy array. To get only the value of the prediction simply access the value of the first element
    return render_template("high_or_low.html", message=prediction[0])

@app.route('/svm_model')
def svm_model():
    return render_template('svm_model.html')

@app.route('/svm_high_or_low', methods=["POST"])
def svm_high_or_low():
    article = request.form['w3review']
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    article = article.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    article = word_tokenize(article)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV


    tokenized_article = "";
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(article):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    tokenized_article = str(Final_words)

    vectorizer = pickle.load(open('Tfidf_vect.pickle', 'rb'))
    model = pickle.load(open("svm_pkl", 'rb'))
    prediction = model.predict(vectorizer.transform([tokenized_article]))
    outcome = ""

    if prediction[0] == 0:
        outcome = "HIGH"
    else:
        outcome = "LOW"

    return render_template("svm_model.html", message=outcome)

@app.route('/game_day')
def game_day():
    #gameDay = pickle.load(open('gameDay_df.pickle', 'rb'))
    #return render_template("game_day.html", game=gameDay['Year'].iloc[0])
    return render_template("game_day.html")

@app.route('/season1')
def season1():
    gameDay = pickle.load(open('gameDay_df.pickle', 'rb'))
    return render_template("season1.html", game=gameDay[gameDay['Year'] == 2019])

@app.route('/season2')
def season2():
    gameDay = pickle.load(open('gameDay_df.pickle', 'rb'))
    return render_template("season2.html", game=gameDay[gameDay['Year'] == 2020])

@app.route('/season3')
def season3():
    gameDay = pickle.load(open('gameDay_df.pickle', 'rb'))
    return render_template("season3.html", game=gameDay[gameDay['Year'] == 2021])



if __name__ == "__main__":
    app.run(debug=True)

