# Code Snippet taken from : https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
import flask
import random
from flask import request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib




import os
import subprocess
from glob import glob

from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
    DistutilsError
from distutils.file_util import copy_file

from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
    filter_sources, get_ext_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt






app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/',methods = ['GET])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


@app.route('/classify', methods=['GET'])
def classify():
    my_prediction = "NOT"
    df = pd.read_csv("spamham.csv")
    df.drop_duplicates(inplace=True)
    X = df['text']
    y = df['spam']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    #Alternative Usage of Saved Model
    #joblib.dump(clf, 'NB_spam_model.pkl')
    #NB_spam_model = open('NB_spam_model.pkl','rb')
    #clf = joblib.load(NB_spam_model)

    if request.method == 'GET':
        message = request.args['msg']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return jsonify({'spam': int(my_prediction[0])})


if __name__ == '__main__':           
    app.run()
