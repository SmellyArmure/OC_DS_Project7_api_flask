''' 
-*- coding: utf-8 -*-
To run from the directory 'WEB':
python api_flask.py # api/
'''

# Load librairies
import os
import sys
import joblib
import dill
import pandas as pd
import sklearn
from flask import Flask, jsonify, request
import json
from sklearn.neighbors import NearestNeighbors
import shap

from P7_functions import CustTransformer # (le module doit avoir le même nom que celui utilisé pour le pickle du modèle !!)
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

#########################################################
# Loading data and model (all the files are in WEB/data)
#--------------------------------------------------------
# description of each feature
path = os.path.join('data', 'feat_desc.csv')
feat_desc = pd.read_csv(path, index_col=0)

#--------------------------------------------------------
# best model (pipeline)
path = os.path.join('model', 'bestmodel_joblib.pkl')
with open(path, 'rb') as file:
    bestmodel = joblib.load(file)
path = os.path.join('model', 'threshold.pkl')
with open(path, 'rb') as file:
    thresh = joblib.load(file)

# Split the steps of the best pipeline
preproc_step = bestmodel.named_steps['preproc']
featsel_step = bestmodel.named_steps['featsel']
clf_step = bestmodel.named_steps['clf']

#--------------------------------------------------------
# # load training and test set from csv files
# X_train = dict_cleaned['X_train']
# y_train = dict_cleaned['y_train']
# X_test = dict_cleaned['X_test']
path = os.path.join('data', 'X_train.csv')
X_train = pd.read_csv(path, index_col='SK_ID_CURR')
path = os.path.join('data', 'y_train.csv')
y_train = pd.read_csv(path, index_col='SK_ID_CURR')
path = os.path.join('data', 'X_test.csv')
X_test = pd.read_csv(path, index_col='SK_ID_CURR')

# compute the preprocessed data (encoding and standardization)
X_tr_prepro = preproc_step.transform(X_train)
X_te_prepro = preproc_step.transform(X_test)
# get the name of the columns after encoding
preproc_cols = X_tr_prepro.columns
# get the name of the columns selected using SelectFromModel
featsel_cols = preproc_cols[featsel_step.get_support()]
# compute the data to be used by the best classifier
X_tr_featsel = X_tr_prepro[featsel_cols]
X_te_featsel = X_te_prepro[featsel_cols]


# # refit the model on X_train (avoid pbes with importance getter ?)
# clf_step.fit(X_tr_featsel, y_train['TARGET']);

###############################################################
# instantiate Flask object
app = Flask(__name__)

# view when API is launched
# Test : http://127.0.0.1:5000
@app.route("/")
def index():
    return "API loaded, models and data loaded, data computed…"

# return json object of feature description when needed
# Test : http://127.0.0.1:5000/api/feat_desc
@app.route('/api/feat_desc/')
def send_feat_desc():
    # Convert pd.Series to JSON
    features_desc_json = json.loads(feat_desc.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': features_desc_json})

# answer when asking for sk_ids
#  Test : http://127.0.0.1:5000/api/sk_ids/
@app.route('/api/sk_ids/')
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = pd.Series(list(X_test.index.sort_values()))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
    		        'data': sk_ids_json})

# return data of one customer when requested (SK_ID_CURR)
# Test : http://127.0.0.1:5000/api/data_cust/?SK_ID_CURR=100128
@app.route('/api/data_cust/')
def data_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the personal data for the customer (pd.Series)
    X_cust_ser = X_test.loc[sk_id_cust, :]
    X_cust_proc_ser = X_te_featsel.loc[sk_id_cust, :]
    # Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust_ser.to_json())
    X_cust_proc_json = json.loads(X_cust_proc_ser.to_json())
    # Return the cleaned data
    return jsonify({'status': 'ok',
    				'data': X_cust_json,
    				'data_proc': X_cust_proc_json})

# find 20 nearrest neighbors among the training set
def get_df_neigh(sk_id_cust):
    # get data of 20 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(X_tr_featsel)
    X_cust = X_te_featsel.loc[sk_id_cust: sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=20,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_tr_featsel.iloc[idx].index)
    X_neigh_df = X_tr_featsel.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    return X_neigh_df, y_neigh

# return data of 20 neighbors of one customer when requested (SK_ID_CURR)
# Test : http://127.0.0.1:5000/api/neigh_cust/?SK_ID_CURR=100128
@app.route('/api/neigh_cust/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh_df, y_neigh = get_df_neigh(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json})

# return all data of training set when requested
# Test : http://127.0.0.1:5000/api/all_proc_data_tr/
@app.route('/api/all_proc_data_tr/')
def all_proc_data_tr():
    # get all data from X_tr_featsel, X_te_featsel and y_train data
    # and convert the data to JSON
    X_tr_featsel_json = json.loads(X_tr_featsel.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_tr_proc': X_tr_featsel_json,
    				'y_train': y_train_json})

# answer when asking for score and decision about one customer
# Test : http://127.0.0.1:5000/api/scoring_cust/?SK_ID_CURR=100128
@app.route('/api/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = bestmodel.predict_proba(X_cust)[:,1][0]
    # Return score
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
                    'thresh': thresh})

@app.route('/api/shap_values/')
# get shap values of the customer and 20 nearest neighbors
# Test : http://127.0.0.1:5000/api/shap_values/?SK_ID_CURR=100128
def shap_values():
    # refit the classifier to avoid 'objective' value error in shap...
    clf_step.fit(X_tr_featsel, y_train['TARGET'])
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # create the shap tree explainer of our classifier
    explainer = shap.TreeExplainer(clf_step)
    # return the nearest neighbors
    X_neigh, y_neigh = get_df_neigh(sk_id_cust)
    # return data of the customer
    X_cust = X_te_featsel.loc[sk_id_cust:sk_id_cust]
    # compute the SHAP values of the 20 neighbors + customer for the model
    X_neigh_ = pd.concat([X_neigh, X_cust], axis=0)
    # shap values pour X train and test
    shap_val_all = pd.DataFrame(explainer.shap_values(X_neigh_)[1],
                                index=X_neigh_.index,
                                columns=X_neigh_.columns)
    # compute expected value (approx mean of predictions on training set)
    # NB to be calculated AFTER shap values !!!!!!!!!!!!!!
    expected_value = explainer.expected_value[1] # depends on the model only (already fitted on training set)
    # Converting the pd.Series to JSON
    shap_val_all_json = json.loads(shap_val_all.to_json())
    X_neigh__json = json.loads(X_neigh_.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'shap_val': shap_val_all_json,
                    'exp_val': expected_value,
                    'X_neigh_': X_neigh__json})


####################################
# if the api is run and not imported as a module
if __name__ == "__main__":
    app.run(debug=True)