"""
-*- coding: utf-8 -*-
To run from the directory 'PROJECT7_api'
$ python api_flask.py
""" 

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
# # (le module doit avoir le même nom que celui utilisé pour le pickle du modèle !!)
# from custtransformer import CustTransformer
from P7_functions import CustTransformer
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
# load training and test set from csv files
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

# SHAP values of the train set and test set
path = os.path.join('data', 'shap_val_X_tr_te.csv')
shap_val_X_tr_te = pd.read_csv(path, index_col=0)
# expected value
path = os.path.join('data', 'expected_val.pkl')
with open(path, 'rb') as file:
    expected_val = joblib.load(file)

###############################################################
# instantiate Flask object
app = Flask(__name__)

# view when API is launched
# Test local : http://127.0.0.1:5000
# Test : https://oc-api-flask-mm.herokuapp.com
@app.route("/")
def index():
    return "API loaded, models and data loaded, data computed…"

# answer when asking for sk_ids
# Test local: http://127.0.0.1:5000/api/sk_ids/
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/sk_ids/
@app.route('/api/sk_ids/')
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = pd.Series(list(X_test.index.sort_values()))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
    		        'data': sk_ids_json})

# return json object of feature description when needed
# Test local : http://127.0.0.1:5000/api/feat_desc
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/feat_desc
@app.route('/api/feat_desc/')
def send_feat_desc():
    # Convert pd.Series to JSON
    features_desc_json = json.loads(feat_desc.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': features_desc_json})
    
# return json object of feature importance (lgbm attribute)
# Test local : http://127.0.0.1:5000/api/feat_imp
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/feat_imp
@app.route('/api/feat_imp/')
def send_feat_imp():
    feat_imp = pd.Series(clf_step.feature_importances_,
                         index=X_te_featsel.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': feat_imp_json})

# return data of one customer when requested (SK_ID_CURR)
# Test local : http://127.0.0.1:5000/api/data_cust/?SK_ID_CURR=100128
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/data_cust/?SK_ID_CURR=100128
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

# find 20 nearest neighbors among the training set
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
# Test local : http://127.0.0.1:5000/api/neigh_cust/?SK_ID_CURR=100128
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/neigh_cust/?SK_ID_CURR=100128
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
# Test local : http://127.0.0.1:5000/api/all_proc_data_tr/
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/all_proc_data_tr/
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
# Test local : http://127.0.0.1:5000/api/scoring_cust/?SK_ID_CURR=100128
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/scoring_cust/?SK_ID_CURR=100128
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

#Importing the logit function for the base value transformation
from scipy.special import expit 
# Conversion of shap values from log odds to probabilities
def shap_transform_scale(shap_values, expected_value, model_prediction):
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    expected_value_transformed = expit(expected_value)
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = sum(shap_values)
    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = model_prediction - expected_value_transformed
    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient
    return shap_values_transformed, expected_value_transformed

@app.route('/api/shap_values/')
# get shap values of the customer and 20 nearest neighbors
# Test local : http://127.0.0.1:5000/api/shap_values/?SK_ID_CURR=100128
# Test Heroku : https://oc-api-flask-mm.herokuapp.com/api/shap_values/?SK_ID_CURR=100128
def shap_values():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh, y_neigh = get_df_neigh(sk_id_cust)
    shap_val_neigh =  shap_val_X_tr_te.loc[X_neigh.index]
    # Conversion of shap values from log odds to probabilities of the customer's shap values
    shap_t, exp_t = shap_transform_scale(shap_val_X_tr_te.loc[sk_id_cust],
                                         expected_val,
                                         clf_step.predict_proba(X_neigh)[:,1][-1])
    shap_val_cust_trans = pd.Series(shap_t,
                                    index=X_neigh.columns)
    # Converting the pd.Series to JSON
    X_neigh__json = json.loads(X_neigh.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh.to_json())
    shap_val_cust_trans_json = json.loads(shap_val_cust_trans.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'shap_val': shap_val_neigh_json, # pd.DataFrame
                    'shap_val_cust_trans': shap_val_cust_trans_json, # pd.Series
                    'exp_val': expected_val,
                    'exp_val_trans': exp_t,
                    'X_neigh_': X_neigh__json})

####################################
# if the api is run and not imported as a module
if __name__ == "__main__":
    app.run(debug=True)