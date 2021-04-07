# OC_DS_Project7_api_flask

Pour interagir avec l'api (tester les requêtes) :

# view when API is launched (index)
https://oc-api-flask-mm.herokuapp.com

# answer when asking for sk_ids
https://oc-api-flask-mm.herokuapp.com/api/sk_ids/

# return json object of feature description when needed
https://oc-api-flask-mm.herokuapp.com/api/feat_desc
    
# return json object of feature importance (lgbm attribute)
https://oc-api-flask-mm.herokuapp.com/api/feat_imp

# return data of one customer when requested (SK_ID_CURR)
https://oc-api-flask-mm.herokuapp.com/api/data_cust/?SK_ID_CURR=100128

# return data of 20 neighbors of one customer when requested (SK_ID_CURR)
https://oc-api-flask-mm.herokuapp.com/api/neigh_cust/?SK_ID_CURR=100128

# return all data of training set when requested
https://oc-api-flask-mm.herokuapp.com/api/all_proc_data_tr/

# answer when asking for score and decision about one customer
https://oc-api-flask-mm.herokuapp.com/api/scoring_cust/?SK_ID_CURR=100128

# get shap values of the customer and 20 nearest neighbors
https://oc-api-flask-mm.herokuapp.com/api/shap_values/?SK_ID_CURR=100128
