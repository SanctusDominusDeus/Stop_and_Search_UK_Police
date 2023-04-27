from utils import *
import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError, BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from uuid import uuid4


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd



class Date(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
  
        return self

    def transform(self, X, y=None):
        Xdata = X.copy()
        Xdata['Date'] = pd.to_datetime(Xdata['Date'], infer_datetime_format=True)
        Xdata['Month'] = Xdata.Date.dt.month
        Xdata['Week_day'] = Xdata.Date.dt.weekday
        Xdata['Hour'] = Xdata.Date.dt.hour
        Xdata['Night'] = (Xdata['Hour'] >= 19) | (Xdata['Hour'] <= 7)
        Xdata = Xdata.drop(['Date'], axis=1)
        return Xdata
#custom transformer for the latitude and longitude fillna
class LatLong_fillna(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
  
        return self

    def transform(self, X, y=None):
        Xdata = X.copy()
        Xdata["Latitude"] = Xdata.groupby("station").transform(lambda x: x.fillna(x.mean()))
        Xdata["Longitude"] = Xdata.groupby("station").transform(lambda x: x.fillna(x.mean()))
        
        return Xdata

########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation_data = TextField()
    predicted_outcome = BooleanField()
    actual_outcome = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################



########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/should_search/', methods=['POST'])
def predict():
    
    observation = request.get_json()
    
    #implement the mapping for the valid values
    valid_category_map = {
                "observation_id":str ,
                "Type":['Person search','Person and Vehicle search','Vehicle search'],
                'Date':str,
                'Part of a policing operation':[True, False],
                'Latitude':float,#try to add range for the uk
                'Longitude':float,#try to add range for the uk
                'Gender': ['Male','Female','Other'],
                'Age range':['under 10','10-17','18-24','25-34','over 34'],
                'Officer-defined ethnicity':['White','Asian','Black','Other','Mixed'],
                'Legislation':str,
                'Object of search':str,#should it be a list???
                'station':str

    }

    # the followig is to verify the range
    _id = None
    for key, valid_categories in valid_category_map.items():
        if key in observation.keys():
            _id = observation['observation_id']
            value = observation[key]
            #checking only format
            if type(valid_categories)!=list:
                if type(value)==valid_categories:
                    continue
                else:
                    error = "Invalid format provided for {}. Please provide a {}".format(key,valid_categories)
                    return jsonify({"observation_id":_id,"error":error})

            #checking format and values
            elif value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return jsonify({"observation_id":_id,"error":error})
        else:
            error = '{} is not provided.'.format(key)
            return jsonify({"observation_id":_id,"error":error})
    
    #not run with more than expected inputs
    for key in observation.keys():
        if key not in valid_category_map.keys():
            error = '{} is not a valid input.'.format(key)
            return jsonify({"observation_id":_id,"error":error})


    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    prediction = pipeline.predict(obs)[0]
    #proba = pipeline.predict_proba(obs)[0,1]

    response = dict()
    response['outcome'] = bool(prediction)
    p = Prediction(
        observation_id=_id,
        observation_data=observation,
        predicted_outcome = prediction
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Admission ID: {} already exists'.format(_id)
        response['error'] = error_msg
        DB.rollback()
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()

    # the followig is to verify the range => it is stupid to repeat this code but I cannot implemented inside a function that works for both cases
    #implement the mapping for the valid values
    validinput_category_map = {
                "observation_id":str ,
                "outcome":[True, False]
    }
    _id = None
    for key, valid_categories in validinput_category_map.items():
        if key in obs.keys():
            _id = obs['observation_id']
            value = obs[key]
            #checking only format
            if type(valid_categories)!=list:
                if type(value)==valid_categories:
                    continue
                else:
                    error = "Invalid format provided for {}. Please provide a {}".format(key,valid_categories)
                    return jsonify({"observation_id":_id,"error":error})

            #checking format and values
            elif value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return jsonify({"observation_id":_id,"error":error})
        else:
            error = '{} is not provided.'.format(key)
            return jsonify({"observation_id":_id,"error":error})
    
    #not run with more than expected inputs
    for key in obs.keys():
        if key not in validinput_category_map.keys():
            error = '{} is not a valid input.'.format(key)
            return jsonify({"observation_id":_id,"error":error})

    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.actual_outcome = obs['outcome']
        p.save()
        ret_dict = model_to_dict(p)
        k = {'observation_id':ret_dict['observation_id'],'outcome':ret_dict['actual_outcome'],'predicted_outcome':ret_dict['predicted_outcome']}
        return jsonify(k)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
