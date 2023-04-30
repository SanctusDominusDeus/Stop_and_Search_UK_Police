
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
        #Xdata["Latitude"] = Xdata.groupby("station").transform(lambda x: x.fillna(x.mean()))
        #Xdata["Longitude"] = Xdata.groupby("station").transform(lambda x: x.fillna(x.mean()))
        Xdata["Latitude"] = Xdata.groupby("station")["Latitude"].apply(lambda x: x.fillna(x.mean()))
        Xdata["Longitude"] = Xdata.groupby("station")["Longitude"].apply(lambda x: x.fillna(x.mean()))
        
        #to avoid overfitting to these features and discrimination
        Xdata = Xdata.drop(['Officer-defined ethnicity','station'], axis=1)        
        
        return Xdata
