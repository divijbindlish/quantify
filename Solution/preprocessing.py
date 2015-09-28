from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import dateutil.parser as dateparser
import pickle
import os

numerical_fields = [
	'SP_rating',
	'Moody_rating',
	'Seniority',
	'Days_to_Settle',
	'Coupon_Frequency',
	'Ticker'
]

categorical_fields = [
	'Currency',
	'Collateral_Type',
	'Coupon_Type',
	'Industry_Group',
	'Industry_Sector',
	'Industry_SubGroup',
	'Issuer_Name',
	'Country_Of_Domicile',
	'Risk_Stripe'
]

bool_fields = [
	'Is_Emerging_Market',
	'Callable'
]

date_fields = [
	'Issue_Date',
	'Maturity_Date'
]

def get_num( s ):
	x = ''.join( e for e in s if e.isdigit() )
	if x == '':
		return np.nan
	return int(x)

def get_timestamp( s ):
	if s == 'nan':
		return np.nan
	dt = dateparser.parse( s )
	epoch = datetime(1970,1,1)
	diff = dt - epoch
	return int( diff.total_seconds() )

def get_bool( s ):
	if s == 'nan':
		return np.nan
	if s == 'N':
		return 0
	if s == 'Y':
		return 1

def preprocess( training_data, test_data ):
	training_data = clean( training_data )
	test_data = clean( test_data )
	target_data = training_data['Risk_Stripe']
	training_data = training_data.drop( 'Risk_Stripe', axis=1 )
	# print training_data
	# print test_data
	training_data, test_data = impute( training_data, test_data, 'mean' )
	# training_data, test_data = tokenize( training_data, test_data )
	training_data, test_data = normalize( training_data, test_data )
	return (training_data, target_data, test_data)

def clean( data ):
	data.set_index( 'ISIN', drop=True, inplace='True' )

	for f in numerical_fields:
		if f in data:
			data[f] = data[f].map(str).map(get_num)

	for f in categorical_fields:
		if f in data:
			data[f] = data[f].map(str).map(get_num)

	for f in date_fields:
		data[f] = data[f].map(str).map(get_timestamp)

	for f in bool_fields:
		data[f] = data[f].map(str).map(get_bool)

	return data

def impute( training_data, test_data, mode = 'mean' ):
	if mode == 'delete':
		data = delete_missing( data )
		
	if mode == 'mean':
		imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
		imp = imp.fit( training_data )
		values = imp.transform( training_data )
		training_data = pd.DataFrame( values, index=training_data.index, columns=training_data.columns )
		values = imp.transform( test_data )
		test_data = pd.DataFrame( values, index=test_data.index, columns=test_data.columns )
		for c in training_data:
			training_data[c] = training_data[c].map(int)
		for c in test_data:
			test_data[c] = test_data[c].map(int)

	if mode == 'approximate':
		data = approximate_missing( data )

	return (training_data, test_data)

def delete_missing_values( data ):
	return data	

def approximate_missing_values( data ):
	complete_cols = []
	incomplete_cols = []
	for col in data:
		if len( data[col] ) == len( data[col][data[col].notnull()] ):
			complete_cols.append( col )
		else:
			incomplete_cols.append( col )

	print complete_cols
	print incomplete_cols

	if 'Risk_Stripe' in complete_cols:
		complete_cols.remove( 'Risk_Stripe' )

	for col in incomplete_cols:
		complete_cols.append( col )
		train = data[complete_cols][data[col].notnull()]
		complete_cols.remove( col )
		test = data[complete_cols][data[col].isnull()]
		# print col, len(train.columns), len(test.columns)

	return data

def tokenize( training_data, test_data ):
	# print training_data.shape
	if 'Risk_Stripe' in categorical_fields:
		categorical_fields.remove( 'Risk_Stripe' )
	
	
	for c in categorical_fields:
		training_data[c] = training_data[c].map(str)
		test_data[c] = test_data[c].map(str)

	cat_data = training_data[categorical_fields]
	ts_cat_data = test_data[categorical_fields]
	# print cat_data.shape
	vec = DictVectorizer()
	tr_cat_data_dict = cat_data.T.to_dict().values()
	ts_cat_data_dict = ts_cat_data.T.to_dict().values()
	tr_cat_data_array = vec.fit_transform( tr_cat_data_dict ).toarray()
	ts_cat_data_array = vec.transform( ts_cat_data_dict ).toarray()
	# print tr_cat_data_array.shape
	# print ts_cat_data_array.shape
	non_cat_data = training_data.drop( categorical_fields, axis=1 )
	non_cat_data = np.array( non_cat_data ).astype(np.float)
	new_tr_data = np.concatenate( (tr_cat_data_array, non_cat_data), axis=1 )
	# print new_tr_data.shape
	non_cat_data = test_data.drop( categorical_fields, axis=1 )
	non_cat_data = np.array( non_cat_data ).astype(np.float)
	new_ts_data = np.concatenate( (ts_cat_data_array, non_cat_data), axis=1 )
	# print new_ts_data.shape
	new_tr_data = pd.DataFrame( new_tr_data, index=training_data.index )
	new_ts_data = pd.DataFrame( new_ts_data, index=test_data.index )
	return new_tr_data, new_ts_data

def normalize( training_data, test_data ):
	scaler = StandardScaler()
	values = scaler.fit_transform( training_data )
	training_data = pd.DataFrame( values, columns=training_data.columns, index=training_data.index )
	values = scaler.transform( test_data )
	test_data = pd.DataFrame( values, columns=test_data.columns, index=test_data.index )
	return training_data, test_data 

def prepare_data():	
	training_data = pd.read_csv( 'data/Final_Training_Data.csv' )
	test_data = pd.read_csv( 'data/Final_Test_Data.csv' )

	training_data, target_data, test_data = preprocess( training_data, test_data )
	
	pickle.dump( training_data, open( "objects/clean_training_data.p", "wb" ) )
	pickle.dump( target_data, open( "objects/clean_target_data.p", "wb" ) )
	pickle.dump( test_data, open( "objects/clean_test_data.p", "wb" ) )

if __name__ == '__main__':
	prepare_data()