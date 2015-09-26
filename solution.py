from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import dateutil.parser as dateparser
import pickle
import os

def get_num( s ):
	x = ''.join( e for e in s if e.isdigit() )
	if x == '':
		return np.nan
	return int( x )

def get_timestamp( s ):
	if s == 'nan':
		return np.nan
	dt = dateparser.parse( s )
	epoch = datetime(1970,1,1)
	diff = dt - epoch
	# return int( time.mktime(dt.timetuple()) )
	return int( diff.total_seconds() )

def get_bool( s ):
	if s == 'nan':
		return np.nan
	if s == 'N':
		return 0
	if s == 'Y':
		return 1

def preprocess( data ):
	data.set_index( 'ISIN', drop=True, inplace='True' )
	data = clean( data )
	data = impute( data, 'mean' )
	return data

def clean( data ):
	fields_to_prune = [
		'SP_Rating',
		'Moody_Rating',
		'Currency',
		'Seniority',
		'Collateral_Type',
		'Coupon_Frequency',
		'Coupon_Type',
		'Industry_Group',
		'Industry_Sector',
		'Industry_SubGroup',
		'Issuer_Name',
		'Ticker',
		'Country_Of_Domicile',
		'Risk_Stripe'
	]

	for f in fields_to_prune:
		if f in data:
			data[f] = data[f].map(str).map(get_num)

	date_fields = [
		'Issue_Date',
		'Maturity_Date'
	]

	for f in date_fields:
		data[f] = data[f].map(str).map(get_timestamp)

	bool_fields = [
		'Is_Emerging_Market',
		'Callable'
	]

	for f in bool_fields:
		data[f] = data[f].map(str).map(get_bool)

	return data

def impute( data, mode = 'mean' ):
	if mode == 'delete':
		data = delete_missing( data )
		
	if mode == 'mean':
		if os.path.exists( "objects/imputer.p" ):
			imp = pickle.load( open( "objects/imputer.p", 'r' ) )
		else:
			imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
			imp = imp.fit( data )
			pickle.dump( imp, open( "objects/imputer.p", "wb" ) ) # save imputer object

		values = imp.transform( data )
		data = pd.DataFrame( values, index=data.index, columns=data.columns )
		for c in data:
			data[c] = data[c].map(int)

	if mode == 'approximate':
		data = approximate_missing( data )

	return data 

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

def main():
	training_data = pd.read_csv( 'data/Initial_Training_Data.csv' )
	training_data = preprocess( training_data )
	pickle.dump( training_data, open( "objects/clean_training_data.p", "wb" ) )
	test_data = pd.read_csv( 'data/Initial_Test_Data.csv' )
	test_data = preprocess( test_data )
	pickle.dump( test_data, open( "objects/clean_test_data.p", "wb" ) )

if __name__ == '__main__':
	main()