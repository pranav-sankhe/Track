from __future__ import division 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#--------------------------------------Global variables-------------------------------------------------
file_names = ['pos1','31-05_1','31-05_2', '31-05_3', '31-05_4']
base_path = '/home/pranav/project_repositories/lps/Software/data_analysis/rssi/datasets/'
router_pos = [0,137.6]


#--------------------------------------Functions-------------------------------------------------------
def data_cleaning(dataframe):
	# for i in range(len(dataframe)):

	# 	row = dataframe.loc[i:i]
	print "data cleaning initiated"
	
	print "checking values of x and y coordinates"
	dataframe = dataframe[dataframe.x < 999]
	dataframe = dataframe[dataframe.y < 999]
	print "checked"

	dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 < 11]
	dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 > -1]
	
	print "checking ids of nodeMCU's"
	dataframe = dataframe[dataframe.obj0 < 6]
	dataframe = dataframe[dataframe.obj1 < 6]
	dataframe = dataframe[dataframe.obj2 < 6]
	dataframe = dataframe[dataframe.obj3 < 6]
	dataframe = dataframe[dataframe.obj4 < 6]

	dataframe = dataframe[dataframe.obj0 > -1]
	dataframe = dataframe[dataframe.obj1 > -1]
	dataframe = dataframe[dataframe.obj2 > -1]
	dataframe = dataframe[dataframe.obj3 > -1]
	dataframe = dataframe[dataframe.obj4 > -1]
	print 'checked'

	print "checking errors in received datapacket"
	dataframe = dataframe[dataframe.obj0 != dataframe.obj1]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj2]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj0 != dataframe.obj4]
	
	dataframe = dataframe[dataframe.obj1 != dataframe.obj2]
	dataframe = dataframe[dataframe.obj1 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj1 != dataframe.obj4]
	
	dataframe = dataframe[dataframe.obj2 != dataframe.obj3]
	dataframe = dataframe[dataframe.obj2 != dataframe.obj4]

	dataframe = dataframe[dataframe.obj3 != dataframe.obj4]
	print 'checked'

	dataframe = dataframe[dataframe.rssi0>0]
	dataframe = dataframe[dataframe.rssi1>0]
	dataframe = dataframe[dataframe.rssi2>0]
	dataframe = dataframe[dataframe.rssi3>0]
	dataframe = dataframe[dataframe.rssi4>0]

	print "dataset filtered"
	return dataframe

def average(rssi_list,n,Id):
	list_len = len(rssi_list)
	no_of_chuncks = int(list_len/n)
	

	rssi_list_1 = rssi_list[0:no_of_chuncks*n]
	rssi_list_2 = rssi_list[no_of_chuncks*n:no_of_chuncks*n+1]

	rssi_array_1 = np.array(rssi_list_1)
	rssi_list_2 = np.array(rssi_list_2)
	rssi_shaped_array = np.reshape(rssi_array_1, (no_of_chuncks,n) )
	per_avg_array = np.array([])

	for i in range(len(rssi_shaped_array)):
		print i ,': applying averaging filter on RSSI values of reference point', Id, ' with chunk size' ,n
		temp_array = rssi_shaped_array[i:i+1][0]
		avg = np.mean(temp_array)
		temp_avg_array = np.array([avg]*n)
		per_avg_array = np.concatenate((per_avg_array,temp_avg_array))
	
	filtered_array = np.concatenate((per_avg_array,rssi_list_2))

	return filtered_array

def extract_dist(dataframe,router_pos):
	x_cor = dataframe['x'].values
	y_cor = dataframe['y'].values
	x_router = [router_pos[0]]*len(x_cor)
	y_router = [router_pos[1]]*len(y_cor)

	dist = np.sqrt(np.square(x_cor-x_router) + np.square(y_cor-y_router))

	return dist

def RSSI_id(dataframe,Nodemcu_id):
	object_ids = ['obj0','obj1','obj2','obj3','obj4']
	rssi_columns = ['rssi0','rssi1','rssi2','rssi3','rssi4']
	rssi_list = []
	RSSI_columns = ['obj0','rssi0','obj1','rssi1','obj2','rssi2','obj3','rssi3','obj4','rssi4']
	 
	
	for i in range(len(dataframe)): 
		for j in range(5):
			#print "1 row" ,np.ravel(dataframe[RSSI_columns].iloc[i:i+1].values)
			if len(np.ravel(dataframe[RSSI_columns].iloc[i:i+1].values)) > 0:
				# print dataframe[RSSI_columns].iloc[i:i+1][object_ids[j]].values 	
				if dataframe[RSSI_columns].iloc[i:i+1][object_ids[j]].values == Nodemcu_id:
					print "At row  ", i ,' for reference point', Nodemcu_id, 'sorting rssi values'
					rssi_list = rssi_list + list(dataframe[RSSI_columns].iloc[i:i+1][rssi_columns[j]].values)
	return rssi_list				 

def create_sorted_files():

	for file in file_names:

		Datafile = base_path + 'raw_data/' + file + '.csv'
		data = pd.read_csv(Datafile)					#load the datafile and create a dataframe			
		data = data_cleaning(data)                      #Filter out garbage data           	

		train_dataframe = pd.DataFrame()                #define new dataframe for storing processed data

		train_dataframe['distance'] = extract_dist(data,router_pos)  
		train_dataframe['dt'] = data['dt']
		train_dataframe['T'] = data['T']
		print train_dataframe.head()

		train_dataframe['rssi_0'] =  RSSI_id(data,0)
		train_dataframe['rssi_1'] = RSSI_id(data,1)  
		train_dataframe['rssi_2'] = RSSI_id(data,2)
		train_dataframe['rssi_3'] = RSSI_id(data,3)
		train_dataframe['rssi_4'] = RSSI_id(data,4)

		train_dataframe.to_csv( base_path + 'sorted_data/' + file +'_sorted.csv' )


def Plot(a,b):
	plt.plot(a, b, color ='b', linewidth=1)
	plt.show()

Datafile = base_path + 'sorted_data/' + file_names[0] + '_sorted.csv'
data = pd.read_csv(Datafile)					#load the datafile and create a dataframe			

rssi_array = np.ravel(average(data['rssi_1'],1,1))
dist_array = data['distance']
Plot(dist_array,rssi_array)
color = ['r','g','b']
for i in range(3):
	temp_rssi_array = np.array(rssi_array)[i*100:100 + i*100]
	temp_dist_array = np.array(dist_array)[i*100:100 + i*100]
	plt.plot(temp_dist_array, temp_rssi_array, color =color[i], linewidth=1)
plt.show()
