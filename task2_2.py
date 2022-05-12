#!/usr/bin/env python
# coding: utf-8
import sys, json, pyspark, time 
import numpy as np
from pyspark import SparkContext
import xgboost as xgb


def read(data):
    row = data[0].split(',')
    return (row[0], row[1], row[2])


def read_test(data):
    row = data[0].split(',')
    return (row[0], row[1])


def get_features(related_features):
    review_cnt = []
    bus_rate = []
#    check = []
#    photo = []
#    tip = []
    user_re_cnt = []
    user_rate = []
    for i in range(len(related_features)):
        review_cnt.append(list(related_features[i].values())[0][0])
        bus_rate.append(list(related_features[i].values())[0][1])
#        check.append(list(related_features[i].values())[1])
#        photo.append(list(related_features[i].values())[2])
#        tip.append(list(related_features[i].values())[3])
        user_re_cnt.append(list(related_features[i].values())[1][0])
        user_rate.append(list(related_features[i].values())[1][1])
#    return np.column_stack(np.array([review_cnt,bus_rate,check,photo,tip,user_re_cnt,user_rate], dtype=object))
    return np.column_stack(np.array([review_cnt,bus_rate,user_re_cnt,user_rate], dtype=object))


sc = SparkContext('local[*]', 'task2_2')
sc.setLogLevel('WARN')
sc.setLogLevel("ERROR")

start = time.time()

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

# Load RDD
trainRDD = sc.textFile(folder_path + '/yelp_train.csv')
testRDD = sc.textFile(test_file_name)
header = trainRDD.first()

input_trainRDD = trainRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read(x))
input_testRDD = testRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read_test(x))

# list of unique business and unique users 
train_users = input_trainRDD.map(lambda x : x[0]).distinct().collect()
train_business  = input_trainRDD.map(lambda x : x[1]).distinct().collect()



#### Prepare for variables 
# get user.json 
users = sc.textFile(folder_path + "/user.json").map(lambda line : json.loads(line)).map(lambda x : (x['user_id'], [x['review_count'], x['average_stars']])).collectAsMap()

# get business.json 
business = sc.textFile(folder_path + "/business.json").map(lambda x: json.loads(x)).map(lambda x : (x['business_id'],[x['review_count'], x['stars']])).collectAsMap()

# import check.json
#check = sc.textFile(folder_path + "/checkin.json").map(lambda x: json.loads(x)).filter(lambda x: x['business_id'] in train_business).map(lambda x: (x["business_id"], sum(list(x["time"].values())))).collectAsMap()

# import photo.json
#photo = sc.textFile(folder_path + "/photo.json").map(lambda x: json.loads(x)).filter(lambda x: x['business_id'] in train_business).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()

# get tip.json
#tip = sc.textFile(folder_path + "/tip.json").map(lambda x: json.loads(x)).filter(lambda x: x['business_id'] in train_business).map(lambda x: (x['business_id'], x['likes'])).reduceByKey(lambda x, y: x + y).collectAsMap()


train_variables = input_trainRDD.map(
    lambda x: {"business": business.get(x[1], None), \
#               "check": check.get(x[1], None),\
#               "photo": photo.get(x[1], None), \
#               "tip": tip.get(x[1], None),\
               "users":users.get(x[0],None)}).collect()

train_X = get_features(train_variables)
train_y = np.array(input_trainRDD.map(lambda x : float(x[2])).collect())


test_variables = input_testRDD.map(
    lambda x: {"business": business.get(x[1], [0,2.5]), \
#               "check": check.get(x[1], None),\
#               "photo": photo.get(x[1], None), \
#               "tip": tip.get(x[1], None),\
               "users":users.get(x[0],[0, 2.5])}).collect()

test_X = get_features(test_variables)
#test_y = np.array(input_testRDD.map(lambda x : float(x[2])).collect())


#### Model training
model = xgb.XGBRegressor(max_depth = 10, alpha = 2, eta = 0.2)
model.fit(train_X, train_y)
predicted_y = model.predict(test_X)


with open(output_file_name, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for pair in zip(input_testRDD.collect(), predicted_y):
        line = pair[0][0] + "," + pair[0][1] + "," + str(pair[1]) + "\n"
        f.write(line)
    f.close()
    
end = time.time()
duration = end - start 
print("Duration:", duration)

from sklearn.metrics import mean_squared_error
input_testRDD = testRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read(x))
test_y = np.array(input_testRDD.map(lambda x : float(x[2])).collect())
mse = mean_squared_error(test_y, predicted_y)
print("test MSE: %.4f" % mse)
print("RMSE: %.4f" % (mse ** (1 / 2.0)))

predicted_y_train = model.predict(train_X)
train_y = np.array(input_trainRDD.map(lambda x : float(x[2])).collect())
mse_train = mean_squared_error(train_y, predicted_y_train)
print("train MSE: %.4f" % mse_train)
print("RMSE: %.4f" % (mse_train ** (1 / 2.0)))
