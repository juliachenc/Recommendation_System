#!/usr/bin/env python
# coding: utf-8
import pyspark, json, sys, time, itertools, math, functools
import numpy as np
from pyspark import SparkContext
import xgboost as xgb


def pearson(bus_dict, bus1, bus2):
    num = 0
    sum_squared1 = 0
    sum_squared2 = 0
    bus1_dict = dict(bus_dict[bus1])
    bus2_dict = dict(bus_dict[bus2])
    
    if not bus1_dict or not bus2_dict:
        return 0
    co_rated = set(bus1_dict.keys()) & set(bus2_dict.keys())
    
    if len(co_rated) < 75:
        return 0
    b1_avg = functools.reduce(lambda x, y: x + y, list(bus1_dict.values())) / len(bus1_dict)
    b2_avg = functools.reduce(lambda x, y: x + y, list(bus2_dict.values())) / len(bus2_dict)
    for item in co_rated:
        num += ((bus1_dict[item] - b1_avg) * (bus2_dict[item] - b2_avg))
        sum_squared1 += (bus1_dict[item] - b1_avg) ** 2
        sum_squared2 += (bus2_dict[item] - b2_avg) ** 2
        
    return num / (math.sqrt(sum_squared1) * math.sqrt(sum_squared1)) if sum_squared1 and sum_squared2 else 0

def prediction(neighbors, user, user_dict):
    min_neighbors = min(3, len(neighbors))
    filtered_neighbors = filter(lambda x: x[0] > 0, neighbors)

    if not neighbors or not filtered_neighbors:
        return 3    
    top_candidates = sorted(filtered_neighbors, key=lambda x: x[0])[::-1][0 : min_neighbors]
    top_candidates_total = sum([abs(k) for k in list(dict(top_candidates).keys())])
    if top_candidates_total != 0:
        return sum([v * (k) for (k, v) in top_candidates]) / top_candidates_total
    else:
        return sum([v for (k, v) in user_dict[user]]) / len(user_dict[user]) if user_dict[user] else 3

def get_features(related_features):
    review_cnt = []
    bus_rate = []
    user_re_cnt = []
    user_rate = []
    for i in range(len(related_features)):
        review_cnt.append(list(related_features[i].values())[0][0])
        bus_rate.append(list(related_features[i].values())[0][1])
        user_re_cnt.append(list(related_features[i].values())[1][0])
        user_rate.append(list(related_features[i].values())[1][1])
    return np.column_stack(np.array([review_cnt,bus_rate,user_re_cnt,user_rate], dtype=object))

def read(data):
    row = data[0].split(',')
    return (row[0], row[1], float(row[2]))

def read_test(data):
    row = data[0].split(',')
    return (row[0], row[1])

def read1(data):
    row = data[0].split(',')
    return (row[1], row[0], float(row[2]))

def read_test1(data):
    row = data[0].split(',')
    return (row[1], row[0])



sc = SparkContext('local[*]', 'task2_3')
sc.setLogLevel('WARN')
sc.setLogLevel("ERROR")

start = time.time()

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


# ## XGB Model Based


## Load RDD
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
users = sc.textFile(folder_path + "/user.json").map(lambda line : json.loads(line)).filter(lambda x : x['user_id'] in train_users).map(lambda x : (x['user_id'], [x['review_count'], x['average_stars']])).collectAsMap()

# get business.json 
business = sc.textFile(folder_path + "/business.json").map(lambda x: json.loads(x)).filter(lambda x: x['business_id'] in train_business).map(lambda x : (x['business_id'],[x['review_count'], x['stars']])).collectAsMap()


train_variables = input_trainRDD.map(lambda x: {"business": business.get(x[1], None),"users":users.get(x[0],None)}).collect()
train_X = get_features(train_variables)
train_y = np.array(input_trainRDD.map(lambda x : float(x[2])).collect())


test_variables = input_testRDD.map(lambda x: {"business": business.get(x[1], [0,2.5]),"users":users.get(x[0],[0, 2.5])}).collect()
test_X = get_features(test_variables)


model = xgb.XGBRegressor(max_depth = 5, alpha = 2, eta = 0.2)
model.fit(train_X, train_y)
predicted_y = model.predict(test_X)


# ### Collaborative Filtering
input_trainRDD = trainRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read1(x))
input_testRDD = testRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read_test1(x))


# list of unique business and unique users 
train_users = input_trainRDD.map(lambda x : x[1]).distinct().collect()
train_business = input_trainRDD.map(lambda x : x[0]).distinct().collect()

test_users = input_testRDD.map(lambda x : x[1]).distinct().collect()
test_business = input_testRDD.map(lambda x : x[0]).distinct().collect()


all_business = sorted(set(train_business) | set(test_business))
all_users = sorted(set(train_users) | set(test_users))


### Get dictionary of user id & index    
user_dict = dict((user, i) for i, user in enumerate(all_users))

### Reverse  of user id & index
reverse_user_dict = dict(zip(range(len(user_dict)), user_dict))


### Get dictionary of business id & index
business_dict = dict((business, i) for i, business in enumerate(all_business))

### Reverse  of business id & index
reverse_business_dict = dict(zip(range(len(business_dict)), business_dict))


mapping_list = input_testRDD.map(lambda x: (x[0], x[1])).collect()


output_dict = dict(((user_dict[mapping_list[i][1]], business_dict[mapping_list[i][0]]), predicted_y[i]) for i in range(len(predicted_y)))


user_dict_rating = input_trainRDD.map(lambda x: (user_dict[x[1]], (business_dict[x[0]], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1]))).collectAsMap()


list_tuple_user = [(i, []) for i in user_dict.values() if i not in user_dict_rating.keys()]
user_dict_rating.update(list_tuple_user)


business_dict_rating = input_trainRDD.map(lambda x: (business_dict[x[0]], (user_dict[x[1]], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1]))).collectAsMap()


list_tuple_business = [(i, []) for i in business_dict.values() if i not in business_dict_rating.keys()]
business_dict_rating.update(list_tuple_business)


predicted_test = input_testRDD.map(lambda x: (user_dict[x[1]], business_dict[x[0]])).map(lambda x: (x[0], x[1], prediction([(pearson(business_dict_rating, x[1], business_id), score) for (business_id, score) in user_dict_rating[x[0]]], x[0], user_dict_rating))).map(lambda x: (x[0], x[1], output_dict[(x[0], x[1])] * 0.95 + x[2] * 0.05 )).map(lambda x: (reverse_user_dict[x[0]], reverse_business_dict[x[1]], x[2])).collect()


with open(output_file_name, "w") as f:
    f.write("user_id, business_id, prediction\n")
    for line in predicted_test:
        f.write(line[0] + "," + line[1] + "," + str(line[2]) + "\n")
    f.close()
    
    
end = time.time()
duration = end - start 
print("Duration:", duration)


predicted_y = []
for i in range(len(predicted_test)):
    predicted_y.append(predicted_test[i][2])
test_rdd = sc.textFile(test_file_name).map(lambda x: x.strip().split(",")) \
    .filter(lambda x: x[0] != 'user_id') \
    .map(lambda x: (str(x[1]), str(x[0]), float(x[2])))
test_y =  test_rdd.map(lambda x : float(x[2])).collect()

from sklearn.metrics import mean_squared_error


mse = mean_squared_error(test_y, predicted_y)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % (mse ** (1 / 2.0)))



