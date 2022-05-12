#!/usr/bin/env python
# coding: utf-8
import pyspark, sys, time, math
from pyspark import SparkContext
from functools import reduce


def read(data):
    row = data[0].split(',')
    return (row[0], row[1], row[2])


def read_test(data):
    row = data[0].split(',')
    return (row[0], row[1])


def average(score_list):
    float_conversion = [float(pair[1]) for pair in score_list]
    return reduce(lambda x, y: x + y, float_conversion) / len(score_list)


def prediction(scores, business_dict, avg_rating):
    score_list = []
    num = 0
    dom = 0
    
    for i in range(len(list(scores[1]))):
        bus_1, bus_2 = list(scores[1])[i]
        val = business_dict.get((bus_1, scores[0]), 0)
        score_list.append((float(bus_2), val))
    
    for score_weight in sorted(score_list, key=lambda score_weight: score_weight[1])[-50:]:
        num += (score_weight[0] * score_weight[1])
        dom += abs(score_weight[1])
        
    return (scores[0], num/dom) if dom and num else (scores[0], avg_rating.get(scores[0]))


def pearson(bus1, bus2):
    s1 = set()
    s2 = set()
    num = 0
    sum_squared1 = 0
    sum_squared2 = 0
    
    for user_b1 in bus1_rating.keys():
        s1.add(float(bus1_scores[user_b1]))
        s2.add(float(bus2_scores[user_b1]))
    for user_b2 in bus2_rating.keys():
        s1.add(float(bus1_scores[user_b2]))
        s2.add(float(bus2_scores[user_b2]))   
    
    avg1 = reduce(lambda x, y: x + y, list(s1)) / len(s1)
    avg2 = reduce(lambda x, y: x + y, list(s2)) / len(s2)
    
    for i in range(len(list(s1))):
        diff1 = list(s1)[i] - avg1
        diff2 = list(s2)[i] - avg2
        num += diff1 * diff2
        sum_squared1 += diff1 ** 2
        sum_squared2 += diff2 ** 2
    return num / (math.sqrt(sum_squared1) * math.sqrt(sum_squared1)) if sum_squared1 and sum_squared2 else 0



sc = SparkContext('local[*]', 'task2_1')
sc.setLogLevel('WARN')
sc.setLogLevel("ERROR")

start = time.time()

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

# Load RDD
trainRDD = sc.textFile(train_file_name)
testRDD = sc.textFile(test_file_name)
header = trainRDD.first()

input_trainRDD = trainRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read(x))
input_testRDD = testRDD.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read_test(x))

### Get dictionary of user id & index
user_dict = input_trainRDD.map(lambda x: x[0]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()

### Reverse  of user id & index
idx_user_dict = dict(zip(range(len(user_dict)), user_dict))

### Get dictionary of business id & index
business_dict = input_trainRDD.map(lambda x: x[1]).distinct().sortBy(lambda x : x).zipWithIndex().collectAsMap()

### Reverse  of business id & index
idx_business_dict = dict(zip(range(len(business_dict)), business_dict))

### Get business rating & average rating
business_rating = input_trainRDD.map(lambda x: (business_dict[x[1]], (user_dict[x[0]], x[2]))).groupByKey().mapValues(list)
business_all_rating = business_rating.collectAsMap()
business_avg_rating = business_rating.map(lambda x: (x[0], average(x[1]))).collectAsMap()

### Get user rating and average rating
user_rating = input_trainRDD.map(lambda x: (user_dict[x[0]], (business_dict[x[1]], x[2]))).groupByKey().mapValues(list)
user_all_rating = user_rating.collectAsMap()
user_avg_rating = user_rating.map(lambda x: (x[0], average(x[1]))).collectAsMap()

# Clean testing data
test_user_bus_rdd = input_testRDD.map(lambda x: (user_dict.get(x[0], -1), business_dict.get(x[1], -1))).filter(lambda x: x[0] != -1 and x[1] != -1) 

candidate_pairs = test_user_bus_rdd.leftOuterJoin(user_rating).flatMap(lambda x: [(bus_score[0], x[1][0]) for bus_score in x[1][1]])

filtered_candidate = input_testRDD.filter(lambda x : x[1] not in business_dict or x[0] not in user_dict).collect()

final_candidate = candidate_pairs.filter(lambda x : len(dict(business_all_rating.get(x[0])).keys() & dict(business_all_rating.get(x[1])).keys()) >= 300).map(lambda x : (x, pearson(dict(business_all_rating.get(x[0])), dict(business_all_rating.get(x[1]))))).collectAsMap()

output_final = test_user_bus_rdd.leftOuterJoin(user_rating).map(lambda x : (x[0], prediction(x[1], final_candidate, business_avg_rating)))


### Generate the output
final_dict = {}
for pair in output_final.collect():
    final_dict[(idx_user_dict[pair[0]], idx_business_dict[pair[1][0]])] = str(pair[1][1])
for pair in filtered_candidate:
    if pair[0] in user_dict.keys():
        final_dict[(pair[0], pair[1])] = str(user_avg_rating[user_dict[pair[0]]])
    elif pair[1] in business_dict.keys():
        final_dict[(pair[0], pair[1])] = str(business_avg_rating[business_dict[pair[0]]])
    else:
        final_dict[(pair[0], pair[1])] = str(0.0)

with open(output_file_name, "w") as f:
    f.write('user_id, business_id, prediction\n')
    for (user_id, buss_id) in input_testRDD.collect():
        f.write(str(user_id) + "," + str(buss_id) + "," + final_dict[(user_id, buss_id)] + "\n")
        
end = time.time()
duration = end - start 
print("Duration:", duration)

