#!/usr/bin/env python
# coding: utf-8
import sys, json, pyspark, time, random, itertools
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict


def read(data):
    row = data[0].split(',')
    return (row[0], row[1], row[2])


def hashFunc(hash_num):
    listA = random.sample(range(1, sys.maxsize - 1), hash_num)
    listB = random.sample(range(0, sys.maxsize - 1), hash_num)
    return [[listA[i], listB[i]] for i in range(hash_num)]


## get hash table
def getHash(x, hash_list, m):
    return min([((hash_list[0] * t + hash_list[1]) % m) for t in x[1]])


def band(x):
    user = x[1]
    buss = [(x[0])]
    return [((i, tuple(user[i * row :(i + 1) * row])), buss) for i in range(band_num)]


def pair_bus_user(x):
    return sorted(list(combinations(sorted(x), 2)))


def jaccard(users1, users2):
    v1 = len((set(user_bus_matrix_dic[users1])).intersection(set(user_bus_matrix_dic[users2])))
    v2 = len((set(user_bus_matrix_dic[users1])).union(set(user_bus_matrix_dic[users2])))
    return users1, users2, v1/v2


sc = SparkContext('local[*]', 'task1')
sc.setLogLevel('WARN')
sc.setLogLevel("ERROR")


hash_num = 60
band_num = 30
hashed = hashFunc(hash_num)
row = int(len(hashed) / band_num)


start = time.time()
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]



input_rdd = sc.textFile(input_file_path)
header = input_rdd.first()
input_rdd = input_rdd.filter(lambda x : x != header).map(lambda x: x.split('\n')).map(lambda x: read(x))



# business id 
business_dict_rdd = input_rdd.map(lambda x: x[1]).distinct().sortBy(lambda x : x).zipWithIndex()
business_dict = business_dict_rdd.collectAsMap()


# user id
user_dict_rdd = input_rdd.map(lambda x: x[0]).distinct().sortBy(lambda x : x[0]).zipWithIndex()
user_dict = user_dict_rdd.collectAsMap()


# user and business 
user_bus_matrix = input_rdd.map(lambda x: (x[1],user_dict[x[0]])).groupByKey().mapValues(set).sortBy(lambda x: x[0])


user_bus_matrix_dic = user_bus_matrix.collectAsMap()


hashed_user_rdd = user_bus_matrix.map(lambda x: (x[0], [getHash(x, h, len(user_dict)) for h in hashed]))


candidates = hashed_user_rdd.flatMap(band).reduceByKey(lambda a, b: a + b).filter(lambda x: len(x[1]) != 0).flatMap(lambda x: pair_bus_user(list(x[1]))).sortBy(lambda x: (x[0], x[1])).distinct()


final_candidates = candidates.map(lambda x: jaccard(x[0], x[1])).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: (x[0], x[1])).collect()



with open(output_file_path, 'w') as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for pair in final_candidates:
        f.write(pair[0] + "," + pair[1] + "," + str(pair[2]) + "\n")
    f.close()
    
end = time.time()
duration = end - start 
print("Duration:", duration)



