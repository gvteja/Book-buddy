import pickle
import json
import itertools
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics
from os import path
from time import time
from operator import add

#input_file = 'reviews_Books.json.gz'
input_file = 'sample_5p.json'
k = 5
basename = path.basename(input_file)
ts = str(int(time()))
suffix = '_{0}_{1}'.format(basename, ts)
#sc = SparkContext(appName='CF_UI' + suffix)

def extractRating(line):
    '''
    Given a rating json object string, parse it and
    return (item, (user, rating))
    '''
    obj = json.loads(line)
    user = obj['reviewerID']
    item = obj['asin']
    rating = float(obj['overall'])
    return (item, (user, rating))

# Have verified that there are not duplicate (item, user) ratings in the source data
# So avoiding a distinct here to save on time
rdd = sc.textFile(input_file, minPartitions=8)
item_user_map = rdd.map(extractRating)

# Filter out items with less than 25 reviews from users
items_to_remove = item_user_map.map(lambda x: (x[0], 1))\
    .reduceByKey(add)\
    .filter(lambda x: x[1] < 25)
item_user_map = item_user_map.subtractByKey(items_to_remove)

# Filter out users who have rated less than 10 items
user_item_map = item_user_map.map(lambda x: ((x[1][0]), (x[0], x[1][1])))
users_to_remove = user_item_map.map(lambda x: (x[0], (x[1][1], 1)))\
    .reduceByKey(add)\
    .filter(lambda x: x[1] < 10)
user_item_map = user_item_map.subtractByKey(users_to_remove)

item_user_map = user_item_map.map(lambda x: ((x[1][0]), (x[0], x[1][1])))

# Assign a unique integer id to all items and users
# and build 2-way maps for translation
items = item_user_map.keys().distinct().collect()
item_names = {}
item_ids = {}
for i, item in enumerate(items):
    item_names[i] = item
    item_ids[item] = i
item_names = sc.broadcast(item_names)
item_ids = sc.broadcast(item_ids)

users = user_item_map.keys().distinct().collect()
user_names = {}
user_ids = {}
for i, user in enumerate(users):
    user_names[i] = user
    user_ids[user] = i
user_names = sc.broadcast(user_names)
user_ids = sc.broadcast(user_ids)

# Make items and users None to make it eligible for garbage collection
items = None
users = None

# ratings is RDD of (user_id, item_id, rating)
ratings = item_user_map.map(lambda x:\
    (user_ids.value[x[1][0]], item_ids.value[x[0]], x[1][1]))

training, dev, test = ratings.randomSplit([7, 1, 2])

# TODO: do another filtering to ensure that, there is enough data in training(+dev?) for every key, user in test 


# Candidate values for different hyper-parameter to choose from
best_rmse, best_setting = 99999, None
setting_log = []
hyper_params = {
    'rank':[5, 15, 25],
    'iterations':[20],
    'lambda_': [0.1, 1.0, 10.0]
}
params = []
for param, vals in hyper_params.iteritems():
    vals = [(param, val) for val in vals]
    params.append(vals)

# Build a model with each combination of hyper-parameters
# and choose the one with best rmse on the dev set
for setting in itertools.product(*params):
    setting = {k:v for k,v in setting}
    model = ALS.train(training, **setting)
    predictions = model.predictAll(dev.map(lambda x: (x[0], x[1])))
    predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
          .join(dev.map(lambda x: ((x[0], x[1]), x[2])))\
          .values()
    rmse = RegressionMetrics(predictions_ratings).rootMeanSquaredError

    setting_log.append((setting, rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_setting = setting

# Store all the things we want to pickle in this list
to_pickle = []
to_pickle.append(setting_log)
to_pickle.append(best_setting)

# Store all the info we want to print into results
results = []

# Train a new model using the best hyper-param setting and
# using combined training + dev data as the new training data
model = ALS.train(training.union(dev), **best_setting)
predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))

# Calculate rmse using the predicted rating and actual ratings
# on the test set
predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(test.map(lambda x: ((x[0], x[1]), x[2])))\
      .values()
metrics = RegressionMetrics(predictions_ratings)
rmse = metrics.rootMeanSquaredError
results.append('The rmse is {0}'.format(rmse))

# Compute top-k ratings for each user
# and discard users without atleast k ratings
topk = test.map(lambda x: (x[0], [(x[2], x[1])]))\
    .reduceByKey(lambda x,y: x+y)\
    .filter(lambda x: len(x[1]) >= k)\
    .map(lambda x: (x[0], sorted(x[1], reverse=True)[:k]))\
    .flatMap(lambda x: [((x[0], t[1]), t[0]) for t in x[1]])

# Compute rmse on the top k predictions/user
topk_predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(topk)\
      .values()
topk_metrics = RegressionMetrics(topk_predictions_ratings)
topk_rmse = topk_metrics.rootMeanSquaredError
results.append('The topk rmse is {0}'.format(topk_rmse))

# Dump debug data to file
pickle.dump(to_pickle, \
    open('to_pickle' + suffix, "wb"))

# Write result strings to file
with open('results' + suffix, "wb") as f:
    for line in results:
        f.write(line + "\n")
