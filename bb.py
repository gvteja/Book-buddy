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
sc = SparkContext(appName='CF_UI' + suffix)

def extractRating(line):
    obj = json.loads(line)
    user = obj['reviewerID']
    item = obj['asin']
    rating = float(obj['overall'])
    return (item, (user, rating))

# Have verified that there are not duplicate (item, user) ratings in the source data
# So avoiding a distinct here to save on time
rdd = sc.textFile(input_file, minPartitions=8)
item_user_map = rdd.map(extractRating)

items_to_remove = item_user_map.map(lambda x: (x[0], 1))\
    .reduceByKey(add)\
    .filter(lambda x: x[1] < 25)
item_user_map = item_user_map.subtractByKey(items_to_remove)

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


# Optimize hyper-parameters using dev set 
best_rmse, best_setting = 99999, None
setting_log = []
hyper_params = {
    'rank':[5, 15, 25],
    'iterations':[20],
    'lambda_': [0.1, 1, 10.0]
}
params = []
for param, vals in hyper_params.iteritems():
    vals = [(param, val) for val in vals]
    params.append(vals)
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

# Train a new model using the best hyper-param setting and
# using combined training + dev data as the new training data
model = ALS.train(training.union(dev), **best_setting)
predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))

predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(test.map(lambda x: ((x[0], x[1]), x[2])))\
      .values()
metrics = RegressionMetrics(predictions_ratings)
rmse = metrics.rootMeanSquaredError
print rmse

# Compute top-k ratings for each user
# and discard users without atleast k ratings
topk = test.map(lambda x: (x[0], [(x[2], x[1])]))\
    .reduceByKey(lambda x,y: x+y)\
    .filter(lambda x: len(x[1]) >= k)\
    .map(lambda x: (x[0], sorted(x[1], reverse=True)[:k]))\
    .flatMap(lambda x: [((x[0], t[1]), t[0]) for t in x[1]])

topk_predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(topk)\
      .values()
topk_metrics = RegressionMetrics(topk_predictions_ratings)
topk_rmse = topk_metrics.rootMeanSquaredError
print topk_rmse











means = item_user_map.map(lambda x: (x[0], (x[1][1], 1.0)))\
    .reduceByKey(lambda x,y: ((x[0] + y[0]), (x[1] + y[1])))\
    .map(lambda x: (x[0], x[1][0]/x[1][1]))

# normalize item user ratings
item_user_map = item_user_map.join(means)\
    .map(lambda x: (x[0], (x[1][0][0], x[1][0][1] - x[1][1])))

means = means.collectAsMap()
interested_mean = means[interested_item]

# Need to filter out users/items with zero variance i.e zero rows
# But is not really necessary for now since they anyway would be 0 cosine

norms = item_user_map.map(lambda x: (x[0], x[1][1] ** 2))\
    .reduceByKey(add)\
    .map(lambda x: (x[0], x[1] ** 0.5))
norms = sc.broadcast(norms.collectAsMap())

# maybe throw rows with 0 norms. they are zero variance. or rows with < some small epsilon

interested_row = item_user_map.lookup(interested_item)
interested_row = {u:r for (u, r) in interested_row}
interested_row = sc.broadcast(interested_row)

interested_norm = norms.value[interested_item]

# compute cosine similarity scores for and get all valid neighbours
# valid implies having >= 2 users in common 
# and having cosine sim > 0
scores = item_user_map.flatMap(\
    lambda x: computeProductIntersection(x, interested_row.value))\
    .reduceByKey(lambda x,y: ((x[0] + y[0]), (x[1] + y[1])))\
    .filter(lambda x: x[1][1] >= 2)\
    .map(lambda x: computeScore(x, norms.value, interested_norm))\
    .filter(lambda x: x[1] > 0)

scores = scores.collectAsMap()
# With a target row, skip columns that do not have at least 2 neighbors

# compute interested users rdd
rated_users = sc.parallelize([(interested_item, None)])\
    .join(item_user_map)\
    .map(lambda x: (x[1][1][0], None))

user_item_map = item_user_map.map(lambda x: ((x[1][0]), (x[0], x[1][1])))
interested_user_ratings = user_item_map.subtractByKey(rated_users)
predictions = interested_user_ratings.groupByKey()\
    .flatMap(lambda x: predictScore(x, scores))\
    .collect()

full_row = [(i, r + interested_mean) for i, r in interested_row.value.items()]
full_row = full_row + predictions


pickle.dump(scores, \
    open('scores' + suffix, "wb"))
pickle.dump(predictions, \
    open('predictions' + suffix, "wb"))
pickle.dump(full_row, \
    open('row' + suffix, "wb"))

with open('output' + suffix, "wb") as f:
    for item, rating in full_row:
        f.write("{0} - {1}\n".format(item, rating))