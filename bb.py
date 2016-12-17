import pickle
import json
import itertools
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics
from math import sqrt
from os import path
from time import time
from operator import add

input_file = 's3://bd-cluster-e/reviews_Books_5.json.gz'
#input_file = 's3://bd-cluster-e/sample_5p.json'
checkpoint_dir = 's3://bd-cluster-e/checkpoints'
#input_file = 'gs://bd_bucket/sample_5p.json'
#checkpoint_dir = 'gs://bd_bucket/checkpoints'

k = 5
# Store all the things we want to pickle in this list
to_pickle = []
# Store all the info we want to print into results
results = []
basename = path.basename(input_file)
ts = str(int(time()))
suffix = '_{0}_{1}'.format(basename, ts)
#sc = SparkContext(appName='CF_UI' + suffix)
# Set checkpoint dir to avoid long lineage
sc.setCheckpointDir(checkpoint_dir)

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
rdd = sc.textFile(input_file).repartition(sc.defaultParallelism * 4)
item_user_map = rdd.map(extractRating)
#item_user_map = item_user_map.sample(False, 0.75, seed=55)

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

training, dev, test = ratings.randomSplit([7, 1, 2], seed=590)
training.cache()
dev.cache()
training_dev = training.union(dev).cache()

# Compute user bias and global mean using training + dev data
user_ratings_sum = training_dev.map(lambda x: (x[0], (x[2], 1)))\
    .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))

sum_training_ratings, num_training_ratings = user_ratings_sum.values()\
    .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
global_training_mean = sum_training_ratings/num_training_ratings
results.append('The global mean is: {0}'.format(global_training_mean))

user_bias = user_ratings_sum.map(lambda x: 
    (x[0], (x[1][0] - global_training_mean * x[1][1]) / x[1][1]))\
    .collectAsMap()
user_bias = sc.broadcast(user_bias)

# Compute item bias using training + dev data
item_bias = training_dev.map(lambda x:\
    (x[1], (x[2] - global_training_mean - user_bias.value[x[0]], 1)))\
    .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))\
    .map(lambda x: (x[0], x[1][0]/x[1][1]))\
    .collectAsMap()
item_bias = sc.broadcast(item_bias)

# Filter out test data where user/item is not in training/dev data
test = test.filter(lambda x:\
    (x[0] in user_bias.value) and (x[1] in item_bias.value))\
    .cache()

# Build set of users in training data
training_users = training.map(lambda x: x[0])\
    .distinct()\
    .collect()
training_users = sc.broadcast(set(training_users))

# Build set of items in training data
training_items = training.map(lambda x: x[1])\
    .distinct()\
    .collect()
training_items = sc.broadcast(set(training_items))

# Filter out dev data where user/item is not in training data
dev_filtered = dev.filter(lambda x:\
    (x[0] in training_users.value) and (x[1] in training_items.value))\
    .cache()

# Weak baseline predictions
weak_baseline_mse = test.map(lambda x: (x[2] - global_training_mean)**2)\
    .mean()
results.append('Weak baseline rmse is {0}'.format(sqrt(weak_baseline_mse)))

# Strong baseline predictions
strong_baseline_mse = test.map(lambda x:\
    (x[2] - global_training_mean - user_bias.value[x[0]] - item_bias.value[x[1]])**2)\
    .mean()
results.append('Strong baseline rmse is {0}'.format(sqrt(strong_baseline_mse)))

# Compute top-k ratings for each user
topk_ratings = test.map(lambda x: (x[0], [(x[2], x[1])]))\
    .reduceByKey(lambda x,y: x+y)\
    .map(lambda x: (x[0], sorted(x[1], reverse=True)[:k]))\
    .cache()

topk_ratings_flattened = topk_ratings\
    .flatMap(lambda x: [((x[0], t[1]), t[0]) for t in x[1]])\
    .cache()

# Compute top-k rmse for weak baseline
wb_topk_mse = topk_ratings_flattened.map(lambda ((u,i),r):\
    (r - global_training_mean)**2)\
    .mean()
results.append('Weak baseline topk rmse is {0}'.format(sqrt(wb_topk_mse)))

# Compute top-k rmse for strong baseline
sb_topk_mse = topk_ratings_flattened.map(lambda ((u,i),r):\
    (r - global_training_mean - user_bias.value[u] - item_bias.value[i])**2)\
    .mean()
results.append('Strong baseline topk rmse is {0}'.format(sqrt(sb_topk_mse)))

# Candidate values for different hyper-parameters in ALS
hyper_params = {
    'rank':[5, 15, 25],
    'iterations':[20],
    'lambda_': [0.1, 1.0, 10.0]
}
params = []
for param, vals in hyper_params.iteritems():
    vals = [(param, val) for val in vals]
    params.append(vals)

# USER - USER model
uu_best_rmse, uu_best_setting = 99999, None
uu_setting_log = []
# Build a user-user model with each combination of hyper-parameters
# and choose the one with best rmse on the dev_filtered set
for setting in itertools.product(*params):
    setting = {k:v for k,v in setting}
    model = ALS.train(training, **setting)
    predictions = model.predictAll(dev_filtered.map(lambda x: (x[0], x[1])))
    predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2]))\
          .join(dev_filtered.map(lambda x: ((x[0], x[1]), x[2])))\
          .values()
    rmse = RegressionMetrics(predictions_ratings).rootMeanSquaredError

    uu_setting_log.append((setting, rmse))
    if rmse < uu_best_rmse:
        uu_best_rmse = rmse
        uu_best_setting = setting

to_pickle.append(uu_setting_log)
to_pickle.append(uu_best_setting)
results.append('The best setting for user-user model is {0}'.format(uu_best_setting))

# Train a new user-user model using the best hyper-param setting and
# using combined training + dev data as the new training data
uu_model = ALS.train(training_dev, **uu_best_setting)
uu_predictions = uu_model.predictAll(test.map(lambda x: (x[0], x[1])))

# Calculate rmse using the predicted rating and actual ratings
# on the test set
uu_predictions_ratings = uu_predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(test.map(lambda x: ((x[0], x[1]), x[2])))\
      .values()
uu_rmse = RegressionMetrics(uu_predictions_ratings).rootMeanSquaredError
results.append('The user-user rmse is {0}'.format(uu_rmse))

# Compute rmse on the top k predictions/user
uu_topk_predictions_ratings = uu_predictions.map(lambda x: ((x[0], x[1]), x[2]))\
      .join(topk_ratings_flattened)\
      .values()
uu_topk_rmse = RegressionMetrics(uu_topk_predictions_ratings).rootMeanSquaredError
results.append('The user-user topk rmse is {0}'.format(uu_topk_rmse))


# ITEM - ITEM model
# Interchange user and item positions i.e. rows and columns
# in both training and combined training + dev set
training = training.map(lambda x: (x[1], x[0], x[2])).cache()
training_dev = training_dev.map(lambda x: (x[1], x[0], x[2])).cache()
ii_best_rmse, ii_best_setting = 99999, None
ii_setting_log = []

# Trigger computation of training and training_dev to avoid 
# race condition and incorrectly using old training and training_dev
training.count()
training_dev.count()

# Build an item-item model with each combination of hyper-parameters
# and choose the one with best rmse on the dev_filtered set
for setting in itertools.product(*params):
    setting = {k:v for k,v in setting}
    model = ALS.train(training, **setting)
    predictions = model.predictAll(dev_filtered.map(lambda x: (x[1], x[0])))
    predictions_ratings = predictions.map(lambda x: ((x[1], x[0]), x[2]))\
          .join(dev_filtered.map(lambda x: ((x[0], x[1]), x[2])))\
          .values()
    rmse = RegressionMetrics(predictions_ratings).rootMeanSquaredError

    ii_setting_log.append((setting, rmse))
    if rmse < ii_best_rmse:
        ii_best_rmse = rmse
        ii_best_setting = setting

to_pickle.append(ii_setting_log)
to_pickle.append(ii_best_setting)
results.append('The best setting for item-item model is {0}'.format(ii_best_setting))

# Train a new item-item model using the best hyper-param setting and
# using combined training + dev data as the new training data
ii_model = ALS.train(training_dev, **ii_best_setting)
ii_predictions = ii_model.predictAll(test.map(lambda x: (x[1], x[0])))

# Calculate item-item rmse using the predicted rating and actual ratings
# on the test set
ii_predictions_ratings = ii_predictions.map(lambda x: ((x[1], x[0]), x[2]))\
      .join(test.map(lambda x: ((x[0], x[1]), x[2])))\
      .values()
ii_rmse = RegressionMetrics(ii_predictions_ratings).rootMeanSquaredError
results.append('The item-item rmse is {0}'.format(ii_rmse))

# Compute item-item rmse on the top k predictions/user
ii_topk_predictions_ratings = ii_predictions.map(lambda x: ((x[1], x[0]), x[2]))\
      .join(topk_ratings_flattened)\
      .values()
ii_topk_rmse = RegressionMetrics(ii_topk_predictions_ratings).rootMeanSquaredError
results.append('The item-item topk rmse is {0}'.format(ii_topk_rmse))


# NORMALIZED COLLOBARATIVE FILTERING MODELS
# Undo the interchange of user and item positions and build
# normalized rating using user and item biases
training_dev = training_dev.map(lambda (i,u,r): (u, i,\
    (r - global_training_mean - user_bias.value[u] - item_bias.value[i])))\
    .cache()
training = training.map(lambda (i,u,r): (u, i,\
    (r - global_training_mean - user_bias.value[u] - item_bias.value[i])))\
    .cache()

# Trigger computation of training and training_dev to avoid 
# race condition
training.count()
training_dev.count()


# Normalized USER - USER normalized
uu_best_rmse, uu_best_setting = 99999, None
uu_setting_log = []
# Build a normalized user-user model with each combination of hyper-parameters
# and choose the one with best rmse on the dev_filtered set
for setting in itertools.product(*params):
    setting = {k:v for k,v in setting}
    model = ALS.train(training, **setting)
    predictions = model.predictAll(dev_filtered.map(lambda (u,i,r): (u, i)))
    predictions_ratings = predictions.map(lambda (u,i,r): ((u, i), (r + global_training_mean + user_bias.value[u] + item_bias.value[i])))\
          .join(dev_filtered.map(lambda (u,i,r): ((u, i), r)))\
          .values()
    rmse = RegressionMetrics(predictions_ratings).rootMeanSquaredError

    uu_setting_log.append((setting, rmse))
    if rmse < uu_best_rmse:
        uu_best_rmse = rmse
        uu_best_setting = setting

to_pickle.append(uu_setting_log)
to_pickle.append(uu_best_setting)
results.append('The best setting for normalized user-user model is {0}'.format(uu_best_setting))

# Train a new normalized user-user model using the best hyper-param setting and
# using combined training + dev data as the new training data
uu_model = ALS.train(training_dev, **uu_best_setting)
uu_predictions = uu_model.predictAll(test.map(lambda x: (x[0], x[1])))\
    .map(lambda (u,i,r): ((u, i), (r + global_training_mean + user_bias.value[u] + item_bias.value[i])))

# Calculate rmse using the predicted rating and actual ratings
# on the test set
uu_predictions_ratings = uu_predictions.join(test.map(lambda (u,i,r): ((u, i), r)))\
      .values()
uu_rmse = RegressionMetrics(uu_predictions_ratings).rootMeanSquaredError
results.append('The normalized user-user rmse is {0}'.format(uu_rmse))

# Compute rmse on the top k predictions/user
uu_topk_predictions_ratings = uu_predictions.join(topk_ratings_flattened)\
      .values()
uu_topk_rmse = RegressionMetrics(uu_topk_predictions_ratings).rootMeanSquaredError
results.append('The normalized user-user topk rmse is {0}'.format(uu_topk_rmse))


# ITEM - ITEM normalized model
# Interchange user and item positions i.e. rows and columns
# in both training and combined training + dev set
training = training.map(lambda x: (x[1], x[0], x[2])).cache()
training_dev = training_dev.map(lambda x: (x[1], x[0], x[2])).cache()
ii_best_rmse, ii_best_setting = 99999, None
ii_setting_log = []

# Trigger computation of training and training_dev to avoid 
# race condition like before
training.count()
training_dev.count()

# Build an item-item model with each combination of hyper-parameters
# and choose the one with best rmse on the dev_filtered set
for setting in itertools.product(*params):
    setting = {k:v for k,v in setting}
    model = ALS.train(training, **setting)
    predictions = model.predictAll(dev_filtered.map(lambda (u,i,r): (i, u)))
    predictions_ratings = predictions.map(lambda (i,u,r): ((u, i), (r + global_training_mean + user_bias.value[u] + item_bias.value[i])))\
          .join(dev_filtered.map(lambda (u,i,r): ((u, i), r)))\
          .values()
    rmse = RegressionMetrics(predictions_ratings).rootMeanSquaredError

    ii_setting_log.append((setting, rmse))
    if rmse < ii_best_rmse:
        ii_best_rmse = rmse
        ii_best_setting = setting

to_pickle.append(ii_setting_log)
to_pickle.append(ii_best_setting)
results.append('The best setting for normalized item-item model is {0}'.format(ii_best_setting))

# Train a new item-item model using the best hyper-param setting and
# using combined training + dev data as the new training data
ii_model = ALS.train(training_dev, **ii_best_setting)
ii_predictions = ii_model.predictAll(test.map(lambda x: (x[1], x[0])))\
    .map(lambda (i,u,r): ((u, i), (r + global_training_mean + user_bias.value[u] + item_bias.value[i])))

# Calculate item-item rmse using the predicted rating and actual ratings
# on the test set
ii_predictions_ratings = ii_predictions.join(test.map(lambda (u,i,r): ((u, i), r)))\
      .values()
ii_rmse = RegressionMetrics(ii_predictions_ratings).rootMeanSquaredError
results.append('The normalized item-item rmse is {0}'.format(ii_rmse))

# Compute item-item rmse on the top k predictions/user
ii_topk_predictions_ratings = ii_predictions.join(topk_ratings_flattened)\
      .values()
ii_topk_rmse = RegressionMetrics(ii_topk_predictions_ratings).rootMeanSquaredError
results.append('The normalized item-item topk rmse is {0}'.format(ii_topk_rmse))


results.append('The number of ratings are {0}'.format(rdd.count()))
results.append('The number of filtered ratings are {0}'.format(ratings.count()))
results.append('The number of filtered users are {0}'.format(len(user_ids.value)))
results.append('The number of filtered items are {0}'.format(len(item_ids.value)))

# Dump debug data to file
pickle.dump(to_pickle, \
    open('to_pickle' + suffix, "wb"))

# Write result strings to file
with open('results' + suffix, "wb") as f:
    for line in results:
        f.write(line + "\n")