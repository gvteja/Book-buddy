import pickle
import json
from pyspark import SparkContext, StorageLevel
from itertools import combinations
from os import path
from time import time
from operator import add

#input_file = 'reviews_Books.json.gz'
input_file = 'sample_5p.json'
interested_item = '0439023513'
basename = path.basename(input_file)
ts = str(int(time()))
suffix = '_{0}_{1}'.format(basename, ts)
sc = SparkContext(appName='Vj_CF' + suffix)

def extractRating(line):
    obj = json.loads(line)
    user = obj['reviewerID']
    item = obj['asin']
    rating = float(obj['overall'])
    return (item, (user, rating))

def computeScore(tup, norms, interested_norm):
    item, (dot, _) = tup
    norm = norms[item]
    # Can remove this check if we filter out things with 0 norm
    if not norm:
        return (x[0], 0)
    return (item, dot/(norm * interested_norm))

def computeProductIntersection(tup, interested_row):
    item, (user, rating) = tup
    try:
        product = interested_row[user] * rating
    except:
        # user not in interested row
        # no intersection and product is 0
        return []
    return [(item, (product, 1))]

def predictScore(tup, scores):
    user, item_ratings = tup
    neighbour_ratings = []
    for item, r in item_ratings:
        if item in scores:
            neighbour_ratings.append((scores[item], r))
    if len(neighbour_ratings) < 2:
        return []
    # find the top 50 of these items based on their sim score and
    # compute a weighted avg
    neighbour_ratings.sort(reverse=True)
    predicted_score = sum(rating * sim for (sim, rating) in neighbour_ratings[:50])
    predicted_score /= sum(sim for (sim, _) in neighbour_ratings[:50])
    predicted_score += interested_mean
    return [(user, predicted_score)]

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

training, dev, test = item_user_map.randomSplit([7, 1, 2])

# TODO: do another filtering to ensure that, there is enough data in training(+dev?) for every key, user in test 


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