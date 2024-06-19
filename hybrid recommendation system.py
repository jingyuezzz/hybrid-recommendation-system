from pyspark import SparkContext, SparkConf
import sys
import time
import xgboost as xgb
import csv
import math
import json
import pandas as pd
from operator import add
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


'''
Method Description:
For this project, I employed a hybrid recommendation system, combining Item-based CF with a 
model-based recommendation system. Based on the weighting assigned to recommendation systems from homework3, 
the model-based CF demonstrated better performance, carrying a higher weight. To improve the XGBoost model,
I experimented with the inclusion of additional features directly sourced from 'user.json' and 'business.json,' such as
'funny,' 'cool,' 'compliment_hot,' and 'BusinessAcceptsCreditCards.' I also created some features, including 
'friends_count' (counting the number of user's friends), 'encoded_categories' (label-encoded categories from 
'business.json'), and 'photo_ct' (photo count for each business from 'photo.json'). Among these features, only 
'compliment_hot,' 'BusinessAcceptsCreditCards,' and 'photo_ct' demonstrated improvements in the model. Since there is 
a lot of categories, the label-encoded categories did not prove effective as a feature. I identified the top six most 
common categories and transformed them into dummy variables. This resulted in a reduction of RMSE by 0.001.


Error Distribution:
<1     102021
1-2     33077
2-3      6177
3-4       769
>=4         0

RMSE: 
0.9530401889146038

RMSE(without val in train): 
0.9719206221449755

Execution Time:
400s

'''

start = time.time()
sc = SparkContext('local[*]', 'task12-3')



folder = ''
test_path = 'yelp_val.csv'

sc.setLogLevel("WARN")


def pearson_similarity(bus1, bus2):
    if bus1 not in b_u_rating_dict.keys() or bus2 not in b_u_rating_dict.keys():
        return 0
    else:
        common_users = set(b_u_rating_dict[bus1]).intersection(b_u_rating_dict[bus2])
        if len(common_users) == 0:
            return 0
        mean_bus1 = sum(b_u_rating_dict[bus1][user] for user in common_users) / len(common_users)
        mean_bus2 = sum(b_u_rating_dict[bus2][user] for user in common_users) / len(common_users)
        numerator = 0
        den1 = 0
        den2 = 0
        for user in common_users:
            rating_bus1 = b_u_rating_dict[bus1][user]
            rating_bus2 = b_u_rating_dict[bus2][user]
            adj_rating_bus1 = rating_bus1 - mean_bus1
            adj_rating_bus2 = rating_bus2 - mean_bus2
            numerator += adj_rating_bus1 * adj_rating_bus2
            den1 += adj_rating_bus1 ** 2
            den2 += adj_rating_bus2 ** 2
        if den1 == 0 or den2 == 0:
            return 0
        similarity = numerator / (math.sqrt(den1) * math.sqrt(den2))
        return similarity

def predict_score(bus,user):
    if user not in user_list:
        return 3.7
    if bus not in bus_list:
        return user_avg_rating[user]
    score_list = []
    for b in user_bus_dict[user]:
        if (b != bus):
            score = pearson_similarity(b, bus)
            if score>0:
                score_list.append((score, b_u_rating_dict[b][user]))
    score_sorted = sorted(score_list, key=lambda x: -x[0])[:14]
    num = 0
    den = 0
    res = 0
    for sim,rating in score_sorted:
        num += sim*rating
        den += abs(sim)
    if den !=0:
        res = (num/den)*0.2+user_avg_rating[user]*0.4+bus_avg_rating[bus]*0.4
    else:
        res = 3.7
    return res

input_path = folder+'yelp_train.csv'
val_path = folder+'yelp_val.csv'
'''
data = sc.textFile(input_path)
header3 = data.first()
data = data.filter(lambda x: x != header3)
'''
data_train = sc.textFile(input_path)
data_val = sc.textFile(val_path)
# Remove headers from RDDs
header_train = data_train.first()
data_train = data_train.filter(lambda x: x != header_train)
header_val = data_val.first()
data_val = data_val.filter(lambda x: x != header_val)
# Combine the data from both DataFrames
data = data_train.union(data_val)

train_rdd = data.map(lambda line: line.split(',')).map(lambda x: (x[0], x[1], float(x[2])))
#user list business list
user_id = train_rdd.map(lambda x: x[0]).distinct()
user_list = user_id.collect()
bus_id = train_rdd.map(lambda x: x[1]).distinct()
bus_list = bus_id.collect()
#user average rating
user_avg= train_rdd.map(lambda x: (x[0], x[2])).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1])))
user_avg_rating = user_avg.collectAsMap()
#matrix
bus_user = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1])))
bus_user_dict = bus_user.collectAsMap()
user_bus = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], list(x[1])))
user_bus_dict = user_bus.collectAsMap()
#bus-(user,rating)
business_ratings = data.map(lambda line: line.split(',')).map(lambda fields: (fields[1], (fields[0], float(fields[2]))))
grouped_ratings = business_ratings.groupByKey()
business_user_ratings = grouped_ratings.map(lambda x: (x[0], dict(x[1])))
b_u_rating_dict = business_user_ratings.collectAsMap()
#bus average
bus_avg= train_rdd.map(lambda x: (x[1], x[2])) .groupByKey() .map(lambda x: (x[0], sum(x[1]) / len(x[1])))
bus_avg_rating = bus_avg.collectAsMap()

#test
validation_data = sc.textFile(test_path)
header_t = validation_data.first()
validation_data = validation_data.filter(lambda line: line != header_t)
val_rdd = validation_data.map(lambda row: row.split(",")).map(lambda row: (row[1], row[0]))
result_rdd = val_rdd.map(lambda row: (row[1], row[0], predict_score(row[0], row[1])))
head_line = ["user_id", "business_id", "prediction"]
result_list = result_rdd.collect()
item_based_df = pd.DataFrame(result_list, columns=["user_id", "business_id", "item_based_prediction"])


#model based


#preprocessing
#train
train_data = sc.textFile(folder+'yelp_train.csv')

header1 = train_data.first()
data = train_data.filter(lambda x: x != header1)
#train_rdd = data.map(lambda line: line.split(',')).map(lambda x: (x[0], x[1], float(x[2])))
#print(train_rdd.take(10))

#other file
bus_file = sc.textFile(folder+'business.json')

bus_rdd = bus_file.map(json.loads)
user_file = sc.textFile(folder+'user.json')
user_rdd = user_file.map(json.loads)
photo = folder + "photo.json"
photo_rdd = sc.textFile(photo).map(lambda line: json.loads(line)).map(lambda x: (x['business_id'], x['photo_id'], x['label']))


#merge to train

bus_rdd_cate = bus_rdd.filter(lambda x: x['categories'] is not None)
category_counts = bus_rdd_cate.flatMap(lambda x: x['categories'].split(', ')) \
                         .map(lambda category: (category, 1)) \
                         .reduceByKey(lambda a, b: a + b)

# Get top 5 most common categories
top5_most_common = category_counts.takeOrdered(5, key=lambda x: -x[1])

category_set = set()
for category, count in top5_most_common:
    category_set.add(category)




def filter_category(list_cat):
    if list_cat is None:
        return 'noCategory'
    # Split the input string of categories by comma
    categories = list_cat.split(', ')

    # Keep only the categories that are in the category_set
    filtered_categories = [category for category in categories if category in category_set]

    # Join the filtered categories back into a string
    filtered_categories_str = ','.join(filtered_categories)

    return filtered_categories_str


train_merge1 = train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).join(
        user_rdd.map(lambda x: (x['user_id'],(x['review_count'], x['average_stars'], x['fans'],  x['useful'], x['elite'])))
    )
train_merged = train_merge1.map(lambda x: (x[0], x[1][0][0], x[1][0][1],x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4]))

train_merge2 = train_merged.map(lambda x: (x[1], (x[0],x[2],x[3],x[4],x[5],x[6],x[7]))).join(
        bus_rdd.map(lambda x: (x['business_id'],(x['review_count'], x['latitude'], x['longitude'],  x['stars'], x['is_open'],x['attributes'].get('BusinessAcceptsCreditCards') if x['attributes'] is not None else 'False'
                                                 ,filter_category(x['categories'])))))
train_merged = train_merge2.map(lambda x: (x[0], x[1][0][0], x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],x[1][0][6],x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4],x[1][1][5],x[1][1][6]))

#add photo
bus_photo_num = photo_rdd.filter(lambda x: x[2] in ["food", "drink","inside","outside"]).map(lambda x: (x[0], 1)).reduceByKey(add)

merged_result = train_merged.map(lambda x: (x[0], (x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14]))).leftOuterJoin(bus_photo_num.map(lambda x: (x[0],(x[1]))))

train_merged = merged_result.map(lambda x: (x[0], x[1][0][0], x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],x[1][0][6],x[1][0][7],x[1][0][8],x[1][0][9],x[1][0][10],x[1][0][11],x[1][0][12],x[1][0][13],x[1][1] if x[1][1] is not None else 0 ))

train_df = pd.DataFrame(train_merged.collect(), columns=['bus_id', 'user_id', 'label', 'review_count_user', 'average_stars','fans', 'useful', 'elite', 'review_count_bus', 'latitude', 'longitude', 'bus_stars', 'is_open','ifCreditCard','category','photo_ct'])

# Perform one-hot encoding
encoded_category = train_df['category'].str.get_dummies(sep=',')


# Drop the original categorical column from the DataFrame

# Concatenate the one-hot encoded column to the DataFrame
train_df = pd.concat([train_df, encoded_category], axis=1)
train_df['elite'] = train_df['elite'].apply(lambda x: 1 if x != None else 0)
train_df['ifCreditCard'] = train_df['ifCreditCard'].apply(lambda x: 1 if x == 'True' else 0)

features = train_df.drop(columns=['bus_id', 'user_id', 'label','category','noCategory'])
labels = train_df['label']

dtrain = xgb.DMatrix(features, label=labels)

xgbr = xgb.XGBRegressor()

#hyper
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 6, 7],
}

# Perform grid search
grid_search = GridSearchCV(estimator=xgbr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(features, labels)

# Get the best parameters
best_params = grid_search.best_params_
print(best_params)

'''

params = {
    "objective": "reg:linear",
    "max_depth": 7,
    "n_estimators": 400,
    "learning_rate":0.06,
}
'''

# Train the XGBoost model
xgb_model = xgb.train(best_params, dtrain, num_boost_round=best_params["n_estimators"])


#predict
#test
validation_data = sc.textFile(test_path)
head2 = validation_data.first()
data2 = validation_data.filter(lambda x: x != head2)
val_rdd = data2.map(lambda line: line.split(',')).map(lambda x: (x[0], x[1]))

test_merge1 = val_rdd.map(lambda x: (x[0], x[1])).join(
        user_rdd.map(lambda x: (x['user_id'],(x['review_count'], x['average_stars'], x['fans'],  x['useful'], x['elite'])))
    )

test_merged = test_merge1.map(lambda x: (x[0], x[1][0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4]))

test_merge2 = test_merged.map(lambda x: (x[1], (x[0],x[2],x[3],x[4],x[5],x[6]))).join(
        bus_rdd.map(lambda x: (x['business_id'],(x['review_count'], x['latitude'], x['longitude'],  x['stars'], x['is_open'],x['attributes'].get('BusinessAcceptsCreditCards') if x['attributes'] is not None else 'False'
                                                 ,filter_category(x['categories'])))))
test_merged = test_merge2.map(lambda x: (x[0], x[1][0][0], x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],
                                           x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4],x[1][1][5],x[1][1][6]))

#merge photo
merged_result = test_merged.map(lambda x: (x[0], (x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13]))).leftOuterJoin(
    bus_photo_num.map(lambda x: (x[0],(x[1]))))

test_merged = merged_result.map(lambda x: (x[0], x[1][0][0], x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],x[1][0][6],x[1][0][7],x[1][0][8],x[1][0][9],x[1][0][10],x[1][0][11],x[1][0][12],x[1][1] if x[1][1] is not None else 0 ))


test_df = pd.DataFrame(test_merged.collect(), columns=['bus_id', 'user_id', 'review_count_user', 'average_stars','fans', 'useful', 'elite', 'review_count_bus', 'latitude', 'longitude', 'bus_stars', 'is_open','ifCreditCard','category','photo_ct'])

encoded_category = test_df['category'].str.get_dummies(sep=',')
test_df = pd.concat([test_df, encoded_category], axis=1)
test_df['elite'] = test_df['elite'].apply(lambda x: 1 if x != None else 0)
test_df['ifCreditCard'] = test_df['ifCreditCard'].apply(lambda x: 1 if x == 'True' else 0)


features = test_df.drop(columns=['bus_id', 'user_id','category','noCategory'])
dtest = xgb.DMatrix(features)

predictions = xgb_model.predict(dtest)
predictions_df = pd.DataFrame({'user_id': test_df['user_id'],'business_id': test_df['bus_id'],  'prediction': predictions})

#combined




combined_df = item_based_df.merge(predictions_df, on=['user_id', 'business_id'], how='inner')

w_item_based = 0.05
combined_predictions = (
    w_item_based * combined_df["item_based_prediction"] + (1-w_item_based) * combined_df["prediction"])


combined_df = pd.DataFrame({
    "user_id": item_based_df["user_id"],
    "business_id": item_based_df["business_id"],
    "prediction": combined_predictions
})

# Save the combined predictions to a CSV file
combined_df.to_csv('combined_predictions.csv', index=False)


end = time.time()
duration = str(end-start)
print("'Duration: " + duration + "'")

"""
with open(output_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(head_line)
    csv_writer.writerows(result_list)

"""









