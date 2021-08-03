from flask import Flask, render_template, request, jsonify

import numpy as np
import pandas as pd
import json
pd.options.mode.chained_assignment = None
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

full_df = pd.read_pickle('./static/data/austin_raunts_reviews_v2.pkl')
restaurants = pd.read_csv('./static/data/restaurants.csv', index_col=0)
feature_names = ["breakfast_True", "brunch_True", "lunch_True", "dinner_True", "OutdoorSeating_True", "OutdoorSeating_False", "NoiseLevel_'quiet'", "NoiseLevel_'average'", "NoiseLevel_'loud'", "NoiseLevel_'very_loud'", "casual_True", "casual_False"]

combined_business_data = full_df[['user_id','business_id','review_stars', 'date', 'name', 'address']]
combined_business_data_sub = combined_business_data.iloc[1:300000]

rating_crosstab = combined_business_data_sub.pivot_table(
    values='review_stars', 
    index='user_id', 
    columns='business_id', 
    fill_value=0
)

X = rating_crosstab.values.T
SVD = TruncatedSVD(n_components=12, random_state=17)
result_matrix = SVD.fit_transform(X)
corr_matrix = np.corrcoef(result_matrix)

restaurants_list = list(rating_crosstab.columns)
restaurants = restaurants[restaurants["business_id"].isin(restaurants_list)]
restaurant_features = restaurants[feature_names]

unique_restaurants = json.loads(restaurants.to_json(orient="records"))

similarities = None
user_2_similarities = None

scores = None
user_2_scores = None

@app.route('/', methods =["GET", "POST"])
def index():
    if request.method == "POST":
        meal = request.form.get("meal")
        outdoor = request.form.get("outdoor")
        noise = request.form.get("noise")
        formal = request.form.get("formal")

        new_user = restaurant_features.iloc[[0]]
        for col in new_user.columns:
            new_user[col].values[:] = 0

        if meal == "Breakfast":
            new_user["breakfast_True"] = 1
            new_user["brunch_True"] = 0.5
            new_user["lunch_True"] = 0.5
            new_user["dinner_True"] = 0.5
        elif meal == "Brunch":
            new_user["breakfast_True"] = 0.5
            new_user["brunch_True"] = 1
            new_user["lunch_True"] = 0.5
            new_user["dinner_True"] = 0.5
        elif meal == "Lunch":
            new_user["breakfast_True"] = 0.5
            new_user["brunch_True"] = 0.5
            new_user["lunch_True"] = 1
            new_user["dinner_True"] = 0.5
        elif meal == "Dinner":
            new_user["breakfast_True"] = 0.5
            new_user["brunch_True"] = 0.5
            new_user["lunch_True"] = 0.5
            new_user["dinner_True"] = 1

        if outdoor == "Outdoor":
            new_user["OutdoorSeating_True"] = 1
            new_user["OutdoorSeating_False"] = 0
        else:
            new_user["OutdoorSeating_True"] = 0.5
            new_user["OutdoorSeating_False"] = 0.5

        if noise == "Quiet":
            new_user["NoiseLevel_'quiet'"] = 1
            new_user["NoiseLevel_'average'"] = 0.75
            new_user["NoiseLevel_'loud'"] = 0.25
            new_user["NoiseLevel_'very_loud'"] = 0
        elif noise == "Average":
            new_user["NoiseLevel_'quiet'"] = 0.75
            new_user["NoiseLevel_'average'"] = 1
            new_user["NoiseLevel_'loud'"] = 0.75
            new_user["NoiseLevel_'very_loud'"] = 0.5
        elif noise == "Loud":
            new_user["NoiseLevel_'quiet'"] = 0.5
            new_user["NoiseLevel_'average'"] = 0.75
            new_user["NoiseLevel_'loud'"] = 1
            new_user["NoiseLevel_'very_loud'"] = 0.75
        elif noise == "Very Loud":
            new_user["NoiseLevel_'quiet'"] = 0
            new_user["NoiseLevel_'average'"] = 0.25
            new_user["NoiseLevel_'loud'"] = 0.75
            new_user["NoiseLevel_'very_loud'"] = 1

        if formal == "Casual":
            new_user["casual_True"] = 1
            new_user["casual_False"] = 0
        else:
            new_user["casual_True"] = 0
            new_user["casual_False"] = 1

        recs = get_recommendations_for_user(new_user)
        global similarities
        global user_2_similarities

        if similarities is None:
            similarities = get_similarities_for_user(new_user)
        else:
            user_2_similarities = get_similarities_for_user(new_user)

        return jsonify(recs)

    return render_template('index.html')

@app.route('/submit-swipes', methods =["POST"])
def submit_swipes():
    right_swipes = request.form.get("rightSwipes")
    right_swiped_ids = json.loads(right_swipes)

    left_swipes = request.form.get("leftSwipes")
    left_swipe_ids = json.loads(left_swipes)

    final_recs = get_collaborative_recs(left_swipe_ids, right_swiped_ids)

    return jsonify(final_recs)

def get_recommendations_for_user(user):
    user_similarities = cosine_similarity(user, restaurant_features)[0]
    similarities_sorted = sorted(enumerate(user_similarities), reverse=True, key=lambda x: x[1])
    similarities_sorted_indices = [pair[0] for pair in similarities_sorted]
    top_recs = [unique_restaurants[i] for i in similarities_sorted_indices if unique_restaurants[i][" Fast Food"] == 0]

    return top_recs

def get_similarities_for_user(user):
    user_similarities = cosine_similarity(user, restaurant_features)[0]

    return user_similarities

def score_helper(similarities, left_swipe_ids, right_swiped_ids):
    swipe_summation = np.zeros(len(corr_matrix[0]))

    for id in right_swiped_ids:
        index = restaurants_list.index(id)
        swipe_summation = swipe_summation + corr_matrix[index]

    for id in left_swipe_ids:
        index = restaurants_list.index(id)
        swipe_summation = swipe_summation - corr_matrix[index]

    for index, _ in enumerate(restaurants_list):
        swipe_summation[index] += similarities[index]

    return swipe_summation

def get_collaborative_recs(left_swipe_ids, right_swiped_ids):
    global similarities
    global user_2_similarities
    global scores
    
    if user_2_similarities is not None:
        user_2_scores = score_helper(user_2_similarities, left_swipe_ids, right_swiped_ids)
        merged_scores = [scores[index] + user_2_scores[index] for index, _ in enumerate(restaurants_list)]
        user_2_scores_sorted = [unique_restaurants[index] for index, _ in sorted(enumerate(user_2_scores), reverse=True, key=lambda x: x[1])]
        merged_scores_sorted = [unique_restaurants[index] for index, _ in sorted(enumerate(merged_scores), reverse=True, key=lambda x: x[1])]
        return [user_2_scores_sorted, merged_scores_sorted]

    scores = score_helper(similarities, left_swipe_ids, right_swiped_ids)
    scores_sorted = [unique_restaurants[index] for index, _ in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]
    
    return [scores_sorted]
