"""
Sending recs to client
funcs index and recs handle site getters
"""
from time import time
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from recommendation_engine import load_pivot_table, get_recommendation, get_info_of_titles


app = Flask(__name__)
print("Loading data")
loaded_pivot_tab_df = load_pivot_table().fillna(0)
description_df = pd.read_json("../RecommendationProject/Datasets/Book/description-of-books.json")
print("Done!")


def vector_all_ratings(piv_tab, rates_titles):
    """
    from passed titles, those was rate, make vector with ratings of all titles.
    If user didn't rate some title it would be 0.
    :return:
    """
    rates_vector = np.zeros(piv_tab.columns.shape[0])
    rates_vector[np.isin(piv_tab.columns, rates_titles.columns)] = rates_titles.iloc[0, :]
    return rates_vector


@app.route("/")
def index():
    """
    just showing black
    :return:
    """
    return render_template("index.html")


@app.route("/recs", methods=['POST'])
def recs():
    """
    function handle answering on post request.
    input: json of ratings of man. output: json of recommended books, like jsons for Andrew
    :return:
    """
    start_time = time()
    print("I in recs. Ready to catch your request")
    input_json = request.get_json()  # force=True
    print("I caught. And I store request, here it is:\n", input_json)
    print(type(input_json))
    input_df = pd.DataFrame(input_json, index=[0])
    print("DataFrame:", input_df)
    rates = vector_all_ratings(loaded_pivot_tab_df, input_df)
    print("Rates:", rates)
    titles = get_recommendation(loaded_pivot_tab_df, rates)
    print("Titles:", titles)
    output_df = get_info_of_titles(titles, description_df)
    print("output_df:", output_df)
    output_json = output_df.to_json(orient="table")
    print(f"Done in {round(time() - start_time, 1)} secs")
    return output_json
