"""
Calculation of recommendations
"""
from time import time
import numpy as np
import pandas as pd


np.seterr(all='raise')


def work_with_book(threshold) -> (pd.DataFrame, pd.DataFrame):
    """
    Clean data
    :param threshold:
    :return:
    """
    raw_books = pd.read_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning"
                            r"\Datasets\book rate rec/BX-Book-Ratings.csv", sep=";")
    raw_books = raw_books[raw_books["Book-Rating"] > 0]  # drop the useless zeros and nan
    count_of_every_book = pd.value_counts(raw_books["ISBN"])
    count_of_every_book = count_of_every_book[count_of_every_book > threshold].index
    # filter, so books at least 10 people read
    b_rate = raw_books[raw_books["ISBN"].isin(count_of_every_book)]
    b_describe = pd.read_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning"
                             r"\Datasets\book rate rec/BX_Books.csv", sep=";")
    b_describe = b_describe[["ISBN", "Book-Title"]]
    b_describe = b_describe[b_describe["ISBN"].isin(b_rate["ISBN"])]
    return b_rate, b_describe


def get_pivot_table(threshold):
    """
    return sparse pivot table of rates
    matrix_of_rating:
              title title2 ...
    User-ID     6     2
    10         Nan    3
    12          10    2
    ...
    """
    start_time = time()
    books_rate_all, booksDescribeALL = work_with_book(threshold)
    books_rate, booksDescribe = books_rate_all[:], booksDescribeALL[:]
    # now we make matrix with rowIndex is User-ID and colIndex is ISBN. On intersection- rate
    print(f"Preprocess of data is over in {time() - start_time}secs")
    start_time = time()
    matrix_of_rating = pd.pivot_table(books_rate, values="Book-Rating",
                                      index="User-ID", columns="ISBN")
    # instead of isbn there are will be the title of books:
    new_columns = []
    for col in matrix_of_rating.columns:
        new_columns.append(col)
    new_columns = pd.Series(new_columns).replace(booksDescribe["ISBN"].to_list(),
                                                 booksDescribe["Book-Title"].to_list())
    matrix_of_rating.columns = new_columns
    # keep only books with proper names:
    matrix_of_rating = \
        matrix_of_rating.loc[:, matrix_of_rating.columns.isin(booksDescribe["Book-Title"])]
    print(f"Pivot matrix is done in {time() - start_time} secs")
    start_time = time()
    # union the duplicate books
    matrix_of_rating = matrix_of_rating.groupby(by=matrix_of_rating.columns, axis=1).max()
    matrix_of_rating.fillna(0, inplace=True)
    print(f"Duplicates is gone in {time() - start_time} secs")

    # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    # Lord of Chaos (The Wheel of Time, Book 6)
    return matrix_of_rating


def save_pivot_table(pivotTable: pd.DataFrame):
    """
    saving pivot table to json
    :param pivotTable:
    :return:
    """
    rate_rows, rate_cols = np.where(pivotTable != 0)
    temp = []
    for row, col in zip(rate_rows, rate_cols):
        temp.append([row, col, pivotTable.iloc[row, col]])
    saved_pivot_table = pd.DataFrame(temp)
    saved_pivot_table.columns = ["rowIndex", "colIndex", "rate"]
    print(saved_pivot_table)
    saved_titles = pd.DataFrame(pivotTable.columns)
    saved_pivot_table.to_json("../RecommendationProject/Datasets/Book/not-sparse-ratings.json")
    saved_titles.to_json("../RecommendationProject/Datasets/Book/titles.json")
    return saved_pivot_table, saved_titles


def load_pivot_table():
    """
    load pivot table from json
    :return:
    """
    ratings = pd.read_json("../RecommendationProject/Datasets/Book/not-sparse-ratings.json")
    titles = pd.read_json("../RecommendationProject/Datasets/Book/titles.json").sort_index()
    pivot_table = pd.pivot_table(ratings, values="rate", index="rowIndex", columns="colIndex")
    pivot_table.columns = titles.iloc[:, 0]
    return pivot_table


def get_all_titles(matOfRate):
    """
    :return: dataFrame, which contain book information and book's order based on popularity
    """
    popular_books = matOfRate.sum().sort_values(ascending=False)
    popular_books = pd.DataFrame(popular_books)
    popular_books.reset_index(level=0, inplace=True)
    popular_books.columns = ["Book-Title", "Sum-Rate"]

    dop_inf_of_books = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    df1 = dop_inf_of_books[dop_inf_of_books["Book-Title"].isin(popular_books["Book-Title"])][
        ['ISBN', "Book-Title", 'Book-Author', 'Image-URL-M', 'Image-URL-S']]
    df2 = df1[~df1["Book-Title"].duplicated()]
    df3 = df2.merge(popular_books)
    df3.sort_values(by="Sum-Rate", inplace=True, ascending=False)
    df3.drop("Sum-Rate", axis=1, inplace=True)
    # df3.to_json("D:\\output.json", orient="table")
    return df3


def correlation_recommendation(ratingDf, title):
    """
    by one title return top 5 of most correlation
    :param ratingDf:
    :param title:
    :return:
    """
    book = ratingDf.loc[:, title]
    recs = ratingDf.corrwith(book).sort_values(ascending=False)
    return recs[1:5].index


def get_recommendation(pivotTableDF: pd.DataFrame, rate: np.array):
    """
    :returns list of the most recommended books
    """
    rate_indexes = np.where(rate >= 7)[0]
    read_books = pivotTableDF.columns[rate_indexes].to_numpy()
    print("Read books:", read_books)

    start_time = time()
    recommended_items = np.array([])
    for index in rate_indexes:
        title = pivotTableDF.columns[index]
        recommended_items = np.append(recommended_items,
                                      correlation_recommendation(pivotTableDF, title))
    print("Recs was formed in", round(time() - start_time, 2), "secs")

    recommended_items = recommended_items[~np.isin(recommended_items, read_books)]
    print("Recommended books", recommended_items)
    print()
    return recommended_items


if __name__ == "__main__":
    # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback)) 2093
    # Lord of Chaos (The Wheel of Time, Book 6)

    # piv_tab_df = getPivotTable(7)
    # save_pivot_table(piv_tab_df)

    loaded_pivot_tab_df = load_pivot_table().fillna(0)
    recses = get_recommendation(loaded_pivot_tab_df, loaded_pivot_tab_df.loc[1463, :])