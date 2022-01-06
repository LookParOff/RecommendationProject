"""
Calculation of recommendations
"""
import numpy as np
import pandas as pd
from time import time


np.seterr(all='raise')


def workWithBook(threshold) -> (pd.DataFrame, pd.DataFrame):
    rawBooks = pd.read_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\book rate rec/BX-Book-Ratings.csv", sep=";")
    # rawBooks = rawBooks[rawBooks["User-ID"].isin(users["User-ID"])]
    rawBooks = rawBooks[rawBooks["Book-Rating"] > 0]  # drop the useless zeros and nan
    countOfEveryBook = pd.value_counts(rawBooks["ISBN"])
    countOfEveryBook = countOfEveryBook[countOfEveryBook > threshold].index
    bRate = rawBooks[rawBooks["ISBN"].isin(countOfEveryBook)]  # filter, so books at least 10 people read

    bDescribe = pd.read_csv(r"C:\Users\Norma\PycharmProjects\Machine Learning\Datasets\book rate rec/BX_Books.csv", sep=";")
    bDescribe = bDescribe[["ISBN", "Book-Title"]]
    bDescribe = bDescribe[bDescribe["ISBN"].isin(bRate["ISBN"])]
    return bRate, bDescribe


def getPivotTable(threshold):
    t = time()
    booksRateALL, booksDescribeALL = workWithBook(threshold)
    booksRate, booksDescribe = booksRateALL[:], booksDescribeALL[:]
    # now we make matrix with rowIndex is User-ID and colIndex is ISBN. On intersection- rate
    print(f"Preprocess of data is over in {time() - t}secs")
    t = time()
    matrixOfRating = pd.pivot_table(booksRate, values="Book-Rating", index="User-ID", columns="ISBN")
    print("mean", np.nanmean(booksRate["Book-Rating"]), "median", np.nanmedian(booksRate["Book-Rating"]))
    # instead of isbn there are will be the title of books:
    newColumns = []
    for col in matrixOfRating.columns:
        newColumns.append(col)
    newColumns = pd.Series(newColumns).replace(booksDescribe["ISBN"].to_list(), booksDescribe["Book-Title"].to_list())
    matrixOfRating.columns = newColumns
    # keep only books with proper names:
    matrixOfRating = matrixOfRating.loc[:, matrixOfRating.columns.isin(booksDescribe["Book-Title"])]
    print(f"Pivot matrix is done in {time() - t} secs")
    t = time()
    # union the duplicate books
    matrixOfRating = matrixOfRating.groupby(by=matrixOfRating.columns, axis=1).max()
    matrixOfRating.fillna(0, inplace=True)
    print(f"Duplicates is gone in {time() - t} secs")
    t = time()
    # matrixOfRating:
    #           title title2 ...
    # User-ID     6     2
    # 10         Nan    0
    # 12          10    2
    # ...
    # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    # Lord of Chaos (The Wheel of Time, Book 6)
    return matrixOfRating


def savePivotTable(pivotTable: pd.DataFrame):
    rateRows, rateCols = np.where(pivotTable != 0)
    temp = []
    for row, col in zip(rateRows, rateCols):
        temp.append([row, col, pivotTable.iloc[row, col]])
    savedPivotTable = pd.DataFrame(temp)
    savedPivotTable.columns = ["rowIndex", "colIndex", "rate"]
    print(savedPivotTable)
    savedTitles = pd.DataFrame(pivotTable.columns)
    savedPivotTable.to_json("../RecommendationProject/Datasets/Book/not-sparse-ratings.json")
    savedTitles.to_json("../RecommendationProject/Datasets/Book/titles.json")
    return savedPivotTable, savedTitles


def loadPivotTable():
    ratings = pd.read_json("../RecommendationProject/Datasets/Book/not-sparse-ratings.json")
    titles = pd.read_json("../RecommendationProject/Datasets/Book/titles.json").sort_index()
    pivotTable = pd.pivot_table(ratings, values="rate", index="rowIndex", columns="colIndex")
    pivotTable.columns = titles.iloc[:, 0]
    return pivotTable


def getAllTitles(matOfRate):
    """
    :return: dataFrame, which contain book information and book's order based on popularity
    """
    popularBooks = matOfRate.sum().sort_values(ascending=False)
    popularBooks = pd.DataFrame(popularBooks)
    popularBooks.reset_index(level=0, inplace=True)
    popularBooks.columns = ["Book-Title", "Sum-Rate"]

    dopInfOfBooks = pd.read_csv("../Datasets/book rate rec/BX_Books.csv", sep=";")
    df1 = dopInfOfBooks[dopInfOfBooks["Book-Title"].isin(popularBooks["Book-Title"])][
        ['ISBN', "Book-Title", 'Book-Author', 'Image-URL-M', 'Image-URL-S']]
    df2 = df1[~df1["Book-Title"].duplicated()]
    df3 = df2.merge(popularBooks)
    df3.sort_values(by="Sum-Rate", inplace=True, ascending=False)
    df3.drop("Sum-Rate", axis=1, inplace=True)
    # df3.to_json("D:\\output.json", orient="table")
    return df3


def correlationRecommendation(ratingDf, title):
    """
    by one title return top 5 of most correlation
    :param ratingDf:
    :param title:
    :return:
    """
    book = ratingDf.loc[:, title]
    recs = ratingDf.corrwith(book).sort_values(ascending=False)
    # print(recs[1:5].index)
    return recs[1:5].index


def getRecommendation(pivotTableDF: pd.DataFrame, rate: np.array):
    """
    :returns list of the most recommended books
    """
    rateIndexes = np.where(rate >= 7)[0]
    readBooks = pivotTableDF.columns[rateIndexes].to_numpy()
    print("Read books:", readBooks)

    t = time()
    recommendedItems = np.array([])
    for index in rateIndexes:
        title = pivotTableDF.columns[index]
        recommendedItems = np.append(recommendedItems, correlationRecommendation(pivotTableDF, title))
    print("Recs was formed in", round(time() - t, 2), "secs")

    recommendedItems = recommendedItems[~np.isin(recommendedItems, readBooks)]
    print("Recommended books", recommendedItems)
    print()
    return recommendedItems


if __name__ == "__main__":
    # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback)) 2093
    # Lord of Chaos (The Wheel of Time, Book 6)

    # pivTabDF = getPivotTable(7)
    # savePivotTable(pivTabDF)

    loadedPivTabDF = loadPivotTable().fillna(0)
    recses = getRecommendation(loadedPivTabDF, loadedPivTabDF.loc[1463, :])
