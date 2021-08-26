# @File   : process_data.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2021/8/5,下午8:27

from config import ml_10m as config_ml
import pandas as pd
import numpy as np
import os

class Data:
    """
    This class is used to process three separate datasets
    and combine their information into a training set and a test set.
    """

    def __init__(self, config_ml):
        """
        Define the original parameters.

        Args:
            config_ml:A dictionary that stores the values of various parameters.
        """
        self.config = config_ml
        self.movies_path = os.path.abspath(self.config['movies_path'])
        self.ratings_path = os.path.abspath(self.config['ratings_path'])
        self.tags_path = os.path.abspath(self.config['tags_path'])
        self.movies_columns = self.config['movies_columns']
        self.ratings_columns = self.config['ratings_columns']
        self.tags_columns = self.config['tags_columns']
        # self.num_users = self.process_data()["UserID_idx"].max() + 1
        # self.num_movies = self.process_data()["MovieID_idx"].max() + 1
        # self.num_genres = self.process_data()["Genres_idx"].max() + 1
        # self.num_titles = self.process_data()["Title_idx"].max() + 1
        self.ratio = self.config['ratio']

    def process_data(self):
        """
        Import the data and save it in pandas format, add columns.

        Returns:
            data set, training set and test set.
        """
        pd_movies = pd.read_table(self.movies_path, sep="::", engine='python')
        pd_ratings = pd.read_table(self.ratings_path, sep="::", engine='python')
        pd_tags = pd.read_table(self.tags_path, sep="::", engine='python')
        pd_movies.columns = self.movies_columns
        pd_ratings.columns = self.ratings_columns
        pd_tags.columns = self.tags_columns

        # Specify the type in genres as the first type.
        for i in np.arange(len(pd_movies)):
            if not pd_movies.iloc[i, 2].isalpha():
                pd_movies.iloc[i, 2] = pd_movies.iloc[i, 2].split(sep='|')[0]

        # Remove the year from the movie title.
        for i in np.arange(len(pd_movies)):
            pd_movies.iloc[i, 1] = pd_movies.iloc[i, 1][:pd_movies.iloc[i, 1].index('(')]

        # will be regenerated with the following,
        # as their index values may not be consecutive,
        # resulting in an actual capacity smaller than the maximum number of serial numbers.
        data = pd.merge(pd_movies, pd_ratings, how="inner", on=['MovieID'])
        self.add_index_column(data,"MovieID")
        self.add_index_column(data,"UserID")
        self.add_index_column(data,"Title")
        self.add_index_column(data,"Genres")

        # ratings normalization 0~5->0~1.
        min_rating = data["ratings"].min()
        max_rating = data["ratings"].max()
        data["ratings"] = data["ratings"].map(lambda x: (x - min_rating) / (max_rating - min_rating))

        # Divide the training set from the test set
        # and keep the original data set.
        # train_set = data.sample(frac=0.1)
        # train_X = train_set[["UserID_idx", "MovieID_idx", "Title_idx", "Genres_idx"]]
        #
        # train_y = train_set.pop("ratings")
        return data

    def add_index_column(self,param_df, column_name):
        """
        Add a numeric index column to the column.
        The purpose: to prevent embedding too large.

        Args:
            @param_df:
            @column_name:
        """
        values = list(param_df[column_name].unique())
        value_index_dict = {value: idx for idx, value in enumerate(values)}
        param_df[f"{column_name}_idx"] = param_df[column_name].map(value_index_dict)

    def pd_diplay(self):
        """
        Omissions occur when printing pandas tables.
        The following code regulates the length of the maximum column.
        """
        pd.set_option('display.width', 1000)        # the table line will not appear in segments.
        pd.set_option('display.max_columns', None)  # Show all columns.
        pd.set_option('display.max_rows', None)     # Show all rows.

    def data(self):
        """
            Avoid reading data and processing it repeatedly every time you use it.
        """
        if (os.path.isfile("processed_data.csv")):
            print("processed_data.csv has finished")
        else:
            s = Data(config_ml)
            data = s.process_data()
            data.to_csv("processed_data.csv")
        processed_data = pd.read_csv("processed_data.csv")
        return processed_data

if __name__ == "__main__":
    # print(len(data))
    # s.pd_diplay()
    # print(data[1].head(10))
    # print(data[2].head(10))
    pass


