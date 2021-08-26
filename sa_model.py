# @File   : sa_model.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2021/8/5,下午8:48
import pandas as pd
import torch
import process_data
from config import ml_10m as config_ml

class model(torch.nn.Module):
    def __init__(self, config_ml):
        super(model, self).__init__()
        self.config = config_ml
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_size = self.config['hidden_size']
        self.data = pd.read_csv('processed_data.csv')

        self.embedding_users = torch.nn.Embedding(
            num_embeddings=self.data['UserID_idx'].max()+1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_titles = torch.nn.Embedding(
            num_embeddings=self.data['Title_idx'].max()+1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_movies = torch.nn.Embedding(
            num_embeddings=self.data['MovieID_idx'].max()+1,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genres = torch.nn.Embedding(
            num_embeddings=self.data['Genres_idx'].max()+1,
            embedding_dim=self.embedding_dim
        )
        self.linear_users = torch.nn.Linear(self.embedding_dim,self.hidden_size)
        self.linear1_movies = torch.nn.Linear(self.embedding_dim*3,self.hidden_size)
        self.liner = torch.nn.Linear(self.hidden_size,1)

    def forward(self,X):
        users_idx  = X[:,0]
        movies_idx = X[:,1]
        titles_idx = X[:,2]
        genres_idx = X[:,3]

        users_emb = self.embedding_users(users_idx)
        movies_emb = self.embedding_movies(movies_idx)
        titles_emb = self.embedding_titles(titles_idx)
        genres_emb = self.embedding_genres(genres_idx)

        users = self.linear_users(users_emb)

        movies = torch.cat((titles_emb,movies_emb,genres_emb),1)
        movies = self.linear1_movies(movies)

        rat = movies*users
        rat = self.liner(rat)
        rat = rat.squeeze(dim=1)
        return rat

if __name__ == "__main__":
    sa = model(config_ml)
