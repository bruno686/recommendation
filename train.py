# @File   : train.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2021/8/13,下午3:20
import torch
from sa_model import model
from config import ml_10m as config_ml
from torch.utils import data
from process_data import Data
from sklearn.metrics import mean_absolute_error, mean_squared_error

class model_train():
    def __init__(self,config_ml):
        self.config = config_ml
        self.batch_size = self.config['batch_size']
        self.num_epoches = self.config['epoch']
        self.lr = self.config['learning_rate']
        self.fraction = self.config['fraction']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.full_dataset = Data(config_ml).data()[["UserID_idx", "MovieID_idx", "Title_idx", "Genres_idx","ratings"]]
        self.train_dataset_raw = self.full_dataset.sample(frac=self.fraction)
        self.test_dataset_raw = self.full_dataset[~self.full_dataset.index.isin(self.train_dataset_raw.index)]
        self.train_dataset = data.TensorDataset(torch.from_numpy(self.train_dataset_raw[["UserID_idx", "MovieID_idx", "Title_idx", "Genres_idx"]].values),torch.from_numpy(self.train_dataset_raw["ratings"].values))
        self.train_dataset_iter = data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
        self.test_x = torch.from_numpy(self.test_dataset_raw[["UserID_idx", "MovieID_idx", "Title_idx", "Genres_idx"]].values)
        self.test_y = torch.from_numpy(self.test_dataset_raw["ratings"].values)
        self.net = model(config_ml).to(device=self.device)  #First put the model on gpu.

    def train(self):
        loss = torch.nn.MSELoss()
        trainer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        for epoch in range(self.num_epoches):
            for indices, data_batch in enumerate(self.train_dataset_iter):
                X, y = data_batch
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                predict = self.net(X)
                predict = predict.to(torch.float32)
                y = y.to(torch.float32)
                l = loss(predict, y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
                if indices % 2000 == 1999:
                    print(f'epoch {epoch + 1},inices{indices + 1},loss{l:f}')

    def test(self):
        print("start testing")
        self.net = self.net.to('cpu')
        with torch.no_grad():
            y_predict = self.net(self.test_x)
            print("MAE:", mean_absolute_error(self.test_y, y_predict), "\tRMSE:",(mean_squared_error(self.test_y, y_predict) * 25) ** (1 / 2))

if __name__ == 'main':
    model_train = model_train(config_ml)
    model_train.train()