from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader
from dataset import make_dataloader

if __name__ == '__main__':
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = make_dataloader('dataset/data_train', 16, num_workers, True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))