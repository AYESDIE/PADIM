import argparse
import matplotlib.pyplot
import numpy
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import random
import scipy
import torch
import tqdm
from src.DataManager import DatasetManager
from src.models.PADIM import PADIMModule
from src.ImageBrowserManager import ImageBrowserManager
from src.TestingManager import TestingManager
from src.TrainingManager import TrainingManager

def train(dataloader: torch.utils.data.DataLoader , model_dir: str, UUID: str):
    embedding_index = torch.tensor(random.sample(range(0, 448), 100))
    model = PADIMModule("cpu", path = model_dir, is_training = True, UUID = UUID, 
                        embedding_index = embedding_index)
    for x in tqdm.tqdm(dataloader):
        _ = model(x)
    model.calculate_multivariate_gaussian()
    model.save()

def test(dataloader: torch.utils.data.DataLoader , model_dir: str, UUID: str):
    model = PADIMModule("cpu", path = model_dir, is_training = False, UUID = UUID)
    model.load()

    _x = []
    for x in tqdm.tqdm(dataloader):
        _ = model(x)
        _x.extend(x.numpy())
    
    embedding_vector = model.get_embedding_vector()
    B, C, W = embedding_vector.size()
    dist_list = []
    for i in range(W):
        mean = model.get_mean()[:, i]
        covariance_inverse = numpy.linalg.inv(model.get_covariance()[:, :, i].numpy())
        dist = [scipy.spatial.distance.mahalanobis(ev[:, i], mean, covariance_inverse) for ev in embedding_vector]
        dist_list.append(dist)

    dist_list = numpy.array(dist_list).transpose(1, 0).reshape(B, 64, 64)
    dist_list = torch.tensor(dist_list)
    score_map = torch.nn.functional.interpolate(dist_list.unsqueeze(1), size = x.size(2), mode = 'bilinear',
                                        align_corners = False).squeeze().numpy()
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    if not os.path.isdir(f"{model_dir}/viz"):
        os.makedirs(f"{model_dir}/viz")

    for i in range(len(scores)):
        fig, ax = matplotlib.pyplot.subplots(1, 2)
        denorm_test = (((_x[i].transpose(1, 2, 0) * numpy.array([0.229, 0.224, 0.225])) + numpy.array([0.485, 0.456, 0.406])) * 255).astype(numpy.uint8)
        ax[0].imshow(denorm_test)
        ax[1].imshow(255 * scores[i])
        fig.savefig(f"{model_dir}/viz/{i:02}")


class ModelManager(QWidget):
    SIGNAL_training_complete = pyqtSignal()
    SIGNAL_testing_complete = pyqtSignal()
    def __init__(self):
        super(ModelManager, self).__init__()
        self.main_layout = QHBoxLayout()
        self.training_manager = TrainingManager()
        self.testing_manager = TestingManager()
        #self.image_manager = ImageBrowserManager()
        self.main_layout.addWidget(self.training_manager)
        self.training_manager.SIGNAL_start_training.connect(self.start_training)
        self.testing_manager.SIGNAL_start_testing.connect(self.start_testing)
        self.SIGNAL_training_complete.connect(self.set_testing_mode)
        self.SIGNAL_testing_complete.connect(self.set_image_mode)
        self.model_path = None
        self.UUID = None
        self.setLayout(self.main_layout)
    
        self.raise_()
        self.showFullScreen()

    def start_training(self, data_path, model_path, UUID):
        self.UUID = UUID
        self.model_path = model_path
        dataset = DatasetManager(data_path)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
        train(data_loader, model_path, UUID)
        print("Training complete")
        self.SIGNAL_training_complete.emit()
    

    def set_testing_mode(self):
        self.main_layout.removeWidget(self.training_manager)
        self.training_manager.close()
        self.main_layout.addWidget(self.testing_manager)

    
    def start_testing(self, data_path):
        dataset = DatasetManager(data_path)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
        test(data_loader, self.model_path, self.UUID)
        self.SIGNAL_testing_complete.emit()

    
    def set_image_mode(self):
        self.image_manager = ImageBrowserManager(self.model_path + "/viz")
        self.main_layout.removeWidget(self.testing_manager)
        self.testing_manager.close()
        self.main_layout.addWidget(self.image_manager)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--UUID", type=str, required=True)
    argparser.add_argument("--train", action='store_true')
    argparser.add_argument("--test", action='store_true')
    args = argparser.parse_args()

    if args.train:
        embedding_index = torch.tensor(random.sample(range(0, 448), 100))
        model = PADIMModule("cpu", path = args.model_path, is_training = True, UUID = args.UUID, 
                            embedding_index = embedding_index)
        dataset = DatasetManager(args.data_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
        for x in tqdm.tqdm(dataloader):
            _ = model(x)
        model.calculate_multivariate_gaussian()
        model.save()
    
    if args.test:
        model = PADIMModule("cpu", path = args.model_path, is_training = False, UUID = args.UUID)
        model.load()
        dataset = DatasetManager(args.data_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
        _x = []
        for x in tqdm.tqdm(dataloader):
            _ = model(x)
            _x.extend(x.numpy())
        
        embedding_vector = model.get_embedding_vector()
        B, C, W = embedding_vector.size()
        dist_list = []
        for i in range(W):
            mean = model.get_mean()[:, i]
            covariance_inverse = numpy.linalg.inv(model.get_covariance()[:, :, i].numpy())
            dist = [scipy.spatial.distance.mahalanobis(ev[:, i], mean, covariance_inverse) for ev in embedding_vector]
            dist_list.append(dist)

        dist_list = numpy.array(dist_list).transpose(1, 0).reshape(B, 64, 64)
        dist_list = torch.tensor(dist_list)
        score_map = torch.nn.functional.interpolate(dist_list.unsqueeze(1), size = x.size(2), mode = 'bilinear',
                                            align_corners = False).squeeze().numpy()
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        if not os.path.isdir(f"{args.model_path}/viz"):
            os.makedirs(f"{args.model_path}/viz")

        for i in range(len(scores)):
            fig, ax = matplotlib.pyplot.subplots(1, 2)
            denorm_test = (((_x[i].transpose(1, 2, 0) * numpy.array([0.229, 0.224, 0.225])) + numpy.array([0.485, 0.456, 0.406])) * 255).astype(numpy.uint8)
            ax[0].imshow(denorm_test)
            ax[1].imshow(255 * scores[i])
            fig.savefig(f"{args.model_path}/viz/{i:02}")

