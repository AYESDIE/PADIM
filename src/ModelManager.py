import argparse
import matplotlib.pyplot
from models.PADIM import PADIMModule
from models.DataManager import DatasetManager
import numpy
import os
import random
import scipy
import torch
import tqdm

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

