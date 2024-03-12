import numpy
import os
import torch
import torchvision

class PADIMModule(torch.nn.Module):
  # Public
  def __init__(self, device, path = None, is_training = False, embedding_index = None, UUID = None):
    super().__init__()
    # Settings
    self.device = device
    self.is_training = is_training
    self.embedding_index = embedding_index
    self.UUID = UUID
    if UUID == None:
      self.UUID = "PADIMModule_" + str(numpy.random.randint(0, 1000000))
    self.path = path + f"/{self.UUID}"

    # Private
    self.__model = torchvision.models.resnet18(pretrained = True, progress = True)
    self.__model.to(device)
    self.__model.eval()
    self.__model.layer1[-1].register_forward_hook(self.__forward_hook)
    self.__model.layer2[-1].register_forward_hook(self.__forward_hook)
    self.__model.layer3[-1].register_forward_hook(self.__forward_hook)

    self.__hidden_vector = None
    self.__mean = None
    self.__covariance = None
    self.__embedding_vector = None

    self.__embedding_vector_test = None


  def forward(self, x):
    with torch.no_grad():
      self.__hidden_vector = []
      _ = self.__model(x.to(self.device))
      _ = torch.cat((self.__hidden_vector[0],
                     self.__hidden_vector[1],
                     self.__hidden_vector[2]), 1)

      _ = torch.index_select(_, 1, self.embedding_index)

      if self.__embedding_vector == None:
        self.__embedding_vector = _
      else:
        self.__embedding_vector = torch.cat((self.__embedding_vector, _), 0)

      return self.__hidden_vector

  def reset_memory(self):
    self.__hidden_vector = None
    self.__embedding_vector = None

  def get_embedding_vector(self):
    return self.__embedding_vector

  def set_embedding_index(self, idx):
    self.embedding_index = idx

  def get_embedding_index(self, idx):
    return self.embedding_index

  def set_training(self, is_training):
    self.is_training = is_training

  def save(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    torch.save(self.embedding_index, self.path + f"/embeddingidx.pt")
    torch.save(self.__mean, self.path + f"/mean.pt")
    torch.save(self.__covariance, self.path + f"/covariance.pt")

  def save_embedding(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    torch.save(self.__embedding_vector, self.path + f"/embedding.pt")
    torch.save(self.embedding_index, self.path + f"/embeddingidx.pt")

  def load_embedding(self):
    self.__embedding_vector = torch.load(self.path + f"/embedding.pt")
    self.embedding_index = torch.load(self.path + f"/embeddingidx.pt")

  def load(self):
    self.embedding_index = torch.load(self.path + f"/embeddingidx.pt")
    self.__mean = torch.load(self.path + f"/mean.pt")
    self.__covariance = torch.load(self.path + f"/covariance.pt")

  def get_mean(self):
    return self.__mean

  def get_covariance(self):
    return self.__covariance

  def calculate_multivariate_gaussian(self):
    B, C, W = self.__embedding_vector.size()
    self.__mean = torch.mean(self.__embedding_vector, dim = 0).to(self.device)
    __covariance = torch.zeros(C, C, W).to(self.device).numpy()
    I = numpy.identity(C)
    for i in range(W):
      __covariance[:, :, i] = numpy.cov(self.__embedding_vector[:, :, i].numpy(), rowvar=False) + 0.01 * I
    self.__covariance = torch.tensor(__covariance)

  # Private
  def __forward_hook(self, module, input, output):
    B, C, H, W = output.size()
    self.__hidden_vector.append(torch.nn.functional.interpolate(output, size = (64, 64)).view(B, C, 64 * 64))