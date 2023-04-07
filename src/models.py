import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class MLP(nn.Module):
    def __init__(self, name, params):
        super(MLP, self).__init__()
        self.model_name = name
        self.params = params

        num_features = [nn.utils.weight_norm(nn.Linear(1, 8)) for i in range(8)]
        if name == "MLP_FSIW":
            print("using elapse feature")
            num_features.append(nn.utils.weight_norm(nn.Linear(1, 8)))
        cate_features = [nn.utils.weight_norm(nn.EmbeddingBag(cate_bin_size[i - 8], 8, mode='mean')) for i in
                         range(8, 17)]
        all_features = num_features + cate_features

        self.feature_layer = nn.Sequential(*all_features)

        self.fc1 = nn.utils.weight_norm(nn.Linear(sum(cate_bin_size) + 8, 256), dim=None)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.utils.weight_norm(nn.Linear(256, 256), dim=None)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.utils.weight_norm(nn.Linear(256, 128), dim=None)
        self.bn3 = nn.BatchNorm1d(128)

        print("build model {}".format(name))
        if self.model_name == "MLP_EXP_DELAY":
            self.fc4 = nn.Linear(128, 2)
        elif self.model_name == "MLP_tn_dp":
            self.fc4 = nn.Linear(128, 2)
        elif self.model_name in ["MLP_SIG", "MLP_FSIW"]:
            self.fc4 = nn.Linear(128, 1)
        else:
            raise ValueError("model name {} not exist".format(name))

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        if self.model_name == "MLP_EXP_DELAY":
            return {"logits": x[:, 0].view(-1, 1), "log_lamb": x[:, 1].view(-1, 1)}
        elif self.model_name in ["MLP_SIG", "MLP_FSIW"]:
            return {"logits": x}
        elif self.model_name == "MLP_tn_dp":
            return {"tn_logits": x[:, 0].view(-1, 1), "dp_logits": x[:, 1].view(-1, 1)}
        else:
            raise NotImplementedError()

    def predict(self, x):
        return self.forward(x)["logits"]


def get_model(name, params):
    if name in ["MLP_EXP_DELAY", "MLP_SIG", "MLP_tn_dp", "MLP_FSIW"]:
        return MLP(name, params)
    else:
        raise NotImplementedError()
