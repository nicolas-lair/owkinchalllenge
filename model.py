import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from config import baseCONFIG


class MinMaxLayer(nn.Module):
    """
    MinMax layer returning the R max value and R min value
    """

    def __init__(self, R=5):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=R)

    def forward(self, inputs, lengths=None):
        """

        :param inputs: tensor of size N, 1000
        :param lengths: list of size N tracking the number of tiles for each input data
        :return: tensor of size N, 2*R
        """
        if lengths is not None:
            # Split the batch and remove the padded values
            inputs = [inputs[i][:lengths[i]].unsqueeze(0).unsqueeze(0) for i, length_ in enumerate(lengths)]
            neg_inputs = [(-1) * x for x in inputs]
            output = []

            # Compute top_instance and negative evidence for each data in the batch
            for input_, neg_ in zip(inputs, neg_inputs):
                top_instance = self.pooling(input_).squeeze()
                neg_instance = (-1) * self.pooling(neg_).squeeze()

                # Concat, sort and store the top instance and neg evidenve
                output.append(torch.cat([top_instance.view(-1), neg_instance.view(-1)]).sort().values)
            return torch.stack(output, dim=0)
        else:
            raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(baseCONFIG.dropout[0]),
            nn.Linear(in_features, 200),
            nn.Sigmoid(),
            nn.Dropout(baseCONFIG.dropout[1]),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Dropout(baseCONFIG.dropout[2]),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        if baseCONFIG.batchnorm:
            self.mlp = nn.Sequential(nn.BatchNorm1d(num_features=in_features), self.mlp)

    def forward(self, inputs):
        return self.mlp(inputs)


class WeldonModel(nn.Module):
    """
    This implements the WeldonModel as describes in the paper :
    CLASSIFICATION AND DISEASE LOCALIZATION IN HISTOPATHOLOGY USING ONLY GLOBAL LABELS : A WEAKLY-SUPERVISED APPROACH
    https://arxiv.org/pdf/1802.02212.pdf

    It serves as parent class for the chowder model
    """

    def __init__(self, R=10):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=False),
            # nn.ReLU(),
        )
        self.minmax = MinMaxLayer(R=R)

    def compute_features(self, inputs):
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        x = self.projector(x).squeeze()
        x = self.minmax(x, lengths=lengths)
        return x

    def aggregate_func(self, inputs):
        x = inputs.sum(dim=1)
        x = torch.sigmoid(x)
        return x

    def forward(self, inputs):
        x = self.compute_features(inputs)
        output = self.aggregate_func(x)
        return output


class ChowderModel(WeldonModel):
    """
    Define a Chowder model based on the paper
    CLASSIFICATION AND DISEASE LOCALIZATION IN HISTOPATHOLOGY USING ONLY GLOBAL LABELS : A WEAKLY-SUPERVISED APPROACH
    https://arxiv.org/pdf/1802.02212.pdf

    It is based on the WeldonModel class
    """

    def __init__(self, R=10):
        """

        :param R: Number of min and max value to keep for a slide in the minmax layer
        """
        super().__init__(R=R)
        self.mlp = MLP(in_features=2 * R)

    def aggregate_func(self, inputs):
        x = self.mlp(inputs)
        return x.squeeze()


class EnsembleModel(nn.Module):
    """
    Ensemble model containing a list of E models (Weldon or Chowder)
    """

    def __init__(self, model_type=ChowderModel, E=10, **kwargs):
        """

        :param E: Number of chowder models
        :param R: init params for ChowderModel
        """
        super().__init__()
        self.model_list = nn.ModuleList(model_type(**kwargs) for _ in range(E))

    def forward(self, inputs):
        """
        Return the output the different Chowder model in the last dimension without averaging
        Useful for training each model
        :param inputs: tensor of dimension (N, 1000, 2048)
        :return: tensor of dimension (N, R)
        """
        predictions = []
        for m in self.model_list:
            predictions.append(m(inputs))
        return torch.stack(predictions, dim=1)

    def predict(self, inputs):
        """
        Returns the average of the predictions of the different Chowder models.
        Useful for inference
        :param inputs: tensor of dimension (N, 1000, 2048)
        :return: tensor of dimension (N, 1)
        """
        return self(inputs).mean(dim=1)

    @classmethod
    def from_model_list(cls, model_list):
        # Dumb initialization
        model = cls(ChowderModel)

        # Replace model_list by the one provided
        model.model_list = nn.ModuleList(model_list)
        return model


class DeepSetChowder(nn.Module):
    def __init__(self, scaler_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=scaler_size),
            nn.ReLU(),
        )
        self.mlp = MLP(in_features=scaler_size)

    def forward(self, inputs):
        x, lengths = pad_packed_sequence(inputs, batch_first=True)
        x = self.projector(x)
        x = torch.sum(x, dim=1)
        x = x.permute(1, 0) / lengths.to(x.device)
        x = x.permute(1, 0)
        x = self.mlp(x)
        return x.squeeze()
