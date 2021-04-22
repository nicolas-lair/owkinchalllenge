import torch
import torch.nn as nn


class MinMaxLayer(nn.Module):
    """
    MinMax layer returning the R max value and R min value
    """

    def __init__(self, R=5):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=R)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        neg_input = (-1) * inputs
        top_instances = torch.sort(self.pooling(inputs), descending=True)[0]
        neg_evidence = (-1) * torch.sort(self.pooling(neg_input))[0]
        output = torch.cat((top_instances, neg_evidence), dim=-1)
        return output.squeeze()


class ChowderModel(nn.Module):
    """
    Define a Chowder model based on the paper
    CLASSIFICATION AND D ISEASE L OCALIZATION IN HISTOPATHOLOGY USING ONLY GLOBAL LABELS : A WEAKLY-SUPERVISED APPROACH
    https://arxiv.org/pdf/1802.02212.pdf
    """

    def __init__(self, R=10):
        """

        :param R: Number of min and max value to keep for a slide in the minmax layer
        """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=False),
            # nn.ReLU(),
        )
        self.minmax = MinMaxLayer(R=R)

        self.mlp = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(2 * R, 200),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.projector(inputs)
        x = self.minmax(x)
        x = self.mlp(x)
        return x.squeeze()


class EnsembleChowder(nn.Module):
    """
    Ensemble model containing a list of E Chowder models
    """

    def __init__(self, E=10, R=5):
        """

        :param E: Number of chowder models
        :param R: init params for ChowderModel
        """
        super().__init__()
        self.model_list = nn.ModuleList(ChowderModel(R=R) for _ in range(E))

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
        :param inputs:  tensor of dimension (N, 1000, 2048)
        :return: tensor of dimension (N, 1)
        """
        return self(inputs).mean(dim=-1)
