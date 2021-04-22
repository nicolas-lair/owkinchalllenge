import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class MinMaxLayer(nn.Module):
    """
    MinMax layer returning the R max value and R min value
    """

    def __init__(self, R=5):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=R)

    def forward(self, inputs, lengths=None):
        if lengths is not None:
            inputs = [inputs[i][:lengths[i]].unsqueeze(0).unsqueeze(0) for i in range(len(lengths))]
            neg_inputs = [(-1) * x for x in inputs]
            output = []
            for input_, neg_ in zip(inputs, neg_inputs):
                top_instance = self.pooling(input_).squeeze()
                neg_instance = (-1) * self.pooling(neg_).squeeze()
                output.append(torch.cat([top_instance, neg_instance]).sort().values)
            return torch.stack(output, dim=0)
        else:
            raise NotImplementedError


class WeldonModel(nn.Module):
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
    """

    def __init__(self, R=10):
        """

        :param R: Number of min and max value to keep for a slide in the minmax layer
        """
        super().__init__(R=R)
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

    def aggregate_func(self, inputs):
        x = self.mlp(inputs)
        return x.squeeze()


class EnsembleModel(nn.Module):
    """
    Ensemble model containing a list of E Chowder models
    """

    def __init__(self, model_type, E=10, R=5):
        """

        :param E: Number of chowder models
        :param R: init params for ChowderModel
        """
        super().__init__()
        self.model_list = nn.ModuleList(model_type(R=R) for _ in range(E))

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
