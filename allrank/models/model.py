import json
from typing import Dict

import torch.nn as nn
from attr import asdict

from allrank.config import PositionalEncoding, TransformerConfig
from allrank.models.transformer import make_transformer, Encoder
from allrank.utils.python_utils import instantiate_class


def first_arg_id(x, *y):
    return x


class OutputLayer(nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """
    def __init__(self, d_model, d_output, output_activation=None):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = nn.Identity() if output_activation is None else instantiate_class(
            "torch.nn.modules.activation", output_activation)
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        return self.activation(self.w_1(x).squeeze(dim=2))

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)

class FCModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """
    def __init__(self, sizes, input_norm, activation, dropout, n_features):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        self.sizes = sizes
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


class LTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """
    def __init__(self, input_layer: FCModel, encoder:Encoder, output_layer: OutputLayer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        return self.encoder(self.input_layer(x), mask, indices)

    def forward(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """
        return self.output_layer(self.prepare_for_output(x, mask, indices))

    def score(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))

    def serialize_params(self):
        return {
            "n_features": self.input_layer.sizes[0],
            "fc_model": {
                "sizes": self.input_layer.sizes[1:],
                "input_norm": not isinstance(self.input_layer.input_norm, nn.Identity),  # True if not Identity, False otherwise,
                "activation": self.input_layer.activation.__class__.__name__,
                "dropout": self.input_layer.dropout.p,
            },
            "transformer": None if self.encoder is None else {
                "N": self.encoder.N,
                "d_ff": self.encoder.d_ff,
                "h": self.encoder.h,
                "positional_encoding": None if self.encoder.positional_encoding is None else {
                    'strategy': self.encoder.positional_encoding.strategy,
                    'max_indices': self.encoder.positional_encoding.max_indices
                },
                "dropout": self.encoder.dropout,
            },
            "post_model": {
                "d_output": self.output_layer.d_output,
                "output_activation": self.output_layer.activation.__class__.__name__,
            }
        }

    @classmethod
    def create(cls, params: Dict):
        if params['transformer']['positional_encoding']:
            params['transformer']['positional_encoding'] = PositionalEncoding(**params['transformer']['positional_encoding'])
        params['transformer'] = TransformerConfig(**params['transformer'])

        return make_model(**params)

def make_model(fc_model, transformer, post_model, n_features):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    if fc_model:
        fc_model = FCModel(**fc_model, n_features=n_features)  # type: ignore
    d_model = n_features if not fc_model else fc_model.output_size
    if transformer:
        transformer = make_transformer(n_features=d_model, **asdict(transformer, recurse=False))  # type: ignore
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, **post_model))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
