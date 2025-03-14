import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from layers import Embedding, LSTM


class RNNClassifier(nn.Module):
    """A basic architecture including Embedding RNN and output layer"""

    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            **additional_kwargs
        }

        self.hparams = hparams

        self.embedding_layer = Embedding(
            num_embeddings=self.hparams["num_embeddings"],
            embedding_dim=self.hparams["embedding_dim"],
            padding_idx=0
        )

        self.rnn_layer = LSTM(
            input_size=self.hparams["embedding_dim"],
            hidden_size=self.hparams["hidden_size"],
        )

        self.final_linear_layer = nn.Linear(self.hparams["hidden_size"], 1)

        self.activation = nn.Sigmoid()


    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        sequence = self.embedding_layer(sequence)

        if lengths is not None:
            sequence = pack_padded_sequence(sequence, lengths, enforce_sorted=False)

        rnn_output, (hidden, cell) = self.rnn_layer(sequence)


        hidden = hidden[-1]
        output = self.activation(self.final_linear_layer(hidden))

        return output.squeeze(1)