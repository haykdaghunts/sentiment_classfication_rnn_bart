import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence
)


def pack_outputs(state_seq, lengths):
    # Select the last states just before the padding
    last_indices = lengths - 1
    final_states = []
    for b, t in enumerate(last_indices.tolist()):
        final_states.append(state_seq[t, b])
    state = torch.stack(final_states).unsqueeze(0)

    # Packing the final state_seq (h_seq, c_seq e.t.c.)
    state_seq = pack_padded_sequence(state_seq, lengths, enforce_sorted=False)

    return state_seq, state


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_f = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.U_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)


        self.W_i = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.U_i = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.W_o = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.U_o = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.W_c = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.U_c = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Building a one layer LSTM with an activation with the attributes
        # defined above and a forward function below using the nn.Linear()                                 
        # Initialise h and c as 0 if these values are not given.

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        # Below code handles the batches with varying sequence lengths
        lengths = None
        if isinstance(x, PackedSequence):
            x, lengths = pad_packed_sequence(x)

        # State initialization
        state_size = (1, x.size(1), self.hidden_size)
        if h is None:
            h = torch.zeros(state_size, device=x.device, dtype=x.dtype)
        if c is None:
            c = torch.zeros(state_size, device=x.device, dtype=x.dtype)
        assert state_size == h.shape == c.shape

        # Filling the following lists and converting them to tensors
        h_seq = []
        c_seq = []

        for xt in x.unbind(0):
            ft = torch.sigmoid(self.W_f(xt) + self.U_f(h))
            it = torch.sigmoid(self.W_i(xt) + self.U_i(h))
            ot = torch.sigmoid(self.W_o(xt) + self.U_o(h))
            c = ft * c + it * torch.tanh(self.W_c(xt) + self.U_c(h))
            h = ot * torch.tanh(c)
            h_seq.append(h)
            c_seq.append(c)

        h_seq = torch.cat(h_seq, 0)
        c_seq = torch.cat(c_seq, 0)

        # Handling the padding stuff
        if lengths is not None:
            h_seq, h = pack_outputs(h_seq, lengths)
            c_seq, c = pack_outputs(c_seq, lengths)

        return h_seq, (h, c)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        """
        Inputs:
        - num_embeddings: Number of embeddings
        - embedding_dim: Dimension of embedding outputs
        - pad_idx: Index used for padding (i.e. the <eos> id)
        
        self.weight stores the vectors in the embedding space for each word in our vocabulary.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Handling  padding
        self.padding_idx = padding_idx
        self.register_buffer(
            'padding_mask',
            (torch.arange(0, num_embeddings) != padding_idx).view(-1, 1)
        )

        self.weight = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim))


        self.weight.data[padding_idx] = 0

    def forward(self, inputs):
        """
        Inputs:
            inputs: A long tensor of indices of size (seq_len, batch_size)
        Outputs:
            embeddings: A float tensor of size (seq_len, batch_size, embedding_dim)
        """

        # Ensuring <eos> always return zeros
        # and padding gradient is always 0
        weight = self.weight * self.padding_mask

        embeddings = weight[inputs]

        return embeddings