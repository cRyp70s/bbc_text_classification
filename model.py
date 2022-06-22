import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, embed_size, hidden_dim, num_layers, vocab_len, out_features, **kwargs
    ):
        super(Model, self).__init__(**kwargs)
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_len, embedding_dim=embed_size
        )
        self.encoder = nn.LSTM(embed_size, hidden_dim, num_layers, bidirectional=True)
        self.drop = nn.Dropout(0.6)
        self.decoder = nn.Linear(4 * hidden_dim, out_features)

    def forward(self, inputs):
        embeddings = self.drop(self.embeddings(inputs.T))

        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
