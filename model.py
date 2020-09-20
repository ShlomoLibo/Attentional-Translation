from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d


class TwoWayGRUEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=encoder_hidden, bidirectional=True)

    def forward(self, x):
        output = self.embedding(x)
        output, hidden = self.gru(output)
        return output, hidden


class AttentionNetwork(nn.Module):
        def __init__(self, encoder_hidden, decoder_hidden, attn_hidden=32):
            super().__init__()
            self.decoder_hidden = decoder_hidden
            self.encoder_hidden = encoder_hidden
            self.linear1 = nn.Linear(decoder_hidden + 2 * encoder_hidden, attn_hidden)
            self.linear2 = nn.Linear(attn_hidden, 1)
            self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2, nn.Softmax(dim=1))

        def forward(self, decoder_hidden_state, encoder_output):
            decoder_hidden_state = decoder_hidden_state.expand(encoder_output.shape[0], -1, -1)
            attn_input = torch.cat([decoder_hidden_state, encoder_output], dim=2)
            attn_input = attn_input.permute(1, 0, 2)
            attn = self.model(attn_input)
            return attn


class GruAttentionDecoder(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, output_dim, embedding_dim, sos_token, device):
        super().__init__()
        self.decoder_hidden = decoder_hidden
        self.gru = nn.GRU(2 * encoder_hidden + embedding_dim, decoder_hidden)
        self.attn = AttentionNetwork(encoder_hidden, decoder_hidden)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.output_decoder = nn.Linear(decoder_hidden, output_dim)
        self.batch_norm = BatchNorm1d(output_dim)
        self.output_dim = output_dim
        self.device = device
        self.sos_token = sos_token

    def forward(self, encoder_output, max_length):
        output = torch.zeros([max_length, encoder_output.shape[1], self.output_dim]).to(self.device)
        hidden = torch.zeros([1, encoder_output.shape[1], self.decoder_hidden]).to(self.device)
        input_ = self.embedding(torch.ones([encoder_output.shape[1], 1], dtype=torch.long).to(self.device) * self.sos_token)
        input_ = input_.permute(1, 0, 2)
        for i in range(max_length):
            attention = self.attn(hidden, encoder_output)
            context = torch.bmm(encoder_output.permute(1, 2, 0), attention)
            context = context.permute(2, 0, 1)
            gru_input = torch.cat([context, input_], dim=2)
            single_output, hidden = self.gru(gru_input, hidden)
            single_output = single_output.squeeze()

            output[i] = self.output_decoder(single_output)
            input_ = torch.argmax(output[i], dim=1, keepdim=True)
            input_ = self.embedding(input_)
            input_ = input_.permute(1, 0, 2)
        return output


class Translator(nn.Module):
    def __init__(self, encoder_hidden, decoder_hidden, input_dim, output_dim, embedding_dim_encoder, embedding_dim_decoder, sos_token, device):
        super().__init__()
        self.encoder = TwoWayGRUEncoder(input_dim, embedding_dim_encoder, encoder_hidden)
        self.decoder = GruAttentionDecoder(encoder_hidden, decoder_hidden, output_dim, embedding_dim_decoder, sos_token, device)

    def test(self, input_sentence, max_length):
        with torch.no_grad:
            encoded, _ = self.encoder(input_sentence=15)
            output = self.decoder.test(encoded, max_length)


    def forward(self, source, target):
        encoded, _ = self.encoder(source)
        output = self.decoder(encoded, target)
        return output




