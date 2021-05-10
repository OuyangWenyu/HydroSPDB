import torch
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
from hydroDL.transformer_xl.masks import generate_square_subsequent_mask
from torch.autograd import Variable
from hydroDL.meta_models.merging_model import MergingModel
from hydroDL.transformer_xl.lower_upper_config import activation_dict


class SimpleTransformer(torch.nn.Module):
    def __init__(
            self,
            number_time_series: int,
            seq_length: int = 48,
            output_seq_len: int = None,
            d_model: int = 128,
            n_heads: int = 8,
            dropout=0.1,
            forward_dim=2048,
            sigmoid=False):
        """
        Full transformer model
        """
        super().__init__()
        if output_seq_len is None:
            output_seq_len = seq_length
        self.out_seq_len = output_seq_len
        self.mask = generate_square_subsequent_mask(seq_length)
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(d_model, nhead=n_heads)
        self.final_layer = torch.nn.Linear(d_model, 1)
        self.sequence_size = seq_length
        self.tgt_mask = generate_square_subsequent_mask(output_seq_len)
        self.sigmoid = None
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, t: torch.Tensor, tgt_mask=None, src_mask=None):
        if src_mask:
            x = self.encode_sequence(x[:, :-1, :], src_mask)
        else:
            x = self.encode_sequence(x[:, :-1, :], src_mask)
        return self.decode_seq(x, t, tgt_mask)

    def basic_feature(self, x: torch.Tensor):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def encode_sequence(self, x, src_mask=None):
        x = self.basic_feature(x)
        x = self.transformer.encoder(x, src_mask)
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) -> torch.Tensor:
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
        t = self.basic_feature(t)
        x = self.transformer.decoder(t, mem, tgt_mask=tgt_mask)
        x = self.final_layer(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x.view(-1, view_number)


class CustomTransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            seq_length: int,
            output_seq_length: int,
            n_time_series: int,
            d_model=128,
            output_dim=1,
            n_layers_encoder=6,
            forward_dim=2048,
            dropout=0.1,
            use_mask=False,
            meta_data=None,
            final_act=None,
            n_heads=8):
        """
        Uses a number of encoder layers with simple linear decoder layer.
        """
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8, forward_dim, dropout)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, n_layers_encoder, encoder_norm)
        self.output_dim_layer = torch.nn.Linear(d_model, output_dim)
        self.output_seq_length = output_seq_length
        self.out_length_lay = torch.nn.Linear(seq_length, output_seq_length)
        self.mask = generate_square_subsequent_mask(seq_length)
        self.out_dim = output_dim
        self.mask_it = use_mask
        self.final_act = None
        if final_act:
            self.final_act = activation_dict[final_act]
        if meta_data:
            self.meta_merger = MergingModel(meta_data["method"], meta_data["params"])

    def forward(self, x: torch.Tensor, meta_data=None) -> torch.Tensor:
        """
        Performs forward pass on tensor of (batch_size, sequence_length, n_time_series)
        Return tensor of dim (batch_size, output_seq_length)
        """
        x = self.dense_shape(x)
        if type(meta_data) == torch.Tensor:
            # batch_size = x.shape[0]
            # meta_data = meta_data.repeat(batch_size, 1).unsqueeze(2)
            # x = x.permute(0, 2, 1).contiguous()
            x = self.meta_merger(x, meta_data)
            # x = x.permute(0, 2, 1)
        x = self.pe(x)
        # (L, B, N)
        x = x.permute(1, 0, 2)
        if self.mask_it:
            x = self.transformer_enc(x, self.mask)
        else:
            # Allow no mask
            x = self.transformer_enc(x)
        x = self.output_dim_layer(x)
        # (B, N, L)
        x = x.permute(1, 2, 0)
        x = self.out_length_lay(x)
        if self.final_act:
            x = self.final_act(x)
        if self.out_dim > 1:
            return x.permute(0, 2, 1)
        return x.view(-1, self.output_seq_length)


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def greedy_decode(
        model,
        src: torch.Tensor,
        max_len: int,
        real_target: torch.Tensor,
        unsqueeze_dim=1,
        output_len=1,
        device='cpu',
        multi_targets=1,
        probabilistic=False,
        scaler=None):
    """
    Mechanism to sequentially decode the model
    :src Historical time series values
    :real_target The real values (they should be masked), however if want can include known real values.
    :returns tensor
    """
    src = src.float()
    real_target = real_target.float()
    if hasattr(model, "mask"):
        src_mask = model.mask
    memory = model.encode_sequence(src, src_mask)
    # Get last element of src array to forecast from
    ys = src[:, -1, :].unsqueeze(unsqueeze_dim)
    for i in range(max_len):
        mask = generate_square_subsequent_mask(i + 1).to(device)
        with torch.no_grad():
            out = model.decode_seq(memory,
                                   Variable(ys),
                                   Variable(mask), i + 1)
            real_target[:, i, 0] = out[:, i]
            src = torch.cat((src, real_target[:, i, :].unsqueeze(1)), 1)
            ys = torch.cat((ys, real_target[:, i, :].unsqueeze(1)), 1)
        memory = model.encode_sequence(src[:, i + 1:, :], src_mask)
    return ys[:, 1:, :]
