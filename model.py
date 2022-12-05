import torch
import torch.nn as nn
from torch import Tensor

# positional encoding from my small HW 7

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.empty((1, max_len, embed_dim))
        pe[0, :, 0::2] = torch.sin(
            torch.arange(max_len)[:,None] / 10000**(torch.arange(0, embed_dim, 2) / embed_dim)
        )
        pe[0, :, 1::2] = torch.cos(
            torch.arange(max_len)[:,None] / 10000**(torch.arange(0, embed_dim, 2) / embed_dim)
        )
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        res = x + self.pe[:,:x.shape[1]]
        return res


# result is similar to torch tutorial
# but I have no idea how to make it not similar (except making solution suboptimally)
class TranslationModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        dim_feedforward: int,
        n_head: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout_prob: float,
        src_pad_id: int,
        tgt_pad_id: int,
        max_len: int
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        # standard PRE-LN transformer (it should be faster for low budgets)
        self.transformer = nn.Transformer(
            emb_size,
            n_head,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout_prob,
            batch_first=True,
            norm_first=True
        )
        self.linear = nn.Linear(emb_size, tgt_vocab_size)

        # embeddings
        self.pe = PositionalEncoding(emb_size, max_len)
        self.src_embed = nn.Embedding(src_vocab_size, emb_size, padding_idx=src_pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=tgt_pad_id)

    def forward(
        self,
        tgt_tokens: Tensor,
        src_tokens: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        decoded = self.transformer(
            src=self.pe(self.src_embed(src_tokens)),
            tgt=self.pe(self.tgt_embed(tgt_tokens)),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.linear(decoded)


    def encode(
        self,
        src_tokens: Tensor,
        src_padding_mask: Tensor,
    ):
        return self.transformer.encoder(
            src=self.pe(self.src_embed(src_tokens)),
            src_key_padding_mask=src_padding_mask,
        )


    def decode_last(
        self,
        encoded_src: Tensor,
        tgt_tokens: Tensor,
        tgt_mask: Tensor = None,
    ):
        decoded = self.transformer.decoder(
            tgt=self.pe(self.tgt_embed(tgt_tokens)),
            memory=encoded_src,
            tgt_mask=tgt_mask,
        )
        return self.linear(decoded[:,-1])