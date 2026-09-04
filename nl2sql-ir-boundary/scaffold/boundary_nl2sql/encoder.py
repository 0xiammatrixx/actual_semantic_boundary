"""Fixed encoder (do not modify)."""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.out_dim = hidden * 2

    def encode(self, ids, lens):
        e = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out  # [B, L, out_dim]

    def encode_names(self, name_ids):
        e = self.emb(name_ids)
        out, _ = self.lstm(e)
        mask = (name_ids != 0).float().unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # [S, out_dim]
