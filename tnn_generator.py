import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encoder import PositionalEncoding


class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer(src, src_mask, src_key_padding_mask)
        return self.fc_out(output)

    def generate(self, max_len: int = 100, temperature: float = 1.0, device: str = 'cpu') -> list[int]:
        self.eval()
        with torch.no_grad():
            current_token = torch.tensor([[self.vocab.smiles_to_index['<sos>']]], device=device)
            generated = [current_token.item()]

            for _ in range(max_len):
                tgt_mask = self._generate_square_subsequent_mask(current_token.size(0)).to(device)

                output = self(current_token, tgt_mask)
                probs = F.softmax(output[-1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

                if next_token.item() == self.vocab.smiles_to_index['<eos>']:
                    break

                generated.append(next_token.item())
                current_token = torch.cat([current_token, next_token.unsqueeze(0)], dim=0)

            return generated

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
