import math
import torch
import torch.nn as nn
import torch.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from typing import List, Dict, Optional
from positional_encoder import PositionalEncoding


class SMILESPropertyPredictor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.property_heads = nn.ModuleDict({
            'mw': nn.Linear(d_model, 1),  # Molecular weight
            'logp': nn.Linear(d_model, 1),  # LogP
            'hbd': nn.Linear(d_model, 1),  # Hydrogen bond donors
            'hba': nn.Linear(d_model, 1),  # Hydrogen bond acceptors
            'qed': nn.Linear(d_model, 1),  # QED drug-likeness
            'tpsa': nn.Linear(d_model, 1),  # Polar surface area
            'rotatable': nn.Linear(d_model, 1),  # Rotatable bonds
            'sas': nn.Linear(d_model, 1)  # Synthetic accessibility
        })

        self.property_norms = {
            'mw': (500.0, 100.0),  # (mean, std)
            'logp': (2.5, 2.0),
            'hbd': (2.0, 1.5),
            'hba': (5.0, 2.0),
            'qed': (0.6, 0.2),
            'tpsa': (70.0, 30.0),
            'rotatable': (5.0, 3.0),
            'sas': (3.0, 1.0)
        }

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        # Input shape: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer(src, src_mask)

        cls_output = output[0]

        properties = {}
        for name, head in self.property_heads.items():
            properties[name] = head(cls_output)

            if name == 'qed':
                properties[name] = torch.sigmoid(properties[name])
            elif name in ['hbd', 'hba', 'rotatable']:
                properties[name] = F.relu(properties[name])

        return properties

    def predict_from_smiles(self, smiles: str, device: str = 'cpu') -> Dict[str, float]:
        if not smiles:
            return {name: 0.0 for name in self.property_heads.keys()}

        tokens = self.vocab.encode(smiles)
        src = torch.tensor(tokens, device=device).unsqueeze(1)

        with torch.no_grad():
            props = self.forward(src)

        result = {}
        for name, tensor in props.items():
            mean, std = self.property_norms[name]
            val = tensor.item() * std + mean
            result[name] = float(val)

        return result

    def calculate_reward(self, smiles: str, device: str = 'cpu') -> float:
        props = self.predict_from_smiles(smiles, device)

        mw_score = 1.0 - min(1.0, max(0.0, (props['mw'] - 300) / 400))  # Ideal: 300-700
        logp_score = 1.0 - min(1.0, abs(props['logp'] - 2.5) / 3.0)  # Ideal: 0-5
        hbd_score = max(0.0, 1.0 - props['hbd'] / 5.0)  # Ideal: ≤5
        hba_score = max(0.0, 1.0 - props['hba'] / 10.0)  # Ideal: ≤10
        qed_score = props['qed']
        sas_score = max(0.0, 1.0 - props['sas'] / 6.0)  # Ideal: <6

        return 0.2 * mw_score + 0.2 * logp_score + 0.15 * hbd_score + \
            0.15 * hba_score + 0.2 * qed_score + 0.1 * sas_score



