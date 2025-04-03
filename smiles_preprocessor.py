import re
from collections import defaultdict
import numpy as np
import torch


class SMILESPreprocessor:
    def __init__(self, smiles_list: list[str] = None):
        self.tokens = ['<pad>', '<sos>', '<eos>', '<unk>'] + \
                      list('BCNOSPFIbcnosp=#()[]+-.0123456789:@?>*%') + \
                      ['Br', 'Cl']

        if smiles_list:
            self._build_from_smiles(smiles_list)

        self.smiles_to_index = {s: i for i, s in enumerate(self.tokens)}
        self.index_to_smiles = {i: s for i, s in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

    def _build_from_smiles(self, smiles_list: list[str]):
        token_counts = defaultdict(int)
        for smi in smiles_list:
            tokens = self._tokenize(smi)
            for token in tokens:
                token_counts[token] += 1

        for token, count in token_counts.items():
            if count > 10 and token not in self.tokens:
                self.tokens.append(token)

    def _tokenize(self, smi: str) -> list:
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
        return re.findall(pattern, smi)

    def encode(self, smi: str) -> list[int]:
        tokens = ['<sos>'] + self._tokenize(smi) + ['<eos>']
        return [self.smiles_to_index.get(t, self.smiles_to_index['<unk>']) for t in tokens]

    def decode(self, indices: list[int]) -> str:
        tokens = [self.index_to_smiles.get(i, '<unk>') for i in indices]
        return ''.join(tokens).replace('<sos>', '').replace('<eos>', '')
