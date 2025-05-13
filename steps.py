from torch.utils.data import DataLoader, TensorDataset
import torch

from data_access import DataAccess
import os
from dotenv import load_dotenv
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from save_backup import SFTPUploader
from smiles_preprocessor import SMILESPreprocessor
from tnn_generator import TransformerGenerator


class Steps:
    @staticmethod
    def fetch_data(**kwargs) -> dict:
        DataAccess.get_general_training_chembl_data("data/general" if "save_path" not in kwargs else kwargs["save_path"], limit=1000, smi_length=100, tc=2496335)
        return {} if "save_path" not in kwargs else kwargs

    @staticmethod
    def pretrain_generator(**kwargs) -> dict:
        smiles_list = []

        for file in os.listdir("data/general" if "save_path" not in kwargs else kwargs["save_path"]):
            with open(f"{'data/general' if 'save_path' not in kwargs else kwargs['save_path']}/{file}", "r") as f:
                for l in f.readlines():
                    if len(l) > 3:
                        smiles_list.append(l.rstrip())

        preprocessor = SMILESPreprocessor(smiles_list)
        generator = TransformerGenerator(preprocessor)

        optimizer = optim.Adam(generator.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        encoded = pad_sequence([torch.Tensor(preprocessor.encode(smiles)) for smiles in smiles_list], batch_first=True, padding_value=preprocessor.smiles_to_index["<pad>"])
        tokens = TensorDataset(encoded)
        dl = DataLoader(tokens, batch_size=32, shuffle=True)

        generator.train()
        for epoch in range(1000):
            tl = 0
            for batch in dl:
                input_seq = batch[0][:, :-1].to("cuda")
                target_seq = batch[0][:, 1:].to("cuda")

                optimizer.zero_grad()
                output = generator(input_seq)
                loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                loss.backward()
                optimizer.step()

                tl += loss.item()
            print(f"[Pretrain] Epoch {epoch+1}: Loss = {tl / len(dl):.4f}")
        generator.save("pretrained_generator.pt")
        # Steps.save_backup_files("pretrained_generator.pt")
        return {"pretrained_generator": generator}

    @staticmethod
    def train_dqn(**kwargs) -> dict:
        return {}

    @staticmethod
    def save_backup_files(*paths):
        conn = SFTPUploader(
            host=os.getenv("BACKUP_HOST"),
            port=os.getenv("BACKUP_PORT"),
            username=os.getenv("BACKUP_USER"),
            key_file=os.getenv("BACKUP_KEY_PATH")
        )

        conn.connect()
        for path in paths:
            conn.upload(path, f"/home/ubuntu/{path}")

        conn.close()
