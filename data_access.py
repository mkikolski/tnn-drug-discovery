import requests
from tqdm import tqdm
from datetime import datetime


class DataAccess:
    @staticmethod
    def _create_error_log(*args):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        with open(f"logs/error_{dt_string}.log", "a") as f:
            f.write("\n".join(args) + "\n")

    @staticmethod
    def get_general_training_chembl_data(path: str, limit: int = 10, smi_length: int = 100, checkpoint_path: str | None = None, tc: int = 2496335):
        offset = 0
        if checkpoint_path:
            with open(checkpoint_path, 'r') as f:
                offset = int(f.read().strip())
        if not path.endswith("/"):
            path += "/"
        for o in tqdm(range(offset, tc - limit, limit), desc="Fetching data", unit="batch"):
            fname = f"{path}chembl_data_{o}.csv"
            with open(fname, "w") as f:
                f.write("chembl_id;smiles\n")
            r = requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit={limit}&only=molecule_chembl_id,molecule_structures&offset={o}")
            if r.status_code == 200:
                data = r.json()
                for molecule in data["molecules"]:
                    try:
                        name = molecule["molecule_chembl_id"]
                        smiles = molecule["molecule_structures"]["canonical_smiles"]
                        if len(smiles) < smi_length:
                            with open(fname, "a") as f:
                                f.write(f"{name};{smiles}\n")
                    except Exception as e:
                        with open("checkpoints/chembl_checkpoint", "w") as f:
                            f.write(str(o))
                        DataAccess._create_error_log(f"Error: {e}", f"Offset: {o}", str(molecule))
                offset = o
            else:
                with open("checkpoints/chembl_checkpoint", "w") as f:
                    f.write(str(o))
                raise Exception(f"Failed to fetch data: {r.status_code}")
        with open("checkpoints/chembl_checkpoint", "w") as f:
            f.write(str(offset))
