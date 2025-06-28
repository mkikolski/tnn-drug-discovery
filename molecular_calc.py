from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
import subprocess as sp
import re

class MolCalc():
    def __init__(self, qvina2_gpu_path, openbabel_path, receptor_path, output_dir, x, y, z, x_size, y_size, z_size, threads):
        self.qvina2_gpu_path = qvina2_gpu_path
        self.openbabel_path = openbabel_path
        self.receptor_path = receptor_path
        self.output_dir = output_dir
        self.x = x
        self.y = y
        self.z = z
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.threads = threads

        self.__prepare_receptor()

    def __prepare_receptor(self):
        sp.run([f"{self.openbabel_path} {self.receptor_path} -O {self.output_dir}/target_no_water.pdb -xr HOH"], shell=True)
        sp.run([f"{self.openbabel_path} {self.output_dir}/target_no_water.pdb -O {self.output_dir}/target_h.pdb -h -xr"], shell=True)
        sp.run([f"{self.openbabel_path} {self.output_dir}/target_h.pdb -O {self.output_dir}/target_out.pdbqt -xr --partialcharge gasteiger"], shell=True)

    def __prepare_ligand(self, smiles):
        sp.run([f"{self.openbabel_path} -:\"{smiles}\" -O {self.output_dir}/ligand.pdb -h --gen3d"], shell=True)
        sp.run([f"{self.openbabel_path} {self.output_dir}/ligand.pdb -O {self.output_dir}/ligand.pdbqt --partialcharge gasteiger"], shell=True)

    def process_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        qed_score = QED.qed(mol)
        sa_score = sascorer.calculateScore(mol)
        docking_score = self.docking(smiles)
        return qed_score, sa_score, docking_score

    def __parse_docking_output(self, output):
        result = re.search(r'^\s*1\s+([-+]?\d+\.\d+)', output, re.MULTILINE)
        if result:
            return float(result.group(1))
        return 0.0
    
    def docking(self, smiles):
        self.__prepare_ligand(smiles)
        out = sp.run([f"{self.qvina2_gpu_path} --receptor {self.output_dir}/target_out.pdbqt --ligand {self.output_dir}/ligand.pdbqt --center_x {self.x} --center_y {self.y} --center_z {self.z} --size_x {self.x_size} --size_y {self.y_size} --size_z {self.z_size} --threads {self.threads}"], shell=True, capture_output=True, text=True)
        return self.__parse_docking_output(out.stdout)