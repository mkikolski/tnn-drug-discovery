from fsspec.registry import default

from pipeline import Pipeline
from steps import Steps
import argparse

parser = argparse.ArgumentParser(description="Run the pipeline or specific steps.")
group = parser.add_mutually_exclusive_group()

group.add_argument("-s", "--step", type=str, default=None, help="Specify a step to run. Options: fetch, pretrain, dqn.")
group.add_argument("-f", "--full", action="store_true", help="Run the full pipeline.")

parser.add_argument("-sp", "--smiles-path", type=str, default=None, help="Pass path to save location of SMILES for pretraining process. Defaults to data/general.")
parser.add_argument("-cp", "--checkpoint-path", type=str, default=None, help="Pass path to checkpoint file to be used for storing/loading state of download.")
parser.add_argument("-m", "--model-states", type=str, default=None, help="Pass path to pretrained model or training checkpoints.")
parser.add_argument("-O", "--output", type=str, default=None, help="Specifies an output path of the model save file.")
parser.add_argument("-C", "--checkpoint", type=str, default=None, help="File path for training to know where to save checkpoint files")

args = parser.parse_args()

if __name__ == "__main__":
    initial = {}
    if args.smiles_path is not None:
        initial["save_path"] = args.smiles_path
    if args.model_states is not None:
        initial["model_dict"] = args.model_states
    if args.output is not None:
        initial["model_save"] = args.output
    if args.checkpoint_path is not None:
        initial["checkpoint_path"] = args.checkpoint_path
    p = Pipeline(**initial)
    if args.step == "fetch":
        print("Entered fetch step")
        p.add_step({"name": "Fetching pretraining data", "function": Steps.fetch_data})
    elif args.step == "pretrain":
        print("Entered pretraining step")
        p.add_step({"name": "Pretraining generator", "function": Steps.pretrain_generator})
    elif args.step == "dqn":
        print("Entered dqn training step")
        p.add_step({"name": "Training DQN", "function": Steps.train_dqn})
    elif args.full:
        print("Executing full pipeline")
        p.add_step({"name": "Fetching pretraining data", "function": Steps.fetch_data})\
            .add_step({"name": "Pretraining generator", "function": Steps.pretrain_generator})\
            .add_step({"name": "Training DQN", "function": Steps.train_dqn})
    p.run()
    