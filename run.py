from pipeline import Pipeline
from steps import Steps
import argparse

parser = argparse.ArgumentParser(description="Run the pipeline or specific steps.")
group = parser.add_mutually_exclusive_group()

group.add_argument("-s", "--step", type=str, default=None, help="Specify a step to run. Options: fetch, pretrain, dqn.")
group.add_argument("-f", "--full", action="store_true", help="Run the full pipeline.")

args = parser.parse_args()

if __name__ == "__main__":
    p = Pipeline()
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
    