from pipeline import Pipeline
from steps import Steps

if __name__ == "__main__":
    p = Pipeline()
    p.add_step({"name": "Fetching pretraining data", "function": Steps.fetch_data})\
        .add_step({"name": "Pretraining generator", "function": Steps.pretrain_generator})\
        .add_step({"name": "Training DQN", "function": Steps.train_dqn})
    p.run()
    