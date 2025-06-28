import argparse
import toml
import os
import torch
from tokenizer import SMILESTokenizer
from dataset import SMILESDataset
from tnn import TNN
from trainer import SMILESTrainer, RLAgent, populate_replay_buffer_from_csv, custom_reward, compute_mc_returns


def main():
    parser = argparse.ArgumentParser(description='Run molecule generation and RL pipeline')
    parser.add_argument('--config', type=str, default='example_config.toml', help='Path to TOML config file')
    parser.add_argument('--step', type=str, choices=['pretrain', 'rl', 'generate', 'reward'], default='pretrain', help='Which section to run')
    args = parser.parse_args()

    config = toml.load(args.config)
    general = config['general']
    training = config['training']
    model_cfg = config['model']
    rl_cfg = config['rl']
    generate_cfg = config.get('generate', {})
    reward_cfg = config.get('reward', {})

    device = general.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = SMILESTokenizer()

    dataset = SMILESDataset(general['data_path'], tokenizer, max_length=general['data_max_len'])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training['batch_size'],
        shuffle=False,
        num_workers=4
    )

    mol_calc = MolCalc(
        qvina2_gpu_path=reward_cfg['qvina2_gpu_path'],
        openbabel_path=reward_cfg['openbabel_path'],
        receptor_path=reward_cfg['receptor_path'],
        output_dir=reward_cfg['output_dir'],
        x=reward_cfg['x'],
        y=reward_cfg['y'],
        z=reward_cfg['z'],
        x_size=reward_cfg['x_size'],
        y_size=reward_cfg['y_size'],
        z_size=reward_cfg['z_size'],
        threads=reward_cfg['threads']
    )

    model = TNN(
        n_embeddings=tokenizer.vocab_size,
        hidden_size=model_cfg['hidden_size'],
        n_layers=model_cfg['n_layers'],
        nheads=model_cfg['nheads']
    )

    if args.step == 'pretrain':
        trainer = SMILESTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        os.makedirs(training['checkpoint_dir'], exist_ok=True)
        trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            num_epochs=training['pretraining_epochs'],
            save_dir=training['checkpoint_dir'],
            save_frequency=training['save_frequency'],
            resume_from=training['resume_from'] if training['resume_from'] else None
        )

    elif args.step == 'rl':
        state_dim = general['data_max_len']
        action_dim = tokenizer.vocab_size
        agent = RLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_capacity=rl_cfg['buffer_capacity'],
            batch_size=rl_cfg['rl_batch_size'],
            gamma=rl_cfg['gamma'],
            lr=rl_cfg['rl_learning_rate']
        )

        if os.path.exists(rl_cfg['replay_csv']):
            populate_replay_buffer_from_csv(agent.buffer, rl_cfg['replay_csv'], tokenizer, general['data_max_len'])
            print(f"Replay buffer prepopulated with {len(agent.buffer)} entries from {rl_cfg['replay_csv']}")
        else:
            print(f"Replay CSV {rl_cfg['replay_csv']} not found. RL will start with empty buffer.")

        trainer = SMILESTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        checkpoint_path = training['resume_from']
        if checkpoint_path and os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded generator checkpoint from {checkpoint_path}")
        else:
            print("No generator checkpoint found for RL. Please pretrain and set resume_from.")
            return

        num_episodes = rl_cfg.get('num_episodes', 1000)
        steps_per_episode = rl_cfg.get('steps_per_episode', 32)
        epsilon = rl_cfg.get('epsilon', 0.1)
        save_every = rl_cfg.get('save_every', 100)
        batch_size = rl_cfg['rl_batch_size']

        for episode in range(1, num_episodes + 1):
            smiles_batch = trainer.generate(num_samples=steps_per_episode, temperature=0.7)
            states, rewards, qeds, sas, dockings = [], [], [], [], []
            for smiles in smiles_batch:
                qed, sa, docking = mol_calc.process_molecule(smiles)
                reward = custom_reward(smiles, qed, sa, docking)
                state = tokenizer.encode(smiles, general['data_max_len']).float()
                states.append(state)
                rewards.append(reward)
                qeds.append(qed)
                sas.append(sa)
                dockings.append(docking)

            mc_returns = compute_mc_returns(rewards, gamma=rl_cfg.get('gamma', 0.99))
            for state, mc_return in zip(states, mc_returns):
                agent.buffer.push(state, 0, mc_return, None, False)
            for _ in range(steps_per_episode):
                agent.update()
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} | Buffer size: {len(agent.buffer)}")
            if episode % save_every == 0:
                torch.save(agent.dqn.state_dict(), f"dqn_agent_episode_{episode}.pt")
                print(f"Saved DQN agent at episode {episode}")
        print("RL training complete.")

    elif args.step == 'generate':
        trainer = SMILESTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        checkpoint_path = training['resume_from']
        if checkpoint_path and os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint found for generation. Make sure to pass a proper checkpoint path.")
            return
        num_molecules = generate_cfg.get('num_molecules', 10)
        temperature = generate_cfg.get('temperature', 0.7)
        molecules = trainer.generate(num_samples=num_molecules, temperature=temperature)
        print(f"\nGenerated {num_molecules} molecules:")
        for i, mol in enumerate(molecules, 1):
            print(f"Molecule {i}: {mol}")

    elif args.step == 'reward':
        num_mols = reward_cfg.get('pretraining_molecules', 1000)
        rows = []
        for i in range(min(num_mols, len(dataset))):
            smiles = dataset.sequences[i]
            qed, sa, docking = mol_calc.process_molecule(smiles)
            rows.append([smiles, qed, sa, docking])
        with open(rl_cfg['replay_csv'], 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'qed', 'sa', 'docking'])
            writer.writerows(rows)

if __name__ == "__main__":
    main() 