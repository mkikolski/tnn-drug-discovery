import torch
from torch.utils.data import DataLoader, random_split
import os
import argparse
from tokenizer import SMILESTokenizer
from dataset import SMILESDataset
from tnn import TNN
from trainer import SMILESTrainer


def main(args):
    tokenizer = SMILESTokenizer()
    
    dataset = SMILESDataset(Global.DATA_PATH, tokenizer, max_length=Global.MAX_LEN)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Global.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Global.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    model = TNN(
        n_embeddings=tokenizer.vocab_size,
        hidden_size=256,
        n_layers=6,
        nheads=8
    )
    
    trainer = SMILESTrainer(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        num_epochs=Global.PRETRAINING_EPOCHS,
        save_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency,
        resume_from=args.resume_from
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SMILES generator model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--save_frequency', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    main(args) 