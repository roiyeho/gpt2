import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from dataset_manager import DatasetManager
from gpt2 import GPT2

class Trainer:
    def __init__(self, num_epochs=3, learning_rate=5e-4, batch_size=8, seq_length=128, 
                 vocab_size=32000, warmup_steps=1000):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.warmup_steps = warmup_steps
        
        # Prepare dataset and dataloader
        self.dataset_manager = DatasetManager(vocab_size=vocab_size, seq_length=seq_length, batch_size=batch_size)
        self.train_loader, self.val_loader = self.dataset_manager.get_dataloaders()
        
        # Initialize model, loss function, and optimizer
        self.model = GPT2(vocab_size, max_seq_len=seq_length, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler with OneCycleLR (warmup + cosine decay)
        total_steps = self.num_epochs * len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=self.warmup_steps / total_steps,
            anneal_strategy="cos"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """Runs the training loop."""
        print("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, (input_seq, target_seq) in enumerate(self.train_loader):
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(input_seq)
                loss = self.criterion(output.view(-1, self.vocab_size), target_seq.view(-1))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx}/{len(self.train_loader)}], Train Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(self.train_loader)            
            avg_val_loss, val_perplexity = self.evaluate()
            print(f"Epoch [{epoch+1}/{self.num_epochs}] completed. Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, " 
                  f"Perplexity: {val_perplexity:.4f}")
            
             # Example text generation
            print(self.generate_text("Once upon a time,"))
            
            # Save model checkpoint
            torch.save(self.model.state_dict(), f"checkpoints/gpt2_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved: gpt2_epoch_{epoch+1}.pt")

    def evaluate(self):
        """Computes the validation loss."""
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)
                         
                output = self.model(input_seq)
                
                loss = self.criterion(output.view(-1, self.vocab_size), target_seq.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        perplexity = np.exp(avg_val_loss)
        return avg_val_loss, perplexity  

    def generate_text(self, prompt, max_length=50):
        """Generates text using greedy decoding."""
        self.model.eval()
        input_ids = torch.tensor(self.dataset_manager.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(input_ids)
                next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        return self.dataset_manager.tokenizer.decode(input_ids.squeeze().tolist())
      
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
