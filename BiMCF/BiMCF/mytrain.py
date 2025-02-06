import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score
import torch.distributed as dist
import torch.multiprocessing as mp
from mydataset import MyDataset
from mymodel import BiMCF

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, lr=1e-4, lambda1=1.0, lambda2=1.0, lambda3=1.0, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.steps_per_epoch = len(train_loader)

    def train_step(self, data):
        self.model.train()
        data = {k: v.to(self.device) for k, v in data.items()}
        outputs = self.model(data)
        loss = self.lambda1 * outputs['task1_loss'] + self.lambda2 * outputs['task2_loss'] + self.lambda3 * outputs['task3_loss']
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return outputs, loss.item()

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        task1_preds, task1_labels = [], []
        task2_preds, task2_labels = [], []
        task3_preds, task3_labels = [], []

        with torch.no_grad():
            for data in loader:
                data = {k: v.to(self.device) for k, v in data.items()}
                outputs = self.model(data)
                loss = self.lambda1 * outputs['task1_loss'] + self.lambda2 * outputs['task2_loss'] + self.lambda3 * outputs['task3_loss']
                total_loss += loss.item()

                task1_preds.extend(torch.argmax(outputs['logit1'], dim=1).cpu().numpy())
                task1_labels.extend(data['task1_label'].cpu().numpy())
                task2_preds.extend(torch.argmax(outputs['logit2'], dim=1).cpu().numpy())
                task2_labels.extend(data['task2_label'].cpu().numpy())
                task3_preds.extend(torch.argmax(outputs['logit3'], dim=1).cpu().numpy())
                task3_labels.extend(data['task3_label'].cpu().numpy())

        avg_loss = total_loss / len(loader)
        task1_acc = accuracy_score(task1_labels, task1_preds)
        task2_acc = accuracy_score(task2_labels, task2_preds)
        task3_acc = accuracy_score(task3_labels, task3_preds)

        return avg_loss, task1_acc, task2_acc, task3_acc

    def save_checkpoint(self, step):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
        }, checkpoint_path)

    def train(self, num_steps, eval_interval=10):
        step = 0
        while step < num_steps:
            for data in self.train_loader:
                outputs, loss = self.train_step(data)
                step += 1

                current_epoch = step / self.steps_per_epoch

                if step % 10 == 0:
                    print(f'Step {step}, Epoch {current_epoch:.2f}, Loss: {loss:.4f}, Task1 Loss: {outputs["task1_loss"].item():.4f}, Task2 Loss: {outputs["task2_loss"].item():.4f}, Task3 Loss: {outputs["task3_loss"].item():.4f}')
                if step % eval_interval == 0:
                    val_loss, task1_acc, task2_acc, task3_acc = self.evaluate(self.val_loader)
                    print(f'Validation - Step {step}, Epoch {current_epoch:.2f}, Loss: {val_loss:.4f}, Task1 Acc: {task1_acc:.4f}, Task2 Acc: {task2_acc:.4f}, Task3 Acc: {task3_acc:.4f}')
                    self.save_checkpoint(step)

                if step >= num_steps:
                    break

def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda1', type=float, default=1.0, help='Lambda1 for loss calculation')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Lambda2 for loss calculation')
    parser.add_argument('--lambda3', type=float, default=1.0, help='Lambda3 for loss calculation')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()

    print(args)

    train_dataset = MyDataset(args.train_path)
    val_dataset = MyDataset(args.val_path)
    test_dataset = MyDataset(args.test_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiMCF()

    print(model)
    
    trainer = Trainer(model, train_loader, val_loader, test_loader, device, args.lr, args.lambda1, args.lambda2, args.lambda3, args.checkpoint_dir)
    trainer.train(num_steps=args.num_steps, eval_interval=args.eval_interval)

    

if __name__ == "__main__":
    main()