import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for frames, labels, _ in tqdm(dataloader, desc="Training"):
        # Prepare input [B, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels, _ in dataloader:
            frames = frames.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)
            
            logits = model(frames)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy
