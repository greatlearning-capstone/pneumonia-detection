"""
Simple training script for pneumonia detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from models.model import PneumoniaClassifier

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nCreating model...")
    model = PneumoniaClassifier(num_classes=3, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ##### TODO - Training + Validation loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

    print("\n" + "=" * 60)
    print("Training completed!")
    

if __name__ == "__main__":
    main()
