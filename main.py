import torch
import torch.nn as nn

from trainer import Trainer

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model, loss function, and optimizer
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create Trainer instance
    trainer = Trainer(model, criterion, optimizer, device)

    # Dummy data for demonstration
    train_data = torch.randn(100, 10).to(device)
    train_labels = torch.randn(100, 1).to(device)

    # Train the model
    trainer.train(train_data, train_labels, epochs=5)


if __name__ == "__main__":
    main()