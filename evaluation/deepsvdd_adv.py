import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepSVDD Model
class DeepSVDD(nn.Module):
    def __init__(self, ts_dims, hidden_dim, latent_dim, window_length):
        super(DeepSVDD, self).__init__()
        
        # Convolutional Encoder for Temporal Features
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=ts_dims, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Fully Connected Layer for Latent Space
        self.fc = nn.Linear(latent_dim * window_length, latent_dim)
        
        # Register the initial centre as a buffer
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, x):
        # Expecting input x of shape (batch_size, window_length, ts_dims)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, ts_dims, window_length)
        x = self.encoder(x)  # Pass through the convolutional encoder
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)  # Pass through the fully connected layer
        return x

    def calculate_center(self, data_loader):
        """Calculate the initial centre of the data in the latent space."""
        self.eval()
        center = torch.zeros(self.fc.out_features, device=device)
        n_samples = 0

        with torch.no_grad():
            for data in data_loader:
                inputs = data[0].to(device)  # Data is in (batch_size, window_length, ts_dims)
                representations = self(inputs)
                center += representations.sum(dim=0)
                n_samples += representations.size(0)

        center /= n_samples
        self.center = center

    def loss_function(self, latent_space):
        """DeepSVDD loss: Minimise distance to the centre."""
        distances = torch.sum((latent_space - self.center) ** 2, dim=1)
        return torch.mean(distances)


# Anomaly score function
def anomaly_score(model, data):
    """Compute anomaly scores for given data."""
    model.eval()
    with torch.no_grad():
        representations = model(data.to(device))
        scores = torch.sum((representations - model.center) ** 2, dim=1)
    return representations, scores


# Training function
def train_deepsvdd(model, train_loader, num_epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")  # Track the best loss
    best_model_state = None  # Store the best model weights
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)  # Data is in (batch_size, window_length, ts_dims)
            optimizer.zero_grad()
            representations = model(inputs)
            loss = model.loss_function(representations)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            
        if((epoch%10) == 0):
            model.calculate_center(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        if best_model_state:
            model.load_state_dict(best_model_state)

# Prepare data
def prepare_data(data, batch_size):
    """Prepare a DataLoader for the given dataset."""
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader