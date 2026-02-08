import torch
from torch import nn
import torch.nn.functional as F, torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 2 * 10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
    
X = torch.randn(20, 3, 5, 20, device=device)
y = torch.randint(0, 2, (20,), device=device)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    test_input = torch.randn(1, 3, 5, 20, device=device)
    test_output = model(test_input)
    predicted_class = torch.argmax(test_output, dim=1).item()
    print(f"Predicted class: {predicted_class}")