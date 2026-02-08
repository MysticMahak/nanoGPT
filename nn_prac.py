import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size*20, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

X = torch.randn(20, 5, 20, device=device)
y = torch.randint(0, 2, (20,), device=device)

model = NN(5).to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    l = loss(outputs, y)
    l.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/10, Loss: {l.item():.4f}")

model.eval()

with torch.no_grad():
    test_input = torch.randn(1, 5, 20, device=device)
    test_output = model(test_input)
    predicted_class = torch.argmax(test_output, dim=1).item()
    print(f"Predicted class: {predicted_class}")

