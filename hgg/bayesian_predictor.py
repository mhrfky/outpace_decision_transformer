import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class BayesianPredictor:
    def __init__(self, input_size=2, hidden_size=64, output_size=1, learning_rate=0.001, weight_decay=1e-4, evaluator=None, limits=[[-2,10],[-2,10]],final_goal = None):
        self.model = BayesianPredictionModel(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.evaluator = evaluator
        self.limits = limits
        self.final_goal = final_goal
        self.final_goal_negative_batch_size = 150
    def get_random_batch(self, batch_size):
        xs = []
        ys = []
        for _ in range(batch_size):
            x = [np.random.uniform(self.limits[0][0], self.limits[0][1]), np.random.uniform(self.limits[1][0], self.limits[1][1])]
            y = self.evaluator(x)
            x = torch.tensor(x, device='cuda', dtype=torch.float32)
            y = torch.tensor(y, device='cuda', dtype=torch.float32)
            xs.append(x)
            ys.append(y)
        for _ in range(self.final_goal_negative_batch_size):
            rand_x_1, rand_x_2 = np.random.uniform(-1,1), np.random.uniform(-1,1)
            rand = np.array([rand_x_1, rand_x_2]) / np.linalg.norm([rand_x_1, rand_x_2])
            x = self.final_goal + rand
            y = 0
            x = torch.tensor(x, device='cuda', dtype=torch.float32)
            y = torch.tensor(y, device='cuda', dtype=torch.float32)
            xs.append(x)
            ys.append(y)
            
        return torch.stack(xs), torch.stack(ys)
    
    def train(self, positives=None, random_sample_size=1000, batch_size=64, epochs=1):
        for epoch in range(epochs):
            random_xs, random_ys = self.get_random_batch(random_sample_size)
            
            if positives is not None:
                positives = torch.tensor(positives, device='cuda', dtype=torch.float32)
                values = torch.ones(positives.size(0), device='cuda', dtype=torch.float32)
                
                # Combine random batch with positive examples
                xs = torch.cat((random_xs, positives), dim=0)
                ys = torch.cat((random_ys, values), dim=0)
            else:
                xs = random_xs
                ys = random_ys
            
            # Shuffle the combined dataset before each epoch
            permutation = torch.randperm(xs.size()[0])
            xs = xs[permutation]
            ys = ys[permutation]
            
            # Train in batches
            for i in range(0, xs.size(0), batch_size):
                batch_xs = xs[i:i + batch_size]
                batch_ys = ys[i:i + batch_size]
                
                self.optimizer.zero_grad()
                output = self.model(batch_xs)
                loss = self.criterion(output.squeeze(-1), batch_ys)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device='cuda', dtype=torch.float32)
        with torch.no_grad():  # Disable gradient calculation for inference
            return self.model(X).detach().cpu().numpy()
    
class BayesianPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(BayesianPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64).cuda()
        self.fc2 = nn.Linear(64, 32).cuda()
        self.fc3 = nn.Linear(32, 1).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x