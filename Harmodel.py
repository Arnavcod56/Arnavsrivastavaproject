class TopologicalHARModel(nn.Module):
    """
    Hybrid model combining topological features with temporal convolutions.
    """
    
    def __init__(self, topo_input_size, raw_input_size, num_classes, hidden_size=128):
        super(TopologicalHARModel, self).__init__()

        self.topo_branch = nn.Sequential(
            nn.Linear(topo_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.raw_branch = nn.Sequential(
            nn.Conv1d(raw_input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, topo_features, raw_signals):
        topo_out = self.topo_branch(topo_features)

        raw_signals = raw_signals.transpose(1, 2) 
        raw_out = self.raw_branch(raw_signals)
  
        fused = torch.cat([topo_out, raw_out], dim=1)
  
        output = self.classifier(fused)
        return output
