class HARDataset(Dataset):
    """
    Custom dataset for HAR with topological features.
    """
    
    def __init__(self, windowed_data, topological_features, labels):
        self.windowed_data = torch.FloatTensor(windowed_data)
        self.topological_features = torch.FloatTensor(topological_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.topological_features[idx],
            self.windowed_data[idx],
            self.labels[idx]
        )
