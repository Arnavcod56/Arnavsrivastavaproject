class HARTrainer:
    """
    Training pipeline for the HAR model.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for topo_features, raw_signals, labels in dataloader:
            topo_features = topo_features.to(self.device)
            raw_signals = raw_signals.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(topo_features, raw_signals)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for topo_features, raw_signals, labels in dataloader:
                topo_features = topo_features.to(self.device)
                raw_signals = raw_signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(topo_features, raw_signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Complete training loop."""
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0
        
        for epoch in range(num_epochs):

            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_acc)
    
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_har_model.pth')

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Acc: {train_acc:.2f}%, '
                      f'Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
