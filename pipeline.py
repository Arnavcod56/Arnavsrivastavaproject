def main_pipeline():
    """
    Complete HAR pipeline demonstration.
    """
    print("HAR with Topological Deep Learning Pipeline")
    print("=" * 50)
    
    print("1. Generating synthetic sensor data...")
    np.random.seed(42)
    n_samples = 10000
    n_features = 6 
    n_classes = 6 
    data = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)
   
    for i in range(n_classes):
        mask = labels == i
        freq = (i + 1) * 0.1
        time = np.arange(np.sum(mask))
        pattern = np.sin(2 * np.pi * freq * time / 50).reshape(-1, 1)
        data[mask] += pattern * (i + 1) * 0.5
    
    print(f"Generated {n_samples} samples with {n_features} features and {n_classes} classes")

    print("\n2. Preprocessing sensor data...")
    preprocessor = SensorDataPreprocessor(window_size=128, overlap=0.5)
    windowed_data, window_labels = preprocessor.preprocess_pipeline(data, labels)
    print(f"Created {len(windowed_data)} windows of size {windowed_data.shape[1:]}")

    print("\n3. Converting to point clouds...")
    point_mapper = PointMapper()
    point_clouds = point_mapper.map_to_point_clouds(windowed_data)
    print(f"Generated {len(point_clouds)} point clouds")

    print("\n4. Extracting topological features...")
    topo_extractor = TopologicalFeatureExtractor()
    topological_features = []
    
    for i, pc in enumerate(point_clouds[:100]): 
        if i % 20 == 0:
            print(f"Processing point cloud {i+1}/100...")
        features = topo_extractor.extract_topological_features(pc)
        topological_features.append(features)
    
    topological_features = np.array(topological_features)
    print(f"Extracted topological features: {topological_features.shape}")

    print("\n5. Training the model...")
    X_topo = topological_features[:100]
    X_raw = windowed_data[:100]
    y = window_labels[:100]

    indices = np.arange(len(X_topo))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
  
    train_dataset = HARDataset(
        X_raw[train_idx], X_topo[train_idx], y[train_idx]
    )
    test_dataset = HARDataset(
        X_raw[test_idx], X_topo[test_idx], y[test_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = TopologicalHARModel(
        topo_input_size=X_topo.shape[1],
        raw_input_size=n_features,
        num_classes=n_classes
    )

    trainer = HARTrainer(model)
    results = trainer.train(train_loader, test_loader, num_epochs=50)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    
    return results


if __name__ == "__main__":
    main_pipeline()
