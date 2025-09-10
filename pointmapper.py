class PointMapper:
    """
    Converts 1D sensor signals into 3D point clouds for topological analysis.
    Implements multiple mapping strategies.
    """
    
    def __init__(self, mapping_strategy='time_delay_embedding'):
        self.strategy = mapping_strategy
        
    def time_delay_embedding(self, signal, dim=3, tau=1):
        """
        Create point cloud using time-delay embedding (Takens' theorem).
        
        Args:
            signal: 1D time series
            dim: Embedding dimension
            tau: Time delay
            
        Returns:
            point_cloud: Array of shape (n_points, dim)
        """
        n = len(signal)
        n_points = n - (dim - 1) * tau
        
        point_cloud = np.zeros((n_points, dim))
        for i in range(dim):
            point_cloud[:, i] = signal[i * tau:i * tau + n_points]
            
        return point_cloud
    
    def multi_sensor_embedding(self, acc_data, gyro_data):
        """
        Create 3D point cloud from accelerometer and gyroscope data.
        
        Args:
            acc_data: Accelerometer data (n_samples, 3)
            gyro_data: Gyroscope data (n_samples, 3)
            
        Returns:
            point_cloud: Combined 3D point cloud
        """
        acc_mag = np.linalg.norm(acc_data, axis=1)
        gyro_mag = np.linalg.norm(gyro_data, axis=1)
