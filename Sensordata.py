class SensorDataPreprocessor:
    """
    Handles preprocessing of wearable sensor data for HAR.
    Includes filtering, normalization, and window segmentation.
    """
    
    def __init__(self, window_size=128, overlap=0.5, sampling_rate=50):
        self.window_size = window_size  
        self.overlap = overlap  
        self.sampling_rate = sampling_rate  
        self.scaler = StandardScaler()
        
    def butter_lowpass_filter(self, data, cutoff=20, order=4):
        """
        Apply Butterworth low-pass filter to remove high-frequency noise.
        """
        from scipy import signal
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data, axis=0)
    
    def normalize_data(self, data):
        """
        Normalize sensor data using z-score normalization.
        """
        return self.scaler.fit_transform(data)
    
    def create_sliding_windows(self, data, labels):
        """
        Create overlapping windows from continuous sensor data.
        
        Args:
            data: Array of shape (n_samples, n_features)
            labels: Array of shape (n_samples,)
            
        Returns:
            windowed_data: Array of shape (n_windows, window_size, n_features)
            window_labels: Array of shape (n_windows,)
        """
        step_size = int(self.window_size * (1 - self.overlap))
        n_samples = data.shape[0]
        n_windows = (n_samples - self.window_size) 
        
        windowed_data = []
        window_labels = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + self.window_size
            
            window = data[start_idx:end_idx]
            window_label = np.bincount(labels[start_idx:end_idx]).argmax()
            
            windowed_data.append(window)
            window_labels.append(window_label)
            
        return np.array(windowed_data), np.array(window_labels)
    
    def preprocess_pipeline(self, data, labels):
        """
        Complete preprocessing pipeline.
        """
    
        filtered_data = self.butter_lowpass_filter(data)
        
        normalized_data = self.normalize_data(filtered_data)
        
        windowed_data, window_labels = self.create_sliding_windows(
            normalized_data, labels
        )
        
        return windowed_data, window_label
