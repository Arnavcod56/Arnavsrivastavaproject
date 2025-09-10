class TopologicalFeatureExtractor:
    """
    Extracts topological features using persistent homology.
    """
    
    def __init__(self, max_dimension=2, coeff=2):
        self.max_dimension = max_dimension
        self.coeff = coeff
        
    def compute_persistence_diagram(self, point_cloud, max_edge_length=None):
        """
        Compute persistence diagram using Ripser.
        
        Args:
            point_cloud: Array of shape (n_points, dim)
            max_edge_length: Maximum edge length for Rips complex
            
        Returns:
            persistence_diagram: Persistence diagram
        """
        if max_edge_length is None:
            # Set max edge length based on data distribution
            distances = pdist(point_cloud)
            max_edge_length = np.percentile(distances, 90)
        
        result = ripser.ripser(
            point_cloud, 
            maxdim=self.max_dimension,
            thresh=max_edge_length,
            coeff=self.coeff
        )
        
        return result['dgms']
    
    def persistence_statistics(self, dgm):
        """
        Extract statistical features from persistence diagram.
        
        Args:
            dgm: Persistence diagram for specific dimension
            
        Returns:
            features: Statistical features
        """
        if len(dgm) == 0:
            return np.zeros(6)
        
        # Remove infinite persistence (connected components)
        finite_dgm = dgm[dgm[:, 1] < np.inf]
        
        if len(finite_dgm) == 0:
            return np.zeros(6)
        
        births = finite_dgm[:, 0]
        deaths = finite_dgm[:, 1]
        lifetimes = deaths - births
        
        features = np.array([
            len(finite_dgm),  # Number of topological features
            np.mean(lifetimes),  # Mean persistence
            np.std(lifetimes),  # Std persistence
            np.max(lifetimes),  # Max persistence
            np.mean(births),  # Mean birth time
            np.mean(deaths)  # Mean death time
        ])
        
        return features
    
    def persistence_landscape(self, dgm, resolution=100):
        """
        Compute persistence landscape (vectorized representation).
        
        Args:
            dgm: Persistence diagram
            resolution: Number of points in landscape
            
        Returns:
            landscape: Persistence landscape vector
        """
        if len(dgm) == 0:
            return np.zeros(resolution)
        
        finite_dgm = dgm[dgm[:, 1] < np.inf]
        if len(finite_dgm) == 0:
            return np.zeros(resolution)
        
        births = finite_dgm[:, 0]
        deaths = finite_dgm[:, 1]
     
        min_val = np.min(births)
        max_val = np.max(deaths)
        x_vals = np.linspace(min_val, max_val, resolution)
        
        landscape = np.zeros(resolution)
        for i, x in enumerate(x_vals):
            heights = []
            for b, d in zip(births, deaths):
                if b <= x <= d:
                    # Triangle function: min(x-b, d-x)
                    heights.append(min(x - b, d - x))
            
            if heights:
                landscape[i] = max(heights)
        
        return landscape
    
    def extract_topological_features(self, point_cloud):
        """
        Extract complete topological feature vector.
        
        Args:
            point_cloud: 3D point cloud
            
        Returns:
            features: Combined topological features
        """
       
        dgms = self.compute_persistence_diagram(point_cloud)
        
        all_features = []
        
        for dim in range(self.max_dimension + 1):
            if dim < len(dgms):
                dgm = dgms[dim]
                
                stats = self.persistence_statistics(dgm)
                
                landscape = self.persistence_landscape(dgm, resolution=20)
                
                dim_features = np.concatenate([stats, landscape])
            else:
                dim_features = np.zeros(26)  # 6 stats + 20 landscape
            
            all_features.append(dim_features)
        
        return np.concatenate(all_features)
