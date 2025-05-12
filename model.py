import numpy as np
import cv2
import time

class AdaptiveAppearanceModel:
    """Implements the adaptive appearance model using incremental PCA (IVT approach)"""
    
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.mean = None
        self.basis = None
        self.history = []
        self.update_counter = 0
        self.update_frequency = 3
        
    def update(self, image):
        """Update the model with a new image using incremental SVD"""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_vector = image.flatten()
        self.history.append(image_vector)
        if len(self.history) > 50:
            self.history.pop(0)
        
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self._update_subspace()
            self.update_counter = 0
        
    def _update_subspace(self):
        """Update subspace model using the history of images"""
        if not self.history:
            return
            
        data = np.array(self.history)
        self.mean = np.mean(data, axis=0)
        centered_data = data - self.mean
        
        try:
            U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
            self.basis = Vt[:min(self.n_components, Vt.shape[0]), :].T
        except np.linalg.LinAlgError:
            if self.basis is None:
                self.basis = np.zeros((centered_data.shape[1], min(self.n_components, centered_data.shape[0])))
    
    def get_reconstruction_error(self, image):
        """Calculate reconstruction error for an image"""
        if self.mean is None or self.basis is None:
            return 0.0
            
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        image_vector = image.flatten()
        centered = image_vector - self.mean
        projection = self.basis.T @ centered
        reconstruction = self.basis @ projection + self.mean
        error = np.sum((image_vector - reconstruction) ** 2) / len(image_vector)
        return error

class PoseSubspaceModel:
    """Implements the pose constraint model using a set of pose-specific PCA subspaces"""
    
    def __init__(self, n_poses=5, n_components=6):
        self.n_poses = n_poses
        self.n_components = n_components
        self.pose_models = []
        
    def train(self, pose_data):
        """Train pose subspaces with data from different poses"""
        for pose_images in pose_data:
            if len(pose_images) == 0:
                continue
                
            data = []
            for img in pose_images:
                if img is None:
                    continue
                if len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data.append(img.flatten())
                
            if not data:
                continue
                
            data = np.array(data)
            mean = np.mean(data, axis=0)
            centered_data = data - mean
            
            try:
                U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
                basis = Vt[:min(self.n_components, Vt.shape[0]), :].T
                self.pose_models.append({
                    'mean': mean,
                    'basis': basis
                })
            except np.linalg.LinAlgError:
                print("SVD failed for a pose subspace")
        
    def predict_distance(self, image):
        """Calculate the minimum distance to any pose subspace"""
        if not self.pose_models:
            return 0.0
            
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_vector = image.flatten()
        
        errors = []
        for model in self.pose_models:
            mean = model['mean']
            basis = model['basis']
            centered = image_vector - mean
            projection = basis.T @ centered
            reconstruction = basis @ projection + mean
            error = np.sum((image_vector - reconstruction) ** 2) / len(image_vector)
            errors.append(error)
        
        return min(errors) if errors else 0.0

class AlignmentConstraintModel:
    """Implements the alignment constraint model using an SVM classifier"""
    
    def __init__(self, feature_dim=68):
        self.feature_dim = feature_dim
        self.svm = None

    def train(self, aligned_faces, misaligned_faces):
        """Train the alignment constraint model with aligned and misaligned faces"""
        try:
            from sklearn.svm import LinearSVC
            
            if not aligned_faces or not misaligned_faces:
                print("Warning: Empty aligned or misaligned faces")
                X = np.zeros((10, self.feature_dim))
                y = np.zeros(10)
                self.svm = LinearSVC(max_iter=1000, verbose=1)
                self.svm.fit(X, y)
                return
            
            print(f"Processing {len(aligned_faces)} aligned faces and {len(misaligned_faces)} misaligned faces")
            
            start_time = time.time()
            X = []
            y = []
            total_samples = len(aligned_faces) + len(misaligned_faces)
            for i, face in enumerate(aligned_faces):
                if face is None:
                    continue
                if len(face.shape) > 2:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                feature = self._extract_features(face)
                if feature is not None:
                    X.append(feature)
                    y.append(1)
                if i % 100 == 0:
                    print(f"Processed {i}/{total_samples} aligned faces, elapsed: {time.time() - start_time:.2f}s")
                    
            for i, face in enumerate(misaligned_faces):
                if face is None:
                    continue
                if len(face.shape) > 2:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                feature = self._extract_features(face)
                if feature is not None:
                    X.append(feature)
                    y.append(-1)
                if i % 100 == 0:
                    print(f"Processed {i}/{total_samples} misaligned faces, elapsed: {time.time() - start_time:.2f}s")
            
            if len(X) == 0:
                print("Error: No valid features extracted")
                X = np.zeros((10, self.feature_dim))
                y = np.zeros(10)
            else:
                X = np.array(X)
                y = np.array(y)
                print(f"Training LinearSVC with data shape: X={X.shape}, y={y.shape}")
            
            self.svm = LinearSVC(max_iter=1000, verbose=1)
            self.svm.fit(X, y)
            print(f"SVM training completed in {time.time() - start_time:.2f}s")
            
        except ImportError:
            print("scikit-learn not available. Using placeholder SVM.")
            self.svm = None
        except Exception as e:
            print(f"Error during SVM training: {e}")
            self.svm = None
                
    def _extract_features(self, image):
        """Extract features from face image for alignment classification"""
        try:
            if image is None or image.size == 0:
                print("Warning: Empty image provided to feature extraction")
                return None
                
            resized = cv2.resize(image, (32, 32))
            features = resized.flatten()
            
            if len(features) == 0:
                print("Warning: Extracted empty feature vector")
                return None
                
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            elif len(features) > self.feature_dim:
                features = features[:self.feature_dim]
                
            return features
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None
        
    def calculate_confidence(self, image):
        """Calculate alignment confidence for a face image"""
        if self.svm is None:
            return 0.0
            
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = self._extract_features(image)
        
        try:
            confidence = self.svm.decision_function([features])[0]
            return confidence
        except:
            return 0.0