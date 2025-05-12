import numpy as np
from scipy.ndimage import affine_transform
import cv2

class FaceTracker:
    def __init__(self, params: dict):
        # Models for tracking constraints
        self.pose_subspace_model = params.get('pose_subspace_model')
        self.alignment_constraint_model = params.get('alignment_constraint_model')
        self.adaptive_appearance_model = params.get('adaptive_appearance_model')
        
        # Weighting factors for the energy components
        self.lambda_a = params.get('lambda_a', 1.0)  # Weight for adaptive term
        self.lambda_p = params.get('lambda_p', 1.0)  # Weight for pose constraint
        self.lambda_s = params.get('lambda_s', 1.0)  # Weight for alignment constraint
        
        # Particle filtering parameters
        self.n_particles = params.get('n_particles', 100)
        self.sigma = params.get('sigma', 0.1)  # For emission probability
        
        # Fixed: Reduced dynamics covariance for more stable tracking
        # self.dynamics_cov = params.get('dynamics_cov', np.diag([0.5, 0.5, 0.001, 0.001]))  # Î£ covariance for dynamics
        self.dynamics_cov = params.get('dynamics_cov', np.diag([2.0, 2.0, 0.01, 0.01]))  # Î£ covariance for dynamics

        # Standard size for cropped face images
        self.std_size = params.get('std_size', (48, 48))
        
        # Initialize particles and weights
        self.particles = None
        self.weights = None
        self.prev_state = None
        self.history = []  # To store previous tracked face images
        
        # Track quality metrics
        self.track_lost_counter = 0
        self.max_lost_frames = 30  # Reset tracking if lost for this many frames
        self.energy_threshold = 10  # Energy threshold for detecting lost track
        self.prev_energy = 0
        
        # Face detector for re-initialization
        self.face_detector = None
        try:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Warning: Could not load face detector. Automatic recovery disabled.")
    
    def initialize(self, initial_state):
        """Initialize the tracker with the first frame's state"""
        self.prev_state = initial_state
        self.particles = np.zeros((self.n_particles, 4))
        
        # Initialize particles around the initial state with less noise
        for i in range(self.n_particles):
            noise = np.random.multivariate_normal(np.zeros(4), self.dynamics_cov * 0.1)
            self.particles[i] = initial_state + noise
            
        # Equal initial weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.track_lost_counter = 0
        self.prev_energy = 0
        
        # Initialize smoothing
        self.smoothed_state = initial_state.copy() if initial_state is not None else None

    def track_face(self, image: np.array) -> dict:
        """Track face in the given image frame with state smoothing and track quality assessment"""
        if self.prev_state is None:
            raise ValueError("Tracker must be initialized first with initial_state")

        # Apply particle filtering to get raw result
        raw_result = self.apply_particle_filter(image)
        current_state = np.array([
            raw_result['tracking_state']['c_x'],
            raw_result['tracking_state']['c_y'],
            raw_result['tracking_state']['rho'],
            raw_result['tracking_state']['phi']
        ])
        
        # Get current energy
        current_energy = raw_result['energy']
        
        # Check if tracking is lost
        is_track_lost = self._check_if_tracking_lost(current_energy, current_state, image)
        
        if is_track_lost and self.face_detector is not None:
            # Try to reinitialize with face detector
            success = self._reinitialize_with_face_detector(image)
            if success:
                # Restart tracking with new detection
                raw_result = self.apply_particle_filter(image)
                current_state = np.array([
                    raw_result['tracking_state']['c_x'],
                    raw_result['tracking_state']['c_y'],
                    raw_result['tracking_state']['rho'],
                    raw_result['tracking_state']['phi']
                ])
                current_energy = raw_result['energy']
                self.track_lost_counter = 0
            else:
                self.track_lost_counter += 1
        else:
            self.track_lost_counter = 0 if current_energy < self.energy_threshold else self.track_lost_counter + 1

        # Initialize smoothed_state on first call
        if not hasattr(self, 'smoothed_state') or self.smoothed_state is None:
            self.smoothed_state = current_state.copy()

        # Adaptive smoothing factor based on energy
        # Less smoothing when energy is low (good track), more when it's high (uncertain track)
        base_alpha = 0.5  # base smoothing factor
        energy_factor = min(1.0, current_energy / self.energy_threshold)
        alpha = base_alpha * (1 - energy_factor * 0.5)  # Adaptive alpha between 0.25 and 0.5
        
        # Exponential moving average for smoothing
        self.smoothed_state = alpha * current_state + (1 - alpha) * self.smoothed_state

        # Enforce constraints on smoothed state
        self.smoothed_state[2] = np.clip(self.smoothed_state[2], 0.1, 0.5)  # Clamp scale factor
        
        # Update prev_state to smoothed one for next propagation
        self.prev_state = self.smoothed_state.copy()
        self.prev_energy = current_energy

        # Warp image using smoothed state
        sm_cx, sm_cy, sm_rho, sm_phi = self.smoothed_state
        cropped_face = self.warp_image(image, self.smoothed_state)

        # Update history
        if len(self.history) > 10:
            self.history.pop(0)
        self.history.append(cropped_face)

        # Update appearance model
        if self.adaptive_appearance_model:
            self.adaptive_appearance_model.update(cropped_face)

        # In log táº¥t cáº£ cÃ¡c thÃ´ng sá»‘ quan trá»ng táº¡i má»—i frame
        print("------ TRACKER LOG ------")
        print(f"Current state (smoothed): c_x={self.smoothed_state[0]:.2f}, c_y={self.smoothed_state[1]:.2f}, rho={self.smoothed_state[2]:.4f}, phi={self.smoothed_state[3]:.4f}")
        print(f"Prev state: c_x={self.prev_state[0]:.2f}, c_y={self.prev_state[1]:.2f}, rho={self.prev_state[2]:.4f}, phi={self.prev_state[3]:.4f}")
        print(f"Energy: {current_energy:.2f} (prev: {self.prev_energy:.2f})")
        print(f"Track lost counter: {self.track_lost_counter}")
        print(f"Particle mean: {np.mean(self.particles, axis=0)}")
        print(f"Particle std: {np.std(self.particles, axis=0)}")
        print(f"First 5 particles:\n{self.particles[:5]}")
        print(f"First 5 weights: {self.weights[:5]}")
        print("-------------------------")

        # Build and return result dict based on smoothed state
        return {
            'tracking_state': {
                'c_x': sm_cx,
                'c_y': sm_cy,
                'rho': sm_rho,
                'phi': sm_phi
            },
            'energy': current_energy,
            'cropped_face': cropped_face,
            'is_track_stable': not is_track_lost
        }
    
    def _check_if_tracking_lost(self, energy, current_state, image):
        """Check if tracking is likely lost based on energy and state"""
        h, w = image.shape[:2]
        
        # Check if energy is too high
        energy_too_high = energy > self.energy_threshold
        
        # Check if face center is outside the image or too close to the edge
        cx, cy = current_state[0], current_state[1]
        border = 20  # Minimum distance from edge
        position_invalid = (cx < border or cx > w - border or cy < border or cy > h - border)
        
        # Check if scale is too small or too large
        rho = current_state[2]
        scale_invalid = (rho < 0.05 or rho > 0.6)
        
        # Check if energy increased dramatically
        energy_jump = energy > self.prev_energy * 3 and self.prev_energy > 0
        
        return energy_too_high or position_invalid or scale_invalid or energy_jump
    
    def _reinitialize_with_face_detector(self, image):
        """Try to reinitialize tracking using face detector"""
        if self.face_detector is None:
            return False
            
        # Convert to grayscale for face detection
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect faces
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Use the largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            height, width = image.shape[:2]
            
            # Calculate new state
            c_x = x + w/2
            c_y = y + h/2
            rho = max(w, h) / min(width, height)
            phi = 0.0  # Reset rotation
            
            # Initialize with new state
            new_state = np.array([c_x, c_y, rho, phi])
            self.prev_state = new_state
            self.smoothed_state = new_state.copy()
            
            # Reinitialize particles
            for i in range(self.n_particles):
                noise = np.random.multivariate_normal(np.zeros(4), self.dynamics_cov * 0.1)
                self.particles[i] = new_state + noise
                
            self.weights = np.ones(self.n_particles) / self.n_particles
            return True
            
        return False
    
    def apply_particle_filter(self, image: np.array) -> dict:
        """Apply particle filtering to estimate tracking state"""
        # 1. Propagate particles according to dynamics model: P(u_t|u_{t-1})
        self.propagate_particles()
        
        # 2. Compute weights based on observation model: P(F_t|u_t)
        self.compute_particle_weights(image)
        
        # 3. Estimate current state from weighted particles
        current_state = self.estimate_state_from_particles()
        current_state[2] = np.clip(current_state[2], 0.1, 0.5)  # Clamp scale
        
        # 4. Resample particles
        self.resample_particles()
        
        # 5. Update previous state
        self.prev_state = current_state
        
        # 6. Crop the face using the estimated state
        cropped_face = self.warp_image(image, current_state)
        
        # 7. Calculate energy value
        energy = self.calculate_energy(cropped_face)
        
        # 8. Normalize energy to a more reasonable range
        energy = min(energy, 10000)  # Cap extremely high energy values
        
        return {
            "tracking_state": {
                "c_x": current_state[0],
                "c_y": current_state[1],
                "rho": current_state[2],
                "phi": current_state[3]
            },
            "energy": energy,
            "cropped_face": cropped_face
        }
    
    def propagate_particles(self):
        """Propagate particles according to dynamics model P(u_t|u_{t-1})"""
        for i in range(self.n_particles):
            # Add Gaussian noise according to dynamics covariance
            noise = np.random.multivariate_normal(np.zeros(4), self.dynamics_cov)
            self.particles[i] = self.particles[i] + noise
            
            # Enforce reasonable bounds on particles
            self.particles[i, 2] = np.clip(self.particles[i, 2], 0.05, 0.6)  # Bound scale
    
    def compute_particle_weights(self, image: np.array):
        """Compute weights for particles based on emission probability P(F_t|u_t)"""
        new_weights = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            # Warp the image according to the particle state
            cropped_face = self.warp_image(image, self.particles[i])
            
            # Calculate energy for this particle
            energy = self.calculate_energy(cropped_face)
            
            # Cap extremely high energy values
            energy = min(energy, 10000)
            
            # Convert energy to emission probability: P(F_t|u_t) âˆ exp(-E(Ï‰(u_t,F_t);Î¸)/ÏƒÂ²)
            emission_prob = np.exp(-energy / (self.sigma ** 2))
            new_weights[i] = emission_prob
        
        # Normalize weights
        sum_weights = np.sum(new_weights)
        if sum_weights > 0:
            self.weights = new_weights / sum_weights
        else:
            # If all weights are zero, reinitialize with uniform weights
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate_state_from_particles(self):
        """Estimate tracking state from weighted particles"""
        # Weighted average of particles
        estimated_state = np.sum(self.particles.T * self.weights, axis=1)
        return estimated_state
    
    def resample_particles(self):
        """Resample particles according to their weights with added jitter"""
        # Systematic resampling
        N = self.n_particles
        positions = (np.arange(N) + np.random.random()) / N
        
        # Compute cumulative sum of weights
        cumsum = np.cumsum(self.weights)
        
        # Ensure cumsum[-1] is exactly 1 to avoid numerical issues
        if abs(cumsum[-1] - 1.0) > 1e-10:
            cumsum = cumsum / cumsum[-1]
        
        # New particle set
        new_particles = np.zeros_like(self.particles)
        
        # Resampling with small jitter
        i, j = 0, 0
        while i < N:
            if positions[i] < cumsum[j]:
                new_particles[i] = self.particles[j] + np.random.multivariate_normal(np.zeros(4), self.dynamics_cov * 0.01)
                i += 1
            else:
                j += 1
        
        # Update particles
        self.particles = new_particles
        
        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def warp_image(self, image: np.array, state: np.array) -> np.array:
        """Implementation of the warping function Ï‰(u_t, F_t)
        
        Args:
            image: Input image frame
            state: Tracking state [c_x, c_y, rho, phi]
            
        Returns:
            Cropped face image according to the state
        """
        c_x, c_y, rho, phi = state
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Calculate the size of the bounding box
        box_size = int(min(h, w) * rho)
        if box_size < 10:  # Minimum reasonable size
            box_size = 10
            
        # Create transformation matrix
        # 1. Translation to center
        translation_to_center = np.array([
            [1, 0, -c_x],
            [0, 1, -c_y],
            [0, 0, 1]
        ])
        
        # 2. Rotation by phi
        rotation = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        
        # 3. Scaling by 1/rho
        scaling = np.array([
            [1/rho, 0, 0],
            [0, 1/rho, 0],
            [0, 0, 1]
        ])
        
        # 4. Translation to standard size center
        std_h, std_w = self.std_size
        translation_to_std = np.array([
            [1, 0, std_w/2],
            [0, 1, std_h/2],
            [0, 0, 1]
        ])
        
        # Combined transformation
        transform = translation_to_std @ scaling @ rotation @ translation_to_center
        transform = transform[:2, :]  # Remove the last row
        
        # Apply the transformation to get the cropped face
        try:
            cropped_face = cv2.warpAffine(image, transform, self.std_size)
            return cropped_face
        except Exception as e:
            print(f"Error in warping: {e}")
            # Return a black image in case of error
            return np.zeros((*self.std_size, 3) if len(image.shape) == 3 else self.std_size)
    
    def calculate_energy(self, image: np.array) -> float:
        """Calculate the energy function combining adaptive term with visual constraints
        
        E(I_t) = Î»_a * d(I_t, M_a(I_0...t-1)) + Î»_p * d(I_t, M_p) - Î»_s * f_s(I_t)
        
        We normalize each component to prevent any one term from dominating.
        """
        # Component 1: Adaptive appearance model term
        adaptive_term = self.calculate_adaptive_term(image)
        
        # Component 2: Pose constraint term
        pose_distance = 0.0
        if self.pose_subspace_model:
            pose_distance = self.pose_subspace_model.predict_distance(image)
        
        # Component 3: Alignment constraint term
        alignment_confidence = 0.0
        if self.alignment_constraint_model:
            alignment_confidence = self.alignment_constraint_model.calculate_confidence(image)
        
        # Normalization factors (learned from typical values)
        norm_adaptive = max(1.0, adaptive_term) / 1000.0
        norm_pose = max(1.0, pose_distance) / 1000.0
        norm_alignment = max(0.1, abs(alignment_confidence))
        
        # Combine the components with weighting factors and normalization
        energy = (self.lambda_a * norm_adaptive + 
                  self.lambda_p * norm_pose - 
                  self.lambda_s * alignment_confidence / norm_alignment)
        # print(f"Energy components: adaptive={norm_adaptive:.4f}, pose={norm_pose:.4f}, alignment={alignment_confidence:.4f}")

        return energy
    
    def calculate_adaptive_term(self, image: np.array) -> float:
        """Calculate the adaptive term for the energy function
        
        This uses the IVT approach: reconstruction error from PCA subspace
        """
        if not self.adaptive_appearance_model or len(self.history) == 0:
            return 0.0
        
        # Get the reconstruction error from the adaptive appearance model
        reconstruction_error = self.adaptive_appearance_model.get_reconstruction_error(image)
        return reconstruction_error
    
    def set_params(self, params: dict):
        """Update tracker parameters"""
        if 'lambda_a' in params:
            self.lambda_a = params['lambda_a']
        if 'lambda_p' in params:
            self.lambda_p = params['lambda_p']
        if 'lambda_s' in params:
            self.lambda_s = params['lambda_s']
        if 'n_particles' in params:
            self.n_particles = params['n_particles']
        if 'sigma' in params:
            self.sigma = params['sigma']
        if 'dynamics_cov' in params:
            self.dynamics_cov = params['dynamics_cov']
        if 'energy_threshold' in params:
            self.energy_threshold = params['energy_threshold']