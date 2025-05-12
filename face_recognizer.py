import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import os
import pandas as pd
from typing import List, Dict, Tuple
import scipy.io as sio
import joblib
from dataset_loader import DatasetLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import glob

class FaceRecognizer:
    def __init__(self, config: dict, n_poses: int = 1, lda_dim: int = 6, desc_dim: int = 10):
        """Initialize Face Recognizer with HMM model."""
        self.config = config
        self.n_poses = n_poses
        self.lda_dim = lda_dim
        self.desc_dim = desc_dim
        self.data_loader = DatasetLoader(config)
        self.lda = LDA(n_components=lda_dim)
        self.pca = PCA(n_components=desc_dim)
        self.scaler = StandardScaler()
        self.hmm_models = {}
        self.subjects = []
        self.standard_face_size = tuple(config['dataset']['std_face_size'])

    def extract_lda_features(self, images: List[np.ndarray], pose_labels: List[int]) -> np.ndarray:
        """Extract LDA features from images."""
        X = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten() for img in images if img is not None])
        if len(X) == 0:
            return np.zeros((1, self.lda_dim))
        self.lda.fit(X, pose_labels[:len(X)])
        lda_features = self.lda.transform(X)
        return lda_features

    def extract_descriptor_features(self, descriptors: List[Dict]) -> np.ndarray:
        """Extract features from precomputed CSLBP descriptors."""
        if not descriptors:
            print("Warning: No descriptors provided")
            return np.zeros((1, self.desc_dim))
        
        desc_features = []
        for desc in descriptors:
            if desc is None or not isinstance(desc, dict):
                print("Warning: Invalid descriptor (None or not a dict)")
                continue
            cslbp = desc.get('VID_DESCS_CSLBP', None)
            if cslbp is None or not isinstance(cslbp, np.ndarray) or cslbp.size == 0:
                print("Warning: No valid CSLBP data in descriptor")
                continue
            desc_features.append(cslbp.flatten()[:480])
        
        if not desc_features:
            print("Warning: No valid CSLBP features extracted")
            return np.zeros((1, self.desc_dim))
        
        desc_features = np.array(desc_features)
        print(f"Extracted {len(desc_features)} CSLBP features with shape {desc_features.shape}")
        
        desc_features = self.scaler.fit_transform(desc_features)
        desc_features = self.pca.fit_transform(desc_features)
        return desc_features

    def train(self, train_data: Dict, pose_labels: List[int]):
        """Train HMM models for each subject."""
        images = train_data['images']
        descriptors = train_data['descriptors']
        meta = train_data['metadata']
        video_names = meta.get('video_names', [])

        lda_features = self.extract_lda_features(images, pose_labels)
        desc_features = self.extract_descriptor_features(descriptors)
        if len(desc_features) < len(lda_features):
            desc_features = np.repeat(desc_features, len(lda_features) // len(desc_features) + 1, axis=0)[:len(lda_features)]
        features = np.concatenate([lda_features, desc_features], axis=1)

        subject_data = {}
        for feat, vname in zip(features, video_names):
            if not isinstance(vname, str):
                continue
            subject = vname.split('/')[0]
            if subject not in subject_data:
                subject_data[subject] = []
            subject_data[subject].append(feat)
            if subject not in self.subjects:
                self.subjects.append(subject)

        for subject in subject_data:
            X = np.array(subject_data[subject])
            if len(X) < self.n_poses:
                print(f"Skipping {subject}: Not enough data ({len(X)} frames, need {self.n_poses})")
                continue
            lengths = [len(subject_data[subject])]
            model = hmm.GaussianHMM(
                n_components=max(1, min(self.n_poses, len(X))),  # Số components phụ thuộc số frame
                covariance_type="diag",
                n_iter=self.config['recognition']['n_iter_hmm'],
                init_params="stmc",
                params="stmc"
            )
            model.fit(X, lengths)
            self.hmm_models[subject] = model
            print(f"Trained HMM for {subject} with {len(X)} frames")

        np.save(os.path.join(self.config['dataset']['root_path'], "lda_features.npy"), lda_features)
        np.save(os.path.join(self.config['dataset']['root_path'], "cslbp_features.npy"), desc_features)

    def predict(self, video_dir: str, tracking_csv: str) -> Tuple[str, float]:
        """Predict subject identity for a sequence of frames."""
        tracking_df = pd.read_csv(tracking_csv)
        frame_paths = sorted(glob.glob(os.path.join(video_dir, 'aligned_detect_*.jpg')))
        if not frame_paths:
            print(f"Error: No frames found in {video_dir}")
            return None, -np.inf

        print(f"Found {len(frame_paths)} frames for prediction in {video_dir}")
        frames = []
        for idx, row in tracking_df.iterrows():
            if idx >= len(frame_paths):
                break
            frame = cv2.imread(frame_paths[idx])
            if frame is None:
                print(f"Error reading frame {frame_paths[idx]}")
                continue
            c_x, c_y, rho, phi = row['c_x'], row['c_y'], row['rho'], row['phi']
            box_size = int(min(frame.shape[0], frame.shape[1]) * rho)
            half = box_size // 2
            x1, y1 = int(c_x - half), int(c_y - half)
            x2, y2 = int(c_x + half), int(c_y + half)
            face = frame[max(0, y1):y2, max(0, x1):x2]
            if face.size > 0:
                face = cv2.resize(face, self.standard_face_size)
                frames.append(face)

        if not frames:
            print(f"Error: No valid faces extracted from {video_dir}")
            return None, -np.inf

        lda_features = self.extract_lda_features(frames, [0] * len(frames))
        desc_features = np.zeros((len(frames), self.desc_dim))
        features = np.concatenate([lda_features, desc_features], axis=1)

        max_score = -np.inf
        predicted_subject = None
        for subject, model in self.hmm_models.items():
            score = model.score(features)
            if score > max_score:
                max_score = score
                predicted_subject = subject

        return predicted_subject, max_score

    def evaluate(self, test_videos: List[str], output_dir: str) -> Dict:
        """Evaluate recognition accuracy on test videos."""
        os.makedirs(output_dir, exist_ok=True)
        true_labels = []
        pred_labels = []
        meta = sio.loadmat(self.config['dataset']['meta_data_path'])
        video_names = [name[0].item() for name in meta['video_names']]

        for video_dir in test_videos:
            folder_name = os.path.basename(os.path.dirname(video_dir))
            video_name = os.path.basename(video_dir)
            tracking_csv = os.path.join(output_dir, f"{folder_name}_{video_name}_tracking.csv")
            if not os.path.exists(tracking_csv):
                print(f"Skipping {video_dir}: No tracking CSV")
                continue

            true_subject = folder_name
            pred_subject, score = self.predict(video_dir, tracking_csv)
            if pred_subject is None:
                print(f"Skipping {video_dir}: Prediction failed")
                continue
            true_labels.append(true_subject)
            pred_labels.append(pred_subject)
            print(f"Video {folder_name}/{video_name}: True = {true_subject}, Predicted = {pred_subject}, Score = {score:.2f}")

        accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0.0
        cm = confusion_matrix(true_labels, pred_labels, labels=self.subjects) if true_labels else np.zeros((len(self.subjects), len(self.subjects)))

        with open(os.path.join(output_dir, "evaluation_results.txt"), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "subjects": self.subjects
        }

    def save_models(self, output_dir: str):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        for subject, model in self.hmm_models.items():
            np.save(os.path.join(output_dir, f"{subject}_hmm.npy"), model)
        joblib.dump(self.lda, os.path.join(output_dir, "lda_model.pkl"))
        joblib.dump(self.pca, os.path.join(output_dir, "pca_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))

    def load_models(self, model_dir: str):
        """Load trained models."""
        for subject in self.subjects:
            model_path = os.path.join(model_dir, f"{subject}_hmm.npy")
            if os.path.exists(model_path):
                self.hmm_models[subject] = np.load(model_path, allow_pickle=True).item()
        self.lda = joblib.load(os.path.join(model_dir, "lda_model.pkl"))
        self.pca = joblib.load(os.path.join(model_dir, "pca_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))