import numpy as np
import os
import cv2
import glob
from typing import List, Dict, Tuple, Any
import scipy.io as sio
from multiprocessing import Pool

class DatasetLoader:
    def __init__(self, config: dict):
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        self.pose_clusters = config.get('training', {}).get('pose_clusters', 5)
        self.pose_subspace_dimension = config.get('training', {}).get('pose_subspace_dimension', 10)
        self.lmt_features_dimension = config.get('training', {}).get('lmt_features_dimension', 68)
        self.pca_reduced_dimension = config.get('training', {}).get('pca_reduced_dimension', 50)
        self.dataset_root = os.path.expanduser(config.get('dataset', {}).get('root_path', '~/datasets'))
        self.dataset_name = config.get('dataset', {}).get('name', 'YouTubeFaces')
        self.aligned_images_path = config.get('dataset', {}).get('frame_images_path', '~/datasets/YouTubeFaces/aligned_images_DB')
        self.meta_data_path = config.get('dataset', {}).get('meta_data_path', '~/datasets/YouTubeFaces/meta_data/meta_and_splits.mat')
        self.descriptors_path = config.get('dataset', {}).get('descriptors_path', '~/datasets/YouTubeFaces/descriptors_DB')
        self.cached_data = None
        self.standard_face_size = tuple(config.get('dataset', {}).get('std_face_size', (48, 48)))
        print(f"Debug: dataset_root = {self.dataset_root}")
        print(f"Debug: dataset_name = {self.dataset_name}")

    def load_splits(self):
        meta = sio.loadmat(self.meta_data_path)
        splits = meta['Splits']
        return splits

    def load_data(self) -> dict:
        if self.cached_data is not None:
            return self.cached_data

        landmarks_data = self.load_landmarks()
        images_data = self.load_images(landmarks_data)
        descriptors_data = self.load_descriptors()
        pose_labels = self.load_headpose()
        pose_clusters = self.cluster_by_pose(images_data, landmarks_data)
        aligned_faces, misaligned_faces = self.generate_alignment_samples(images_data)
        
        preprocessed_data = {
            "images": images_data,
            "landmarks": landmarks_data,
            "descriptors": descriptors_data,
            "pose_labels": pose_labels,
            "pose_clusters": pose_clusters,
            "aligned_faces": aligned_faces,
            "misaligned_faces": misaligned_faces,
            "metadata": {
                "clusters": self.pose_clusters,
                "subspace_dim": self.pose_subspace_dimension,
                "lmt_dim": self.lmt_features_dimension,
                "pca_dim": self.pca_reduced_dimension,
                "video_names": [name[0].item() for name in sio.loadmat(self.meta_data_path)['video_names']]
            }
        }
        self.cached_data = preprocessed_data
        return preprocessed_data

    def load_landmarks(self) -> List[np.ndarray]:
        landmarks_data = []
        meta_path = os.path.join(self.dataset_root, self.dataset_name, 'meta_data', 'meta_and_splits.mat')
        if not os.path.exists(self.descriptors_path):
            print(f"Error: descriptors_DB not found at {self.descriptors_path}")
            return landmarks_data
        if not os.path.exists(meta_path):
            print(f"Error: Meta data file not found at {meta_path}")
            return landmarks_data

        try:
            meta = sio.loadmat(meta_path)
            video_names = [name[0].item() for name in meta['video_names']]
        except Exception as e:
            print(f"Error loading meta data: {e}")
            return landmarks_data

        subjects = sorted(set([name.split('/')[0] for name in video_names]))
        max_subjects = 100  # Tăng để load thêm dữ liệu
        subjects = subjects[:min(max_subjects, len(subjects))]
        print(f"Loading landmarks for {len(subjects)} subjects: {subjects[:5]}...")

        for subject in subjects:
            mat_dir = os.path.join(self.descriptors_path, subject)
            mat_files = glob.glob(os.path.join(mat_dir, "*.mat"))
            for mat_file in mat_files:
                data = sio.loadmat(mat_file)
                landmarks = data.get('landmarks', np.array([[0.5, 0.5]]))
                landmarks_data.append(landmarks)
        print(f"Loaded {len(landmarks_data)} landmarks")
        return landmarks_data

    def load_images(self, landmarks_data: List[np.ndarray]) -> List[np.ndarray]:
        images_data = []
        frame_images_path = os.path.join(self.dataset_root, self.dataset_name, 'aligned_images_DB')
        meta_path = os.path.join(self.dataset_root, self.dataset_name, 'meta_data', 'meta_and_splits.mat')
        
        if not os.path.exists(frame_images_path):
            print(f"Error: aligned_images_DB not found at {frame_images_path}")
            return images_data

        try:
            meta = sio.loadmat(meta_path)
            video_names = [name[0].item() for name in meta['video_names']]
        except Exception as e:
            print(f"Error loading meta data: {e}")
            return images_data

        subjects = sorted(set([name.split('/')[0] for name in video_names]))
        max_subjects = 100  # Tăng để load thêm dữ liệu
        subjects = subjects[:min(max_subjects, len(subjects))]
        print(f"Loading images for {len(subjects)} subjects: {subjects[:5]}...")

        for subject in subjects:
            video_dirs = glob.glob(os.path.join(self.aligned_images_path, subject, "*"))
            for video_dir in video_dirs:
                image_files = sorted(glob.glob(os.path.join(video_dir, 'aligned_detect_*.jpg')))
                print(f"Found {len(image_files)} frames for {subject}/{os.path.basename(video_dir)}")
                for img_path in image_files:  # Load tất cả frame
                    frame = cv2.imread(img_path)
                    if frame is None:
                        print(f"Error reading image {img_path}")
                        continue
                    face_img = cv2.resize(frame, self.standard_face_size)
                    images_data.append(face_img)
        
        print(f"Loaded {len(images_data)} images")
        return images_data

    def load_descriptor_file(self, mat_file: str) -> Dict:
        """Load a single descriptor file."""
        try:
            return sio.loadmat(mat_file)
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            return None

    def load_descriptors(self) -> List[np.ndarray]:
        """Load CSLBP descriptors from descriptors_DB using multiprocessing."""
        meta_path = os.path.join(self.dataset_root, self.dataset_name, 'meta_data', 'meta_and_splits.mat')
        
        if not os.path.exists(self.descriptors_path):
            print(f"Error: descriptors_DB not found at {self.descriptors_path}")
            return []

        try:
            meta = sio.loadmat(meta_path)
            video_names = [name[0].item() for name in meta['video_names']]
        except Exception as e:
            print(f"Error loading meta data: {e}")
            return []

        subjects = sorted(set([name.split('/')[0] for name in video_names]))
        max_subjects = 100  # Tăng để load thêm dữ liệu
        subjects = subjects[:min(max_subjects, len(subjects))]
        print(f"Loading descriptors for {len(subjects)} subjects: {subjects[:5]}...")

        mat_files = []
        for subject in subjects:
            mat_dir = os.path.join(self.descriptors_path, subject)
            mat_files.extend(glob.glob(os.path.join(mat_dir, "aligned_*.mat")))

        with Pool(processes=4) as pool:
            descriptors_data = pool.map(self.load_descriptor_file, mat_files)
        descriptors_data = [d for d in descriptors_data if d is not None]
        
        print(f"Loaded {len(descriptors_data)} descriptors")
        return descriptors_data

    def load_headpose(self) -> List[int]:
        """Load headpose data and assign pose labels."""
        headpose_path = os.path.join(self.dataset_root, self.dataset_name, 'headpose_DB')
        pose_labels = []
        meta = sio.loadmat(self.meta_data_path)
        video_names = [name[0].item() for name in meta['video_names']]
        
        for vname in video_names:
            subject, vid_num = vname.split('/')
            hp_file = os.path.join(headpose_path, f"headorient_apirun_{subject}_{vid_num}.mat")
            if os.path.exists(hp_file):
                hp_data = sio.loadmat(hp_file)
                headpose = hp_data['headpose']
                for i in range(headpose.shape[1]):
                    yaw, pitch, _ = headpose[:, i]
                    if abs(pitch) > 30:
                        pose = 5 if pitch > 0 else 6
                    elif abs(yaw) < 15:
                        pose = 0
                    elif abs(yaw) < 45:
                        pose = 1 if yaw > 0 else 2
                    else:
                        pose = 3 if yaw > 0 else 4
                    pose_labels.append(pose)
        print(f"Loaded {len(pose_labels)} pose labels")
        return pose_labels

    def read_pts_file(self, pts_file: str) -> np.ndarray:
        try:
            with open(pts_file, 'r') as f:
                lines = f.readlines()
            points = []
            n_points = 0
            reading_points = False
            for line in lines:
                line = line.strip()
                if line.startswith('version:'):
                    continue
                elif line.startswith('n_points:'):
                    n_points = int(line.split(':')[1].strip())
                    continue
                elif line == '{':
                    reading_points = True
                    continue
                elif line == '}':
                    reading_points = False
                    continue
                if reading_points and line:
                    try:
                        x, y = map(float, line.split())
                        points.append([x, y])
                    except ValueError:
                        continue
            if len(points) == n_points:
                return np.array(points)
            else:
                print(f"Warning: Expected {n_points} points but read {len(points)} from {pts_file}")
                return None
        except Exception as e:
            print(f"Error reading {pts_file}: {e}")
            return None

    def cluster_by_pose(self, images: List[np.ndarray], landmarks: List[np.ndarray]) -> List[List[np.ndarray]]:
        pose_clusters = [[] for _ in range(self.pose_clusters)]
        for i, (image, lm) in enumerate(zip(images, landmarks)):
            if len(lm) == 0:
                continue
            cluster_idx = i % self.pose_clusters
            pose_clusters[cluster_idx].append(image)
        return pose_clusters

    def generate_alignment_samples(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        aligned_faces = []
        misaligned_faces = []
        if not images:
            print("Warning: No images provided for alignment samples")
            return aligned_faces, misaligned_faces
        
        for image in images:
            if image is None or image.size == 0:
                continue
            aligned_faces.append(image.copy())
            for _ in range(2):
                rows, cols = image.shape[:2]
                tx = np.random.randint(-cols//6, cols//6)
                ty = np.random.randint(-rows//6, rows//6)
                scale = 0.7 + np.random.random() * 0.6
                angle = np.random.randint(-45, 45)
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                misaligned_img = cv2.warpAffine(image, M, (cols, rows))
                misaligned_faces.append(misaligned_img)
        
        print(f"Generated {len(aligned_faces)} aligned faces and {len(misaligned_faces)} misaligned faces")
        return aligned_faces, misaligned_faces

    def get_video_paths(self) -> List[str]:
        video_paths = []
        frame_images_path = os.path.join(self.dataset_root, self.dataset_name, 'aligned_images_DB')
        meta_path = os.path.join(self.dataset_root, self.dataset_name, 'meta_data', 'meta_and_splits.mat')
        
        if not os.path.exists(frame_images_path):
            print(f"Error: aligned_images_DB not found at {frame_images_path}")
            return video_paths

        try:
            meta = sio.loadmat(meta_path)
            video_names = [name[0].item() for name in meta['video_names']]
        except Exception as e:
            print(f"Error loading meta data: {e}")
            return video_paths

        subjects = sorted(set([name.split('/')[0] for name in video_names]))
        max_subjects = 100  # Tăng để load thêm dữ liệu
        subjects = subjects[:min(max_subjects, len(subjects))]
        print(f"Getting video paths for {len(subjects)} subjects: {subjects[:5]}...")

        for subject in subjects:
            subject_videos = [name for name in video_names if name.startswith(subject)]
            for video_name in subject_videos:
                video_dir = os.path.join(frame_images_path, video_name.replace('/', os.sep))
                if os.path.exists(video_dir):
                    video_paths.append(video_dir)
                else:
                    print(f"Video directory not found: {video_dir}")
        
        print(f"Found {len(video_paths)} video paths")
        return video_paths
    