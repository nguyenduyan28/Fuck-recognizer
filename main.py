import os
import numpy as np
import yaml
import cv2
import argparse
import pandas as pd
from dataset_loader import DatasetLoader
from face_tracker import FaceTracker
from model import AdaptiveAppearanceModel, PoseSubspaceModel, AlignmentConstraintModel
from face_recognizer import FaceRecognizer
import scipy.io as sio
import glob

def main():
    parser = argparse.ArgumentParser(
        description='Face Tracking and Recognition Implementation based on IVT with Visual Constraints')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--video', type=str, default='', help='Path to input video directory (optional)')
    parser.add_argument('--output', type=str, default='output', help='Path to output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize tracking results')
    parser.add_argument('--lambda_a', type=float, default=None, help='Override lambda_a for tracking')
    parser.add_argument('--lambda_p', type=float, default=None, help='Override lambda_p for tracking')
    parser.add_argument('--lambda_s', type=float, default=None, help='Override lambda_s for tracking')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    os.makedirs(args.output, exist_ok=True)

    if args.lambda_a is not None:
        config['tracking']['lambda_a'] = args.lambda_a
    if args.lambda_p is not None:
        config['tracking']['lambda_p'] = args.lambda_p
    if args.lambda_s is not None:
        config['tracking']['lambda_s'] = args.lambda_s

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Loading dataset...")
    data_loader = DatasetLoader(config)
    print("Training face tracking models...")
    training_data = data_loader.load_data()

    pose_model = PoseSubspaceModel(
        n_poses=config['training']['pose_clusters'],
        n_components=config['training']['pose_subspace_dimension']
    )
    pose_model.train(training_data['pose_clusters'])

    alignment_model = AlignmentConstraintModel(
        feature_dim=config['training']['lmt_features_dimension']
    )
    alignment_model.train(
        training_data['aligned_faces'],
        training_data['misaligned_faces']
    )

    adaptive_model = AdaptiveAppearanceModel(n_components=config['tracking']['adaptive_components'])

    tracker_params = {
        'pose_subspace_model': pose_model,
        'alignment_constraint_model': alignment_model,
        'adaptive_appearance_model': adaptive_model,
        'lambda_a': config['tracking']['lambda_a'],
        'lambda_p': config['tracking']['lambda_p'],
        'lambda_s': config['tracking']['lambda_s'],
        'n_particles': config['tracking']['n_particles'],
        'sigma': config['tracking']['sigma'],
        'std_size': tuple(config['dataset']['std_face_size'])
    }
    face_tracker = FaceTracker(tracker_params)

    face_recognizer = FaceRecognizer(config)
    print("Training face recognition models...")
    pose_labels = training_data.get('pose_labels', [i % config['training']['pose_clusters'] for i in range(len(training_data['images']))])
    face_recognizer.train(training_data, pose_labels)

    # Load test split
    meta = sio.loadmat(config['dataset']['meta_data_path'])
    splits = meta['Splits']  # Giả sử splits có shape (10, 1)
    test_videos = []
    for split_idx in range(splits.shape[0]):  # Duyệt qua 10 splits
        split = splits[split_idx, 0]  # Truy cập split thứ idx
        try:
            test_field = split['test'][0][0]  # Truy cập đúng cấu trúc
            test_names = [name[0] for name in test_field]
            if test_field.dtype.names:  # Scalar struct
                test_names = [name[0] for name in test_field[0][0]]
            else:  # Array of names
                test_names = [name[0] for name in test_field[0]]
            print(f"Split {split_idx}: Found {len(test_names)} test videos")
            frame_images_path = os.path.join(config['dataset']['root_path'], config['dataset']['name'], 'aligned_images_DB')
            for name in test_names:
                video_dir = os.path.join(frame_images_path, name.replace('/', os.sep))
                if os.path.exists(video_dir):
                    test_videos.append(video_dir)
                else:
                    print(f"Video directory not found: {video_dir}")
        except Exception as e:
            print(f"Error loading test split {split_idx}: {e}")
            continue
    print(f"Loaded {len(test_videos)} test videos")

    if args.video:
        print(f"Tracking and recognizing faces in video directory: {args.video}")
        track_video(face_tracker, args.video, args.output, args.visualize, face_cascade)
        tracking_csv = os.path.join(args.output, f"{os.path.basename(os.path.dirname(args.video))}_{os.path.basename(args.video)}_tracking.csv")
        if os.path.exists(tracking_csv):
            subject, score = face_recognizer.predict(args.video, tracking_csv)
            print(f"Video {args.video}: Predicted subject = {subject}, Score = {score:.2f}")
    else:
        print("Tracking and recognizing faces in all videos in the dataset...")
        video_paths = data_loader.get_video_paths()
        if not video_paths:
            print("No videos found")
        else:
            for video_path in video_paths:
                print(f"Processing video: {video_path}")
                track_video(face_tracker, video_path, args.output, args.visualize, face_cascade)
                folder_name = os.path.basename(os.path.dirname(video_path))
                video_name = os.path.basename(video_path)
                tracking_csv = os.path.join(args.output, f"{folder_name}_{video_name}_tracking.csv")
                if os.path.exists(tracking_csv):
                    subject, score = face_recognizer.predict(video_path, tracking_csv)
                    print(f"Video {folder_name}/{video_name}: Predicted subject = {subject}, Score = {score:.2f}")

    print("Evaluating recognition accuracy...")
    eval_results = face_recognizer.evaluate(test_videos, args.output)
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(eval_results['confusion_matrix'])

    print("Face tracking and recognition completed.")

def track_video(face_tracker, video_dir, output_dir, visualize=False, face_cascade=None):
    # video_dir là thư mục chứa các frame ảnh (e.g., Akmal_Taher/2)
    frame_paths = sorted(glob.glob(os.path.join(video_dir, 'aligned_detect_*.jpg')))
    if not frame_paths:
        print(f"Error: No frames found in {video_dir}")
        return

    print(f"Found {len(frame_paths)} frames in {video_dir}")

    # Đọc frame đầu tiên để lấy thông tin kích thước
    frame = cv2.imread(frame_paths[0])
    if frame is None:
        print(f"Error: Cannot read first frame from {frame_paths[0]}")
        return
    height, width = frame.shape[:2]

    # Khởi tạo video output
    fps = 30  # Giả định FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    folder_name = os.path.basename(os.path.dirname(video_dir))
    video_name = os.path.basename(video_dir)
    out = cv2.VideoWriter(os.path.join(output_dir, f"{folder_name}_{video_name}_tracked.mp4"), fourcc, fps, (width, height))

    # Khởi tạo tracker với face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        c_x = x + w/2; c_y = y + h/2
        rho = max(w, h) / min(width, height)
        phi = 0.0
        print(f"Initial face at {x},{y},{w},{h} -> state: {[c_x, c_y, rho, phi]}")
    else:
        c_x, c_y = width/2, height/2
        rho, phi = 0.2, 0.0
        print("No face detected; using center as initial state.")

    face_tracker.initialize(np.array([c_x, c_y, rho, phi]))

    tracking_states = []
    result = face_tracker.track_face(frame)
    state = result['tracking_state']
    tracking_states.append([0, state['c_x'], state['c_y'], state['rho'], state['phi']])

    # Xử lý từng frame
    frame_idx = 1
    for frame_path in frame_paths[1:]:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error reading frame {frame_path}")
            continue

        result = face_tracker.track_face(frame)
        state = result['tracking_state']
        tracking_states.append([frame_idx, state['c_x'], state['c_y'], state['rho'], state['phi']])
        cx, cy = int(state['c_x']), int(state['c_y'])
        rho, phi = state['rho'], state['phi']

        if visualize:
            box_size = int(min(width, height) * rho)
            half = box_size // 2
            pts = np.array([[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]], np.int32)
            rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(phi), 1.0)
            pts = pts.reshape(-1,1,2)
            pts = cv2.transform(pts, rot)
            cv2.polylines(frame, [pts], True, (0,255,0), 2)
            cv2.putText(frame, f"E={result['energy']:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        cf = result['cropped_face']
        if cf is not None and cf.size > 0:
            h0, w0 = cf.shape[:2]
            max_width = frame.shape[1] - 10  # Giới hạn chiều rộng
            w_fit = min(w0, max_width)
            h_fit = min(h0, frame.shape[0] - 10)  # Giới hạn chiều cao
            start_x = frame.shape[1] - 10 - w_fit
            start_y = 10
            # Đảm bảo không vượt quá biên
            if start_x < 0:
                w_fit += start_x  # Điều chỉnh w_fit
                start_x = 0
            frame[start_y:start_y+h_fit, start_x:start_x+w_fit] = cf[:h_fit, :w_fit]


        out.write(frame)
        frame_idx += 1

    # Lưu tracking states
    df = pd.DataFrame(tracking_states, columns=['frame_idx', 'c_x', 'c_y', 'rho', 'phi'])
    df.to_csv(os.path.join(output_dir, f"{folder_name}_{video_name}_tracking.csv"), index=False)

    out.release()
    print(f"Finished {folder_name}/{video_name}, frames: {frame_idx}")

if __name__ == '__main__':
    main()