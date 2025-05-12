import cv2
import pickle
from hmm_lda_recognizer import HMMLDARecognizer
import os

def main():
    # Load model
    with open('models/hmm_lda_model.pkl', 'rb') as f:
        recognizer = pickle.load(f)

    # Video demo
    video_path = "path/to/test_video.mp4"  # Thay bằng video thực tế
    output_dir = "output_demo"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, "demo_tracked.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    while frame_idx < 100:  # Giới hạn 100 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Nhận diện
        person_id, confidence = recognizer.recognize(video_path)
        print(f"Frame {frame_idx}: Person={person_id}, Confidence={confidence:.2f}")

        # Vẽ kết quả
        cv2.putText(frame, f"ID: {person_id} ({confidence:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Demo completed. Output saved to output_demo/demo_tracked.mp4")

if __name__ == "__main__":
    main()