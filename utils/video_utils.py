import cv2

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the original FPS from the video
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return reduce_fps(frames, 24, original_fps)


def save_video_from_frames(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()


def reduce_fps(frames, target_fps, original_fps):
    if target_fps >= original_fps:
        return frames
    
    # Calculate the frame interval
    interval = original_fps / target_fps
    # Select frames at the calculated interval
    reduced_frames = [frames[int(i * interval)] for i in range(int(len(frames) / interval))]
    
    return reduced_frames