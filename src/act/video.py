"""Video writing helper."""
import cv2


def open_video_writer(path, fps, width, height):
    """H.264 (plays in QuickTime) if this OpenCV build has it, else MPEG-4 Part 2."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")
