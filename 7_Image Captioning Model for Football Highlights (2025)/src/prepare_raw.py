import os
import json
import cv2
from pathlib import Path


def ensure_dirs():
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)


def parse_position(pos_str):
    """Parse position field (milliseconds as string) to int seconds."""
    try:
        return int(pos_str) // 1000
    except Exception:
        # Fallback if already seconds
        try:
            return int(pos_str)
        except Exception:
            return 0


def extract_frames_from_video(video_path, timestamps_seconds, output_prefix, max_frames=1):
    """Extract frames at or near timestamps (seconds) from video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames_saved = []

    for i, ts in enumerate(timestamps_seconds[:max_frames]):
        frame_no = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            # try a few nearby frames
            found = False
            for offset in (1, -1, 2, -2, 5, -5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + offset)
                ret, frame = cap.read()
                if ret:
                    found = True
                    break
            if not found:
                continue

        out_path = f"data/images/{output_prefix}_{i+1}.jpg"
        cv2.imwrite(out_path, frame)
        frames_saved.append(out_path)

    cap.release()
    return frames_saved


def main():
    ensure_dirs()

    raw_root = Path('data/raw')
    captions_out = Path('data/captions.txt')

    all_captions = []

    # Iterate through raw event folders
    for match_dir in raw_root.iterdir():
        if not match_dir.is_dir():
            continue

        labels_file = match_dir / 'Labels-caption.json'
        if not labels_file.exists():
            continue

        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Use annotations descriptions as captions and position for frame extraction
            annotations = data.get('annotations', [])

            # Available video files
            videos = [p for p in match_dir.iterdir() if p.suffix.lower() in ('.mkv', '.mp4', '.avi')]
            video_path = str(videos[0]) if videos else None

            for idx, ann in enumerate(annotations):
                desc = ann.get('anonymized') # ann.get('description') or ann.get('identified') or ann.get('anonymized')
                pos = ann.get('position') or ann.get('position_ms') or '0'
                ts_sec = parse_position(pos)

                if not desc:
                    continue

                # Build a simple id
                image_id = f"{match_dir.name.replace(' ', '_')}_{idx+1}"

                # If we have video, extract a frame
                if video_path:
                    frames = extract_frames_from_video(video_path, [ts_sec], image_id, max_frames=1)
                    if frames:
                        # rename first frame to image_id.jpg
                        target = Path('data/images') / f"{image_id}.jpg"
                        Path(frames[0]).replace(target)
                    else:
                        # skip if no frame
                        continue

                # Save caption
                clean_desc = ' '.join(desc.split())
                all_captions.append(f"{image_id}\t{clean_desc}")

        except Exception as e:
            print(f"Error processing {labels_file}: {e}")
            continue

    # Write captions file
    if all_captions:
        with open(captions_out, 'w', encoding='utf-8') as f:
            for line in all_captions:
                f.write(line + '\n')

        print(f"Wrote {len(all_captions)} captions to {captions_out}")
    else:
        print("No captions extracted")


if __name__ == '__main__':
    main()
