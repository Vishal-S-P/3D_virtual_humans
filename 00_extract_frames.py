import os
import subprocess




def extract_frames(
    video_path,
    out_dir="frames",
    fps=2,
    image_ext="png"
):
  
    os.makedirs(out_dir, exist_ok=True)

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, f"frame_%05d.{image_ext}")
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Frames saved to: {out_dir}")



if __name__ == "__main__":
    extract_frames(
        video_path="./videos/IMG_8613.MOV",
        out_dir="./video_frames/8613",
        fps=8,
        image_ext="png"
    )