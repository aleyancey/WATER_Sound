"""Downloads single YouTube video."""
import pytube
import os


def download_video_ytdlp(video_id, save_dir, conf_file, log_dir):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    log_file = f"{log_dir}/{video_id}.log"

    save_path = f"{save_dir}/{video_id}.mkv"
    if os.path.exists(save_path):
        return {"video_id": video_id, "status": "ALREADY_DOWNLOADED", "download_time": 0}

    # Also check mp4
    mp4_path = f"{save_dir}/{video_id}.mp4"
    if os.path.exists(mp4_path):
        return {"video_id": video_id, "status": "ALREADY_DOWNLOADED", "download_time": 0}

    start = time.time()
    try:
        # command = f"youtube-dl -f bestvideo+bestaudio --merge-output-format mp4 -o {save_dir}/%(id)s.%(ext)s {video_url}"
        command = f"yt-dlp --config-location {conf_file} -o {save_dir}/{video_id} {video_url} > {log_file} 2>&1"
        # command = f"yt-dlp --config-location {conf_file} -o {save_dir}/{video_id} {video_url}"
        call(command, shell=True)
        status = "SUCCESS"
    except:
        status = "FAIL"
        return {"video_id": video_id, "status": status, "download_time": 0}
    end = time.time()
    download_time = end - start

    return {"video_id": video_id, "status": status, "download_time": download_time}


if __name__ == "__main__":
    import os
    import time
    conf_file = "./youtube-dlp.conf"
    log_dir = "../download_metadata/"
    os.makedirs(log_dir, exist_ok=True)
    save_dir = "../videos/"
    os.makedirs(save_dir, exist_ok=True)
    video_id = "K87g4RvO-9k"
    download_video_ytdlp(video_id, save_dir, conf_file, log_dir)
