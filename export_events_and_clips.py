# fd_window_clipper.py
# Window-by-window fight detection (MC3-18) → RAW LABEL segments (no hysteresis) → clips + CSV/JSON
# Run:  python fd_window_clipper.py

import os, cv2, csv, json, time, shutil, subprocess
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# ========== PATHS (YOUR SETUP) ==========
WORK_DIR   = r"D:\PycharmProject\JayantS_Fight_Dectection"
VIDEO_PATH = r"D:\PycharmProject\JayantS_Fight_Dectection\3.mp4"             # change if needed
WEIGHTS    = r"D:\PycharmProject\JayantS_Fight_Dectection\model_16_m3_0.8888.pth"
OUT_ROOT   = r"D:\PycharmProject\JayantS_Fight_Dectection\output_fd"
# ========================================

# -------- SETTINGS (tune as needed) -----
SEQ  = 16      # frames per window (your checkpoint is 16)
SKIP = 1       # take every SKIP-th frame (1 = every frame)
# RAW label pipeline options:
SAVE_FIGHT_CLIPS    = True     # clip segments labeled "fight"
SAVE_NOFIGHT_CLIPS  = False     # also clip "noFight" segments
MERGE_GAP_S         = 0.0      # merge same-label segments if gap between them <= this
MIN_SEG_DUR_S       = 0.0      # drop tiny segments shorter than this (0 = keep all)
PRE_PAD_S           = 0.0      # padding before each segment in seconds
POST_PAD_S          = 0.0      # padding after each segment in seconds
MAKE_HIGHLIGHT      = True     # concatenate all fight clips into a single highlight video
HIGHLIGHT_NAME      = "fight_highlight.mp4"
# ----------------------------------------

# ----- Albumentations transform (same as repo) -----
import albumentations as A
def transform_():
    return A.Compose([
        A.Resize(128, 171, always_apply=True),
        A.CenterCrop(112, 112, always_apply=True),
        A.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989], always_apply=True)
    ])
# ---------------------------------------------------

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def sec_to_hms(s):
    s = max(0.0, float(s))
    h = int(s // 3600); s -= 3600*h
    m = int(s // 60);   s -= 60*m
    return f"{h:02d}h{m:02d}m{s:05.2f}s"

def ffmpeg_available(): return shutil.which("ffmpeg") is not None

def concat_with_ffmpeg(file_list, dst_path, fps_hint=None):
    """
    Concatenate mp4s using ffmpeg concat demuxer. Falls back to moviepy if needed.
    Writes a list file where each line is:  file 'absolute/path.mp4'
    """
    if not file_list:
        return
    dst = str(dst_path)
    list_file = Path(dst_path).with_suffix(".txt")

    # Build the concat list without backslashes in f-string expressions
    with open(list_file, "w", encoding="utf-8", newline="\n") as f:
        for p in file_list:
            # Use absolute, forward-slash style to be friendlier to ffmpeg on Windows
            p_abs = Path(p).resolve()
            p_str = p_abs.as_posix()
            # Escape single quotes per ffmpeg concat demuxer rules: ' -> '\''
            p_str = p_str.replace("'", "'\\''")
            f.write(f"file '{p_str}'\n")

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", dst
        ]
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception:
            pass

    # Fallback: re-encode via moviepy if ffmpeg copy concat fails/unavailable
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        clips = [VideoFileClip(str(p)) for p in file_list]
        out = concatenate_videoclips(clips, method="compose")
        out.write_videofile(dst, audio=False, fps=fps_hint or 25, logger=None)
        for c in clips: c.close()
    except Exception as e:
        print("[Concat] failed:", e)


# ----- Model loader (no dependency on their package) -----
def load_model_mc3_18(weights_path: str, device: torch.device):
    """
    Loads torchvision.models.video.mc3_18 and swaps final FC to 2 classes.
    Expects CLASSES_LIST = ['fight','noFight'] ordering in the checkpoint.
    """
    try:
        from torchvision.models.video import mc3_18
        model = mc3_18(weights=None)
    except Exception:
        import torchvision
        model = torchvision.models.video.mc3_18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_window_label(model, frames, device):
    """
    frames: list of SEQ RGB-normalized frames HxWxC after transform_()
    returns: (label_str, fight_prob)
    """
    with torch.no_grad():
        x = np.array(frames)                            # [T, H, W, C]
        x = np.expand_dims(x, axis=0)                  # [1, T, H, W, C]
        x = np.transpose(x, (0, 4, 1, 2, 3))           # [1, 3, T, H, W]
        x = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(x)                               # [1,2]
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        # CLASSES_LIST = ['fight','noFight'] → index 0 is fight
        fight_prob = float(probs[0])
        label = "fight" if probs[0] >= probs[1] else "noFight"
        return label, fight_prob

def save_clip(src, dst, t0, t1):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if t1 <= t0:
        return
    if ffmpeg_available():
        # try fast copy
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
               "-ss", f"{t0:.3f}","-to", f"{t1:.3f}","-i", str(src), "-c","copy", str(dst)]
        try:
            subprocess.run(cmd, check=True); return
        except Exception:
            pass
        # fallback re-encode tiny segment
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
               "-ss", f"{t0:.3f}","-to", f"{t1:.3f}","-i", str(src),
               "-c:v","libx264","-crf","23","-preset","veryfast","-c:a","aac","-b:a","128k", str(dst)]
        try:
            subprocess.run(cmd, check=True); return
        except Exception:
            pass
    # last fallback: moviepy
    try:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(str(src)) as clip:
            sub = clip.subclip(max(0,t0), t1)
            sub.write_videofile(str(dst), audio=False, logger=None)
    except Exception as e:
        print("[Clip] failed:", e)

def main():
    os.chdir(WORK_DIR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    # Load model
    assert Path(WEIGHTS).exists(), f"Weights not found: {WEIGHTS}"
    print("[Info] loading model weights…")
    model = load_model_mc3_18(WEIGHTS, device)

    # Video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else -1
    dur_s = nframes / fps if nframes > 0 else None

    out_dir = Path(OUT_ROOT) / Path(VIDEO_PATH).stem
    ensure_dir(out_dir)
    clips_dir = out_dir / "clips"
    ensure_dir(clips_dir)
    ensure_dir(clips_dir / "fight")
    ensure_dir(clips_dir / "noFight")

    tr = transform_()
    frames_q = deque(maxlen=SEQ)
    timeline = []  # list of {start_s,end_s,label,fight_prob}

    print(f"[Info] scanning RAW windows… SEQ={SEQ}, SKIP={SKIP}")
    frame_idx = 0
    window_first_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = tr(image=rgb)['image']
            frames_q.append(proc)
            if len(frames_q) == 1:
                window_first_frame = frame_idx

            if len(frames_q) == SEQ:
                label, p_fight = predict_window_label(model, list(frames_q), device)
                start_s = (window_first_frame) / fps
                end_s   = (frame_idx) / fps
                timeline.append({"start_s": start_s, "end_s": end_s,
                                 "label": label, "fight_prob": round(p_fight, 5)})
                # non-overlapping windows
                frames_q = deque(maxlen=SEQ)
                window_first_frame = None

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  {frame_idx} frames…")

    cap.release()
    print(f"[Info] windows: {len(timeline)}")

    # Build segments from RAW labels (merge adjacent same-label windows)
    def build_segments(tl, label_keep):
        segs = []
        cur = None
        max_conf = 0.0
        avg_sum = 0.0
        n = 0
        for w in tl:
            st, et, lab, pf = w["start_s"], w["end_s"], w["label"], w["fight_prob"]
            if lab == label_keep:
                if cur is None:
                    cur = [st, et]
                    max_conf = pf
                    avg_sum = pf
                    n = 1
                else:
                    # contiguous window of same label → extend
                    cur[1] = et
                    max_conf = max(max_conf, pf)
                    avg_sum += pf; n += 1
            else:
                if cur is not None:
                    segs.append([cur[0], cur[1], round(max_conf,4), round(avg_sum/max(n,1),4)])
                    cur = None
        if cur is not None:
            segs.append([cur[0], cur[1], round(max_conf,4), round(avg_sum/max(n,1),4)])
        return segs

    def merge_close(segs, gap_s):
        if not segs or gap_s <= 0: return segs
        merged = [segs[0][:]]
        for st, et, mx, av in segs[1:]:
            gap = st - merged[-1][1]
            if gap <= gap_s:
                merged[-1][1] = et
                merged[-1][2] = max(merged[-1][2], mx)
                merged[-1][3] = round((merged[-1][3] + av) / 2.0, 4)
            else:
                merged.append([st, et, mx, av])
        return merged

    def drop_short(segs, min_dur):
        if min_dur <= 0: return segs
        return [s for s in segs if (s[1]-s[0]) >= min_dur]

    # Fight segments
    fight_segments = build_segments(timeline, "fight")
    fight_segments = merge_close(fight_segments, MERGE_GAP_S)
    fight_segments = drop_short(fight_segments, MIN_SEG_DUR_S)
    print(f"[Info] fight segments: {len(fight_segments)}")

    # Optional: NoFight segments
    nofight_segments = []
    if SAVE_NOFIGHT_CLIPS:
        nofight_segments = build_segments(timeline, "noFight")
        nofight_segments = merge_close(nofight_segments, MERGE_GAP_S)
        nofight_segments = drop_short(nofight_segments, MIN_SEG_DUR_S)
        print(f"[Info] noFight segments: {len(nofight_segments)}")

    # Save CSVs (events)
    if fight_segments:
        ev_csv = out_dir / "events_fight.csv"
        with open(ev_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_id","label","start_time_s","end_time_s","duration_s",
                        "max_conf_fight","avg_conf_fight","start_hms","end_hms"])
            for i, (st, et, mx, av) in enumerate(fight_segments, 1):
                w.writerow([i, "fight", round(st,3), round(et,3), round(et-st,3),
                            mx, av, sec_to_hms(st), sec_to_hms(et)])
        print("[Save]", ev_csv)

    if SAVE_NOFIGHT_CLIPS and nofight_segments:
        ev_csv = out_dir / "events_noFight.csv"
        with open(ev_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_id","label","start_time_s","end_time_s","duration_s",
                        "max_conf_fight","avg_conf_fight","start_hms","end_hms"])
            for i, (st, et, mx, av) in enumerate(nofight_segments, 1):
                w.writerow([i, "noFight", round(st,3), round(et,3), round(et-st,3),
                            mx, av, sec_to_hms(st), sec_to_hms(et)])
        print("[Save]", ev_csv)

    # ---------- Save clips & build clip metadata (for enriching timeline) ----------
    fight_clip_paths = []
    fight_clip_meta = []
    nofight_clip_meta = []

    if SAVE_FIGHT_CLIPS:
        for i, (st, et, mx, av) in enumerate(fight_segments, 1):
            t0c = max(0.0, st - PRE_PAD_S)
            t1c = et + POST_PAD_S
            name = f"{Path(VIDEO_PATH).stem}_F{i:03d}_fight_[{sec_to_hms(t0c)}-{sec_to_hms(t1c)}]_conf{mx:.2f}.mp4"
            dst  = clips_dir / "fight" / name
            # meta for enrichment
            fight_clip_meta.append({
                "id": i, "label": "fight",
                "seg_start_s": st, "seg_end_s": et,
                "clip_start_s": t0c, "clip_end_s": t1c,
                "clip_path": str(dst)
            })
            save_clip(VIDEO_PATH, dst, t0c, t1c)
            fight_clip_paths.append(dst)
            print("  [Clip fight]", dst)

    if SAVE_NOFIGHT_CLIPS:
        for i, (st, et, mx, av) in enumerate(nofight_segments, 1):
            t0c = max(0.0, st - PRE_PAD_S)
            t1c = et + POST_PAD_S
            name = f"{Path(VIDEO_PATH).stem}_N{i:03d}_noFight_[{sec_to_hms(t0c)}-{sec_to_hms(t1c)}]_pf{mx:.2f}.mp4"
            dst  = clips_dir / "noFight" / name
            nofight_clip_meta.append({
                "id": i, "label": "noFight",
                "seg_start_s": st, "seg_end_s": et,
                "clip_start_s": t0c, "clip_end_s": t1c,
                "clip_path": str(dst)
            })
            save_clip(VIDEO_PATH, dst, t0c, t1c)
            print("  [Clip noFight]", dst)

    # Optional highlight concat (fight only) — unchanged
    if MAKE_HIGHLIGHT and fight_clip_paths:
        dst = out_dir / HIGHLIGHT_NAME
        print("[Info] making highlight:", dst)
        concat_with_ffmpeg(fight_clip_paths, dst, fps_hint=int(round(fps)))

    # ---------- Enrich timeline with segment/clip info & save ----------
    def annotate_windows_with_clips(tl, fight_meta, nofight_meta):
        fight_meta = sorted(fight_meta, key=lambda m: m["seg_start_s"])
        nofight_meta = sorted(nofight_meta, key=lambda m: m["seg_start_s"])

        def lookup(meta_list, st, et):
            # window fully inside the segment
            for m in meta_list:
                if m["seg_start_s"] <= st and et <= m["seg_end_s"]:
                    return m
            return None

        enriched = []
        for w in tl:
            row = dict(w)
            row.update({
                "event_id": "",
                "event_label": "",
                "seg_start_s": "",
                "seg_end_s": "",
                "seg_duration_s": "",
                "clip_start_s": "",
                "clip_end_s": "",
                "clip_duration_s": "",
                "clip_path": ""
            })
            st, et = w["start_s"], w["end_s"]
            m = lookup(fight_meta if w["label"] == "fight" else nofight_meta, st, et)
            if m:
                row["event_id"] = m["id"]
                row["event_label"] = m["label"]
                row["seg_start_s"] = round(m["seg_start_s"], 3)
                row["seg_end_s"]   = round(m["seg_end_s"], 3)
                row["seg_duration_s"] = round(m["seg_end_s"] - m["seg_start_s"], 3)
                row["clip_start_s"] = round(m["clip_start_s"], 3)
                row["clip_end_s"]   = round(m["clip_end_s"], 3)
                row["clip_duration_s"] = round(m["clip_end_s"] - m["clip_start_s"], 3)
                row["clip_path"] = m["clip_path"]
            enriched.append(row)
        return enriched

    timeline_enriched = annotate_windows_with_clips(timeline, fight_clip_meta, nofight_clip_meta)
    tl_csv = out_dir / "timeline_windows.csv"
    with open(tl_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "win_id","start_time_s","end_time_s","duration_s","label","fight_prob",
            "event_id","event_label",
            "seg_start_s","seg_end_s","seg_duration_s",
            "clip_start_s","clip_end_s","clip_duration_s","clip_path"
        ])
        for i, t in enumerate(timeline_enriched, 1):
            w.writerow([
                i,
                round(t["start_s"],3), round(t["end_s"],3),
                round(t["end_s"]-t["start_s"],3),
                t["label"], t["fight_prob"],
                t["event_id"], t["event_label"],
                t["seg_start_s"], t["seg_end_s"], t["seg_duration_s"],
                t["clip_start_s"], t["clip_end_s"], t["clip_duration_s"],
                t["clip_path"]
            ])
    print("[Save]", tl_csv)

    # summary.json — unchanged except mode details
    summary = {
        "video": str(VIDEO_PATH),
        "weights": str(WEIGHTS),
        "fps": fps,
        "frames": nframes,
        "duration_s": dur_s,
        "seq": SEQ, "skip": SKIP,
        "mode": "raw_label_segments",
        "merge_gap_s": MERGE_GAP_S,
        "min_seg_dur_s": MIN_SEG_DUR_S,
        "clip_padding": {"pre": PRE_PAD_S, "post": POST_PAD_S},
        "save_fight": SAVE_FIGHT_CLIPS,
        "save_noFight": SAVE_NOFIGHT_CLIPS
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[Save]", out_dir / "summary.json")

    print("\n[Done] Folder:", out_dir.resolve())

if __name__ == "__main__":
    main()
