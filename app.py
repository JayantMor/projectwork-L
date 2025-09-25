# # Fallback Script
#
# import os, io, sys, uuid, zipfile, json, subprocess
# from pathlib import Path
# from datetime import datetime
#
# import streamlit as st
# import pandas as pd
# import plotly.express as px
#
# # ----------------------------
# # Paths (edit if your files live elsewhere)
# # ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# OUTPUTS_DIR = BASE_DIR / "outputs"
# OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#
# YOLO_SCRIPT   = BASE_DIR / "Yolov8_Violence_Detection.py"
# EXPORT_SCRIPT = BASE_DIR / "export_events_and_clips.py"
#
# DEFAULT_YOLO_WEIGHTS  = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\yolo_violence_detection_model.pt")
# DEFAULT_FIGHT_WEIGHTS = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\fight_detection.pth")
#
# # ----------------------------
# # Session state
# # ----------------------------
# if "run_id" not in st.session_state:
#     st.session_state.run_id = None
# if "uploaded_video_path" not in st.session_state:
#     st.session_state.uploaded_video_path = None
# if "result_base" not in st.session_state:
#     st.session_state.result_base = None  # outputs/<run_id>/<video_stem>/
# if "yolo_overlay_path" not in st.session_state:
#     st.session_state.yolo_overlay_path = None
# if "step" not in st.session_state:
#     st.session_state.step = 1
#
# def new_run():
#     st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
#     (OUTPUTS_DIR / st.session_state.run_id).mkdir(parents=True, exist_ok=True)
#
# def current_run_dir() -> Path:
#     if not st.session_state.run_id:
#         new_run()
#     return OUTPUTS_DIR / st.session_state.run_id
#
# # ----------------------------
# # Helpers
# # ----------------------------
# def save_uploaded_video(file) -> Path:
#     run_dir = current_run_dir()
#     dst = run_dir / "input.mp4"
#     with open(dst, "wb") as f:
#         f.write(file.read())
#     return dst
#
# def transcode_web_mp4(src: Path, dst: Path, width: int | None = 720):
#     """
#     Makes a browser-friendly MP4 (H.264 yuv420p). If ffmpeg not present, falls back to moviepy.
#     width=None keeps original size; width=720 gives a neat, fixed preview.
#     """
#     import shutil, subprocess
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if shutil.which("ffmpeg"):
#         cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                "-c:a","aac","-b:a","128k", str(dst)]
#         if width:
#             cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                    "-vf", f"scale={width}:-2",
#                    "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                    "-c:a","aac","-b:a","128k", str(dst)]
#         subprocess.run(cmd, check=True)
#     else:
#         from moviepy.editor import VideoFileClip
#         with VideoFileClip(str(src)) as clip:
#             if width:
#                 clip = clip.resize(width=width)
#             clip.write_videofile(str(dst), codec="libx264", audio_codec="aac", preset="veryfast", logger=None)
#
# def load_json_safe(path: Path) -> dict | None:
#     try:
#         if path.exists():
#             return json.loads(path.read_text())
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def load_csv_safe(path: Path) -> pd.DataFrame | None:
#     try:
#         if path.exists():
#             return pd.read_csv(path)
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def zip_folder(folder: Path) -> bytes:
#     mem = io.BytesIO()
#     with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#         for p in folder.rglob("*"):
#             if p.is_file():
#                 zf.write(p, p.relative_to(folder))
#     mem.seek(0)
#     return mem.read()
#
# # ----------------------------
# # PIPELINES (Option A: subprocess)
# # ----------------------------
# def run_yolo_overlay_cli(video_path: Path, weights_path: Path, out_path: Path, conf: float = 0.40, codec: str = "mp4v"):
#     """
#     Calls your YOLO overlay script with CLI flags to produce an overlay MP4.
#     """
#     cmd = [
#         sys.executable, str(YOLO_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out", str(out_path),
#         "--conf", f"{conf:.2f}",
#         "--codec", codec,
#         "--device", "auto",
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"YOLO overlay failed (exit {e.returncode}).")
#         raise
#     return out_path
#
# def run_fight_export_cli(video_path: Path, weights_path: Path, out_root: Path):
#     """
#     Calls export_events_and_clips.py with your preferred defaults:
#     SEQ=16, SKIP=1, merge_gap=0, min_seg=0, pre/post pad=0,
#     SAVE_FIGHT_CLIPS=True, SAVE_NOFIGHT_CLIPS=False, MAKE_HIGHLIGHT=True
#     """
#     cmd = [
#         sys.executable, str(EXPORT_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out_root", str(out_root),
#         "--seq", "16",
#         "--skip", "1",
#         "--merge_gap_s", "0.0",
#         "--min_seg_dur_s", "0.0",
#         "--pre_pad_s", "0.0",
#         "--post_pad_s", "0.0",
#         "--highlight_name", "fight_highlight.mp4",
#         # important: we DO NOT pass flags that would change your optimal defaults
#         # (no --save_noFight, no --no_highlight, no --no_save_fight)
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"Fight export failed (exit {e.returncode}).")
#         raise
#
#     video_stem = Path(video_path).stem
#     result_base = out_root / video_stem
#     return result_base
#
# # ----------------------------
# # UI
# # ----------------------------
# st.set_page_config(page_title="Fight Detection Dashboard", layout="wide")
# st.title("Milestone 3 — Fight Detection Dashboard")
#
# with st.sidebar:
#     st.header("Controls")
#
#     # YOLO overlay knobs
#     yolo_weights = Path(st.text_input("YOLO weights (.pt)", value=str(DEFAULT_YOLO_WEIGHTS)))
#     yolo_conf = st.slider("YOLO confidence", 0.0, 1.0, 0.40, 0.01)
#     yolo_codec = st.selectbox("YOLO output codec", ["mp4v", "XVID"], index=0, help="Use XVID if MP4 has playback issues")
#
#     st.markdown("---")
#
#     # Fight detector model (no thresholds exposed; using your optimal defaults)
#     fight_weights = Path(st.text_input("Fight model weights (.pth)", value=str(DEFAULT_FIGHT_WEIGHTS)))
#
#     st.markdown("---")
#     if st.button("Start new analysis (reset)", use_container_width=True):
#         st.session_state.step = 1
#         st.session_state.uploaded_video_path = None
#         st.session_state.result_base = None
#         st.session_state.yolo_overlay_path = None
#         new_run()
#         st.rerun()
#
# # ----- REAL TABS (now functional) -----
# tab1, tab2, tab3, tab4 = st.tabs(["① Overview", "② Detection", "③ Analytics", "④ Clips & Labels"])
#
# # ===========================
# # TAB ① — OVERVIEW (Upload)
# # ===========================
# with tab1:
#     # Step 1 — Upload video
#     st.subheader("1) Upload video")
#     up = st.file_uploader("Upload an MP4", type=["mp4"])
#
#     c1, c2 = st.columns([2,1])
#     with c1:
#         if up:
#             new_run()
#             st.session_state.uploaded_video_path = save_uploaded_video(up)
#             st.session_state.step = max(st.session_state.step, 2)
#             st.success(f"Uploaded to: {st.session_state.uploaded_video_path}")
#     with c2:
#         if st.session_state.uploaded_video_path:
#             st.video(str(st.session_state.uploaded_video_path))
#             st.caption(f"Run ID: {st.session_state.run_id}")
#
# # ==========================================
# # TAB ② — DETECTION (Overlay + Fight export)
# # ==========================================
# with tab2:
#     # Step 2 — YOLO overlay (overview)
#     if st.session_state.step >= 2 and st.session_state.uploaded_video_path:
#         st.subheader("2) YOLO overlay (overview)")
#         c3, c4 = st.columns([1,2])
#         with c3:
#             if st.button("Run YOLO overlay", type="primary"):
#                 with st.spinner("Running YOLO overlay…"):
#                     run_dir = current_run_dir()
#                     raw_path = run_dir / "yolo_overlay_raw.mp4"  # written by your YOLO script
#                     web_path = run_dir / "yolo_overlay_web.mp4"  # web-safe preview we’ll show
#                     try:
#                         overlay_raw = run_yolo_overlay_cli(
#                             video_path=Path(st.session_state.uploaded_video_path),
#                             weights_path=yolo_weights,
#                             out_path=raw_path,
#                             conf=yolo_conf,
#                             codec=yolo_codec
#                         )
#                         # Transcode to web-safe (fixed preview width)
#                         transcode_web_mp4(overlay_raw, web_path, width=720)
#                         st.session_state.yolo_overlay_path = web_path
#                         st.session_state.step = max(st.session_state.step, 3)
#                         st.success(f"Overlay ready: {web_path.name}")
#                     except Exception:
#                         st.stop()
#         with c4:
#             p = st.session_state.yolo_overlay_path
#             if p and Path(p).exists() and Path(p).stat().st_size > 0:
#                 st.video(str(p))
#                 # Optional: also give the original file as a download
#                 raw = (current_run_dir() / "yolo_overlay_raw.mp4")
#                 if raw.exists():
#                     st.download_button("Download full overlay (raw)", data=raw.read_bytes(),
#                                        file_name="yolo_overlay.mp4", mime="video/mp4")
#             else:
#                 st.info("Run the overlay to preview detections.")
#     else:
#         st.info("Upload a video in ① Overview.")
#
#     # Step 3 — Fight detector (clips + analytics)
#     if st.session_state.step >= 2 and st.session_state.uploaded_video_path:
#         st.subheader("3) Detect fights, export clips + analytics")
#         if st.button("Run fight detector", type="primary"):
#             with st.spinner("Building clips & analytics…"):
#                 out_root = current_run_dir()
#                 try:
#                     result_base = run_fight_export_cli(
#                         video_path=Path(st.session_state.uploaded_video_path),
#                         weights_path=fight_weights,
#                         out_root=out_root
#                     )
#                     st.session_state.result_base = result_base
#                     st.session_state.step = max(st.session_state.step, 4)
#                     st.success(f"Results ready in: {result_base}")
#                 except Exception:
#                     st.stop()
#
# # ======================================
# # TAB ③ — ANALYTICS (Highlights/Charts)
# # ======================================
# with tab3:
#     # Step 4 — Results (analytics views)
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         summary_json     = base / "summary.json"
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         highlight_mp4    = base / "fight_highlight.mp4"
#
#         st.subheader("4) Highlights")
#         if highlight_mp4.exists():
#             st.video(str(highlight_mp4))
#         else:
#             st.info("No highlight video found.")
#
#         # Metrics
#         st.markdown("### Overview")
#         summary = load_json_safe(summary_json) or {}
#         events_df = load_csv_safe(events_fight_csv)
#         c5, c6, c7, c8, c9, c10 = st.columns(6)
#
#         num_events = int(events_df["event_id"].count()) if isinstance(events_df, pd.DataFrame) else 0
#         total_fight_time = float(events_df["duration_s"].sum()) if isinstance(events_df, pd.DataFrame) else 0.0
#         mean_conf = float(events_df["avg_conf_fight"].mean()) if isinstance(events_df, pd.DataFrame) else 0.0
#         longest_event = float(events_df["duration_s"].max()) if isinstance(events_df, pd.DataFrame) else 0.0
#
#         c5.metric("Events", num_events)
#         c6.metric("Total fight time (s)", round(total_fight_time, 2))
#         c7.metric("Mean conf", round(mean_conf, 3))
#         c8.metric("Longest event (s)", round(longest_event, 2))
#         c9.metric("FPS", round(float(summary.get("fps", 0)), 2))
#         c10.metric("Duration (s)", round(float(summary.get("duration_s", 0)), 2))
#
#         # Timeline — events
#         st.markdown("### Timeline — Events")
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             df = events_df.copy()
#             df["start_dt"] = pd.to_datetime(df["start_time_s"], unit="s", origin="unix")
#             df["end_dt"] = pd.to_datetime(df["end_time_s"], unit="s", origin="unix")
#
#             fig = px.timeline(
#                 df, x_start="start_dt", x_end="end_dt", y="event_id", color="avg_conf_fight",
#                 hover_data={
#                     "event_id": True,
#                     "start_time_s": ":.3f",
#                     "end_time_s": ":.3f",
#                     "duration_s": ":.3f",
#                     "avg_conf_fight": ":.3f",
#                     "max_conf_fight": ":.3f",
#                     "start_dt": False, "end_dt": False,
#                 },
#                 color_continuous_scale="Turbo",
#             )
#             fig.update_yaxes(autorange="reversed", title="Event")
#             fig.update_xaxes(title="Time (mm:ss)", tickformat="%M:%S")  # key line
#             fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("events_fight.csv not found or empty.")
#
#         # Timeline — window strip
#         st.markdown("### Timeline — Window-level confidence")
#         tw = load_csv_safe(timeline_csv)
#         if isinstance(tw, pd.DataFrame) and not tw.empty and "fight_prob" in tw.columns:
#             tw2 = tw.copy()
#             tw2["t_center_s"] = (tw2["start_time_s"] + tw2["end_time_s"]) / 2.0
#             fig2 = px.scatter(
#                 tw2, x="t_center_s", y="fight_prob", color="label",
#                 hover_data={"start_time_s": ":.3f", "end_time_s": ":.3f", "label": True, "fight_prob": ":.3f"},
#             )
#             fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
#                                xaxis_title="Time (s)", yaxis_title="Fight prob.")
#             st.plotly_chart(fig2, use_container_width=True)
#         else:
#             st.info("timeline_windows.csv not found or missing 'fight_prob'.")
#     else:
#         st.info("Run the fight detector in ② Detection to see analytics.")
#
# # =========================================
# # TAB ④ — CLIPS & LABELS (Clips/Validate)
# # =========================================
# with tab4:
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         clips_fight_dir  = base / "clips" / "fight"
#
#         # Clips grid
#         st.subheader("Clips")
#         if clips_fight_dir.exists():
#             clips = sorted(list(clips_fight_dir.glob("*.mp4")))[:6]
#             if clips:
#                 cols = st.columns(3)
#                 for i, clip in enumerate(clips):
#                     with cols[i % 3]:
#                         st.video(str(clip))
#                         st.caption(clip.name)
#             with st.expander("Download all fight clips (.zip)"):
#                 data = zip_folder(clips_fight_dir)
#                 st.download_button("Download clips.zip", data=data, file_name="clips_fight.zip", mime="application/zip")
#         else:
#             st.info("No fight clips directory found.")
#
#         # Validation
#         st.markdown("### Validate events")
#         events_df = load_csv_safe(events_fight_csv)
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             edit_df = events_df.copy()
#             if "validated_label" not in edit_df.columns:
#                 edit_df["validated_label"] = "fight"
#             if "note" not in edit_df.columns:
#                 edit_df["note"] = ""
#             edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed", key="editor_events")
#             if st.button("Save validation"):
#                 out_csv = base / "validated_events.csv"
#                 edited.to_csv(out_csv, index=False)
#                 st.success(f"Saved: {out_csv}")
#         else:
#             st.info("Nothing to validate yet.")
#
#         # Exports
#         st.markdown("### Exports")
#         e1, e2, e3 = st.columns(3)
#         with e1:
#             if events_fight_csv.exists():
#                 st.download_button(
#                     "Download events_fight.csv",
#                     data=events_fight_csv.read_bytes(), file_name="events_fight.csv", mime="text/csv"
#                 )
#         with e2:
#             if timeline_csv.exists():
#                 st.download_button(
#                     "Download timeline_windows.csv",
#                     data=timeline_csv.read_bytes(), file_name="timeline_windows.csv", mime="text/csv"
#                 )
#         with e3:
#             val_csv = base / "validated_events.csv"
#             if val_csv.exists():
#                 st.download_button(
#                     "Download validated_events.csv",
#                     data=val_csv.read_bytes(), file_name="validated_events.csv", mime="text/csv"
#                 )
#     else:
#         st.info("Run the fight detector in ② Detection to see clips & labels.")
#


# Fallback Script (auto-pipeline)
# import os, io, sys, uuid, zipfile, json, subprocess, base64
# from pathlib import Path
# from datetime import datetime
#
# import streamlit as st
# import pandas as pd
# import plotly.express as px
#
# # ----------------------------
# # Paths (edit if your files live elsewhere)
# # ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# OUTPUTS_DIR = BASE_DIR / "outputs"
# OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#
# YOLO_SCRIPT   = BASE_DIR / "Yolov8_Violence_Detection.py"
# EXPORT_SCRIPT = BASE_DIR / "export_events_and_clips.py"
#
# DEFAULT_YOLO_WEIGHTS  = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\yolo_violence_detection_model.pt")
# DEFAULT_FIGHT_WEIGHTS = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\fight_detection.pth")
#
# # ----------------------------
# # Session state
# # ----------------------------
# if "run_id" not in st.session_state:
#     st.session_state.run_id = None
# if "uploaded_video_path" not in st.session_state:
#     st.session_state.uploaded_video_path = None
# if "result_base" not in st.session_state:
#     st.session_state.result_base = None  # outputs/<run_id>/<video_stem>/
# if "yolo_overlay_path" not in st.session_state:
#     st.session_state.yolo_overlay_path = None
# if "step" not in st.session_state:
#     st.session_state.step = 1
#
# # [AUTO] pipeline flags
# if "pipeline" not in st.session_state:
#     st.session_state.pipeline = "idle"   # idle -> overlay_pending -> fight_pending -> done
# if "processing" not in st.session_state:
#     st.session_state.processing = False
#
# def new_run():
#     if not st.session_state.run_id:
#         st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
#         (OUTPUTS_DIR / st.session_state.run_id).mkdir(parents=True, exist_ok=True)
#
#
# def current_run_dir() -> Path:
#     if not st.session_state.run_id:
#         new_run()
#     return OUTPUTS_DIR / st.session_state.run_id
#
# # ----------------------------
# # Helpers
# # ----------------------------
# def save_uploaded_video(file) -> Path:
#     run_dir = current_run_dir()
#     orig = Path(file.name).stem or "input"
#     # unique, readable stem per run (prevents any stem-based collisions)
#     unique_name = f"{orig}__{st.session_state.run_id}.mp4"
#     dst = run_dir / "input.mp4"
#     with open(dst, "wb") as f:
#         f.write(file.read())
#     return dst
#
# def transcode_web_mp4(src: Path, dst: Path, width: int | None = 720):
#     """
#     Makes a browser-friendly MP4 (H.264 yuv420p). If ffmpeg not present, falls back to moviepy.
#     width=None keeps original size; width=720 gives a neat, fixed preview.
#     """
#     import shutil, subprocess
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if shutil.which("ffmpeg"):
#         cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                "-c:a","aac","-b:a","128k", str(dst)]
#         if width:
#             cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                    "-vf", f"scale={width}:-2",
#                    "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                    "-c:a","aac","-b:a","128k", str(dst)]
#         subprocess.run(cmd, check=True)
#     else:
#         from moviepy.editor import VideoFileClip
#         with VideoFileClip(str(src)) as clip:
#             if width:
#                 clip = clip.resize(width=width)
#             clip.write_videofile(str(dst), codec="libx264", audio_codec="aac", preset="veryfast", logger=None)
#
# def load_json_safe(path: Path) -> dict | None:
#     try:
#         if path.exists():
#             return json.loads(path.read_text())
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def load_csv_safe(path: Path) -> pd.DataFrame | None:
#     try:
#         if path.exists():
#             return pd.read_csv(path)
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def zip_folder(folder: Path) -> bytes:
#     mem = io.BytesIO()
#     with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#         for p in folder.rglob("*"):
#             if p.is_file():
#                 zf.write(p, p.relative_to(folder))
#     mem.seek(0)
#     return mem.read()
#
# # [AUTO] nicer, looping/auto-playing video embed (works better than st.video for autoplay)
# def show_video_autoplay(path: Path, width_css="100%", muted=True, loop=True, autoplay=True, controls=True, height_px: int | None = None):
#     if not path or not Path(path).exists():
#         return
#     try:
#         data = Path(path).read_bytes()
#         b64 = base64.b64encode(data).decode()
#         style_h = f"height:{height_px}px;" if height_px else ""
#         html = f"""
#         <video {'controls' if controls else ''} {'autoplay' if autoplay else ''} {'loop' if loop else ''} {'muted' if muted else ''} playsinline
#                style="width:{width_css}; {style_h}">
#             <source src="data:video/mp4;base64,{b64}" type="video/mp4">
#         </video>
#         """
#         st.markdown(html, unsafe_allow_html=True)
#     except Exception as e:
#         # fallback to st.video if base64 is too big
#         st.video(str(path))
#
# # ----------------------------
# # PIPELINES (Option A: subprocess)
# # ----------------------------
# def run_yolo_overlay_cli(video_path: Path, weights_path: Path, out_path: Path, conf: float = 0.40, codec: str = "mp4v"):
#     """
#     Calls your YOLO overlay script with CLI flags to produce an overlay MP4.
#     """
#     cmd = [
#         sys.executable, str(YOLO_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out", str(out_path),
#         "--conf", f"{conf:.2f}",
#         "--codec", codec,
#         "--device", "auto",
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"YOLO overlay failed (exit {e.returncode}).")
#         raise
#     return out_path
#
# def run_fight_export_cli(video_path: Path, weights_path: Path, out_root: Path):
#     """
#     Calls export_events_and_clips.py with your preferred defaults:
#     SEQ=16, SKIP=1, merge_gap=0, min_seg=0, pre/post pad=0,
#     SAVE_FIGHT_CLIPS=True, SAVE_NOFIGHT_CLIPS=False, MAKE_HIGHLIGHT=True
#     """
#     cmd = [
#         sys.executable, str(EXPORT_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out_root", str(out_root),
#         "--seq", "16",
#         "--skip", "1",
#         "--merge_gap_s", "0.0",
#         "--min_seg_dur_s", "0.0",
#         "--pre_pad_s", "0.0",
#         "--post_pad_s", "0.0",
#         "--highlight_name", "fight_highlight.mp4",
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"Fight export failed (exit {e.returncode}).")
#         raise
#
#     video_stem = Path(video_path).stem
#     result_base = out_root / video_stem
#     return result_base
#
# # [AUTO] Orchestrate the auto-pipeline safely (no infinite loops)
# def process_pipeline():
#     if st.session_state.processing:
#         return
#
#     # must have an uploaded video and sidebar params set
#     if not st.session_state.uploaded_video_path:
#         return
#
#     # --- Stage 1: YOLO overlay ---
#     if st.session_state.pipeline == "overlay_pending":
#         st.session_state.processing = True
#         with st.spinner("Running YOLO overlay…"):
#             run_dir = current_run_dir()
#             raw_path = run_dir / "yolo_overlay_raw.mp4"
#             web_path = run_dir / "yolo_overlay_web.mp4"
#             try:
#                 run_yolo_overlay_cli(
#                     video_path=Path(st.session_state.uploaded_video_path),
#                     weights_path=Path(st.session_state.yolo_weights),
#                     out_path=raw_path,
#                     conf=float(st.session_state.yolo_conf),
#                     codec=str(st.session_state.yolo_codec),
#                 )
#                 transcode_web_mp4(raw_path, web_path, width=720)
#                 st.session_state.yolo_overlay_path = str(web_path)
#                 st.session_state.step = max(st.session_state.step, 3)
#                 st.toast("YOLO overlay ready.", icon="✅")
#             finally:
#                 st.session_state.processing = False
#
#         st.session_state.pipeline = "fight_pending"
#         st.rerun()
#
#     # --- Stage 2: Fight detector ---
#     if st.session_state.pipeline == "fight_pending":
#         st.session_state.processing = True
#         with st.spinner("Detecting fights, exporting clips & analytics…"):
#             out_root = current_run_dir()
#             try:
#                 result_base = run_fight_export_cli(
#                     video_path=Path(st.session_state.uploaded_video_path),
#                     weights_path=Path(st.session_state.fight_weights),
#                     out_root=out_root
#                 )
#                 st.session_state.result_base = str(result_base)
#                 st.session_state.step = max(st.session_state.step, 4)
#                 st.toast(f"Fight results ready in: {result_base}", icon="🎬")
#             finally:
#                 st.session_state.processing = False
#
#         st.session_state.pipeline = "done"
#         st.rerun()
#
# # ----------------------------
# # UI
# # ----------------------------
# st.set_page_config(page_title="Fight Detection Dashboard", layout="wide")
# st.title("Milestone 3 — Fight Detection Dashboard")
#
# with st.sidebar:
#     st.header("Controls")
#
#     # YOLO overlay knobs
#     yolo_weights_str = st.text_input("YOLO weights (.pt)", value=str(DEFAULT_YOLO_WEIGHTS))
#     yolo_conf_val = st.slider("YOLO confidence", 0.0, 1.0, 0.40, 0.01)
#     yolo_codec_val = st.selectbox("YOLO output codec", ["mp4v", "XVID"], index=0, help="Use XVID if MP4 has playback issues")
#
#     # [AUTO] persist into session_state so pipeline can use them across reruns
#     st.session_state.yolo_weights = yolo_weights_str
#     st.session_state.yolo_conf = yolo_conf_val
#     st.session_state.yolo_codec = yolo_codec_val
#
#     st.markdown("---")
#
#     # Fight detector model
#     fight_weights_str = st.text_input("Fight model weights (.pth)", value=str(DEFAULT_FIGHT_WEIGHTS))
#     st.session_state.fight_weights = fight_weights_str  # [AUTO] persist
#
#     st.markdown("---")
#     if st.button("Start new analysis (reset)", use_container_width=True):
#         # clear Streamlit caches (prevents mixed dataframes/plots)
#         try:
#             st.cache_data.clear()
#         except Exception:
#             pass
#         try:
#             st.cache_resource.clear()
#         except Exception:
#             pass
#
#         # drop all per-run state
#         st.session_state.step = 1
#         st.session_state.uploaded_video_path = None
#         st.session_state.video_stem = None
#         st.session_state.result_base = None
#         st.session_state.yolo_overlay_path = None
#         st.session_state.pipeline = "idle"
#         st.session_state.processing = False
#         st.session_state.run_id = None
#
#         new_run()  # fresh, empty run folder
#         st.rerun()
#
# # ----- REAL TABS (now functional) -----
# tab1, tab2, tab3, tab4 = st.tabs(["① Overview", "② Detection", "③ Analytics", "④ Clips & Labels"])
#
# # ===========================
# # TAB ① — OVERVIEW (Upload)
# # ===========================
# with tab1:
#     # Step 1 — Upload video
#     st.subheader("1) Upload video")
#     up = st.file_uploader("Upload an MP4", type=["mp4"], key="up_mp4")
#
#     c1, c2 = st.columns([2,1])
#     with c1:
#         if up and not st.session_state.uploaded_video_path:
#             if not st.session_state.run_id:
#                 new_run()
#             saved = save_uploaded_video(up)
#             st.session_state.uploaded_video_path = str(saved)
#             st.session_state.video_stem = Path(saved).stem  # <-- keep this
#             st.session_state.step = max(st.session_state.step, 2)
#             st.success(f"Uploaded to: {saved}")
#             st.session_state.pipeline = "overlay_pending"
#             st.session_state.processing = False
#
#     with c2:
#         # Show live preview: overlay if ready, else original
#         if st.session_state.yolo_overlay_path and Path(st.session_state.yolo_overlay_path).exists():
#             st.caption(f"Run ID: {st.session_state.run_id} • YOLO overlay preview")
#             show_video_autoplay(Path(st.session_state.yolo_overlay_path), height_px=240)  # [AUTO] autoplay & loop
#         elif st.session_state.uploaded_video_path:
#             st.caption(f"Run ID: {st.session_state.run_id} • Original upload")
#             st.video(str(st.session_state.uploaded_video_path))
#
# # ==========================================
# # TAB ② — DETECTION (Overlay + Fight export)
# # ==========================================
# with tab2:
#     if st.session_state.step >= 2 and st.session_state.uploaded_video_path:
#         st.subheader("2) YOLO overlay (overview)")
#         c3, c4 = st.columns([1,2])
#         with c3:
#             if st.button("Re-run YOLO overlay", type="primary"):  # [AUTO] re-run button
#                 with st.spinner("Re-running YOLO overlay…"):
#                     run_dir = current_run_dir()
#                     raw_path = run_dir / "yolo_overlay_raw.mp4"
#                     web_path = run_dir / "yolo_overlay_web.mp4"
#                     try:
#                         overlay_raw = run_yolo_overlay_cli(
#                             video_path=Path(st.session_state.uploaded_video_path),
#                             weights_path=Path(st.session_state.yolo_weights),
#                             out_path=raw_path,
#                             conf=float(st.session_state.yolo_conf),
#                             codec=str(st.session_state.yolo_codec)
#                         )
#                         transcode_web_mp4(overlay_raw, web_path, width=720)
#                         st.session_state.yolo_overlay_path = str(web_path)
#                         st.session_state.step = max(st.session_state.step, 3)
#                         st.success(f"Overlay ready: {web_path.name}")
#                     except Exception:
#                         st.stop()
#         with c4:
#             p = st.session_state.yolo_overlay_path
#             if p and Path(p).exists() and Path(p).stat().st_size > 0:
#                 show_video_autoplay(Path(p), height_px=320)  # [AUTO] autoplay & loop
#                 raw = (current_run_dir() / "yolo_overlay_raw.mp4")
#                 if raw.exists():
#                     st.download_button("Download full overlay (raw)", data=raw.read_bytes(),
#                                        file_name="yolo_overlay.mp4", mime="video/mp4")
#             else:
#                 st.info("Run the overlay to preview detections.")
#
#         st.subheader("3) Detect fights, export clips + analytics")
#         if st.button("Re-run fight detector", type="primary"):  # [AUTO] re-run button
#             with st.spinner("Building clips & analytics…"):
#                 out_root = current_run_dir()
#                 try:
#                     result_base = run_fight_export_cli(
#                         video_path=Path(st.session_state.uploaded_video_path),
#                         weights_path=Path(st.session_state.fight_weights),
#                         out_root=out_root
#                     )
#                     st.session_state.result_base = str(result_base)
#                     st.session_state.step = max(st.session_state.step, 4)
#                     st.success(f"Results ready in: {result_base}")
#                 except Exception:
#                     st.stop()
#     else:
#         st.info("Upload a video in ① Overview.")
#
# # ======================================
# # TAB ③ — ANALYTICS (Highlights/Charts)
# # ======================================
# with tab3:
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         summary_json     = base / "summary.json"
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         highlight_mp4    = base / "fight_highlight.mp4"
#
#         st.subheader("4) Highlights")
#         if highlight_mp4.exists():
#             # [AUTO] autoplay highlight
#             show_video_autoplay(highlight_mp4, height_px=340)
#         else:
#             st.info("No highlight video found.")
#
#         # Metrics
#         st.markdown("### Overview")
#         summary = load_json_safe(summary_json) or {}
#         events_df = load_csv_safe(events_fight_csv)
#         c5, c6, c7, c8, c9, c10 = st.columns(6)
#
#         num_events = int(events_df["event_id"].count()) if isinstance(events_df, pd.DataFrame) else 0
#         total_fight_time = float(events_df["duration_s"].sum()) if isinstance(events_df, pd.DataFrame) else 0.0
#         mean_conf = float(events_df["avg_conf_fight"].mean()) if isinstance(events_df, pd.DataFrame) else 0.0
#         longest_event = float(events_df["duration_s"].max()) if isinstance(events_df, pd.DataFrame) else 0.0
#
#         c5.metric("Events", num_events)
#         c6.metric("Total fight time (s)", round(total_fight_time, 2))
#         c7.metric("Mean conf", round(mean_conf, 3))
#         c8.metric("Longest event (s)", round(longest_event, 2))
#         c9.metric("FPS", round(float(summary.get("fps", 0)), 2))
#         c10.metric("Duration (s)", round(float(summary.get("duration_s", 0)), 2))
#
#         # Timeline — events
#         st.markdown("### Timeline — Events")
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             df = events_df.copy()
#             df["start_dt"] = pd.to_datetime(df["start_time_s"], unit="s", origin="unix")
#             df["end_dt"] = pd.to_datetime(df["end_time_s"], unit="s", origin="unix")
#
#             fig = px.timeline(
#                 df, x_start="start_dt", x_end="end_dt", y="event_id", color="avg_conf_fight",
#                 hover_data={
#                     "event_id": True,
#                     "start_time_s": ":.3f",
#                     "end_time_s": ":.3f",
#                     "duration_s": ":.3f",
#                     "avg_conf_fight": ":.3f",
#                     "max_conf_fight": ":.3f",
#                     "start_dt": False, "end_dt": False,
#                 },
#                 color_continuous_scale="Turbo",
#             )
#             fig.update_yaxes(autorange="reversed", title="Event")
#             fig.update_xaxes(title="Time (mm:ss)", tickformat="%M:%S")
#             fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("events_fight.csv not found or empty.")
#
#         # Timeline — window strip
#         st.markdown("### Timeline — Window-level confidence")
#         tw = load_csv_safe(timeline_csv)
#         if isinstance(tw, pd.DataFrame) and not tw.empty and "fight_prob" in tw.columns:
#             tw2 = tw.copy()
#             tw2["t_center_s"] = (tw2["start_time_s"] + tw2["end_time_s"]) / 2.0
#             fig2 = px.scatter(
#                 tw2, x="t_center_s", y="fight_prob", color="label",
#                 hover_data={"start_time_s": ":.3f", "end_time_s": ":.3f", "label": True, "fight_prob": ":.3f"},
#             )
#             fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
#                                xaxis_title="Time (s)", yaxis_title="Fight prob.")
#             st.plotly_chart(fig2, use_container_width=True)
#         else:
#             st.info("timeline_windows.csv not found or missing 'fight_prob'.")
#     else:
#         st.info("Run the fight detector in ② Detection to see analytics.")
#
# # =========================================
# # TAB ④ — CLIPS & LABELS (Clips/Validate)
# # =========================================
# with tab4:
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         clips_fight_dir  = base / "clips" / "fight"
#
#         # Clips grid
#         st.subheader("Clips")
#         if clips_fight_dir.exists():
#             clips = sorted(list(clips_fight_dir.glob("*.mp4")))[:6]
#             if clips:
#                 cols = st.columns(3)
#                 for i, clip in enumerate(clips):
#                     with cols[i % 3]:
#                         show_video_autoplay(clip, height_px=180)  # [AUTO] autoplay & loop
#                         st.caption(clip.name)
#             with st.expander("Download all fight clips (.zip)"):
#                 data = zip_folder(clips_fight_dir)
#                 st.download_button("Download clips.zip", data=data, file_name="clips_fight.zip", mime="application/zip")
#             st.toast("Clips have been saved to the run folder.", icon="💾")  # [AUTO] small alert
#         else:
#             st.info("No fight clips directory found.")
#
#         # Validation
#         st.markdown("### Validate events")
#         events_df = load_csv_safe(events_fight_csv)
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             edit_df = events_df.copy()
#             if "validated_label" not in edit_df.columns:
#                 edit_df["validated_label"] = "fight"
#             if "note" not in edit_df.columns:
#                 edit_df["note"] = ""
#             edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed", key="editor_events")
#             if st.button("Save validation"):
#                 out_csv = base / "validated_events.csv"
#                 edited.to_csv(out_csv, index=False)
#                 st.success(f"Saved: {out_csv}")
#         else:
#             st.info("Nothing to validate yet.")
#
#         # Exports
#         st.markdown("### Exports")
#         e1, e2, e3 = st.columns(3)
#         with e1:
#             if events_fight_csv.exists():
#                 st.download_button(
#                     "Download events_fight.csv",
#                     data=events_fight_csv.read_bytes(), file_name="events_fight.csv", mime="text/csv"
#                 )
#         with e2:
#             if timeline_csv.exists():
#                 st.download_button(
#                     "Download timeline_windows.csv",
#                     data=timeline_csv.read_bytes(), file_name="timeline_windows.csv", mime="text/csv"
#                 )
#         with e3:
#             val_csv = base / "validated_events.csv"
#             if val_csv.exists():
#                 st.download_button(
#                     "Download validated_events.csv",
#                     data=val_csv.read_bytes(), file_name="validated_events.csv", mime="text/csv"
#                 )
#     else:
#         st.info("Run the fight detector in ② Detection to see clips & labels.")
#
# # [AUTO] Run the orchestrator at the very end so it can read all state/params
# process_pipeline()


# Milestone - 3 updated


# Fight Detection Dashboard (isolated runs)
# Notes:
# - Each upload is saved as <originalName>__<run_id>.mp4 inside outputs/<run_id>/
# - export_events_and_clips.py writes to out_root / <video_stem> (unique per run now)
# - Reset fully clears caches and state



# import os, io, sys, uuid, zipfile, json, subprocess, base64
# from pathlib import Path
# from datetime import datetime
#
# import streamlit as st
# import pandas as pd
# import plotly.express as px
#
# # ----------------------------
# # Paths (edit if your files live elsewhere)
# # ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# OUTPUTS_DIR = BASE_DIR / "outputs"
# OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#
# YOLO_SCRIPT   = BASE_DIR / "Yolov8_Violence_Detection.py"
# EXPORT_SCRIPT = BASE_DIR / "export_events_and_clips.py"
#
# DEFAULT_YOLO_WEIGHTS  = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\yolo_violence_detection_model.pt")
# DEFAULT_FIGHT_WEIGHTS = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\fight_detection.pth")
#
# # ----------------------------
# # Session state
# # ----------------------------
# if "run_id" not in st.session_state:
#     st.session_state.run_id = None
# if "uploaded_video_path" not in st.session_state:
#     st.session_state.uploaded_video_path = None
# if "result_base" not in st.session_state:
#     st.session_state.result_base = None  # outputs/<run_id>/<video_stem>/
# if "yolo_overlay_path" not in st.session_state:
#     st.session_state.yolo_overlay_path = None
# if "step" not in st.session_state:
#     st.session_state.step = 1
#
# # Auto-pipeline flags
# if "pipeline" not in st.session_state:
#     st.session_state.pipeline = "idle"   # idle -> overlay_pending -> fight_pending -> done
# if "processing" not in st.session_state:
#     st.session_state.processing = False
#
# def new_run():
#     if not st.session_state.run_id:
#         st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
#         (OUTPUTS_DIR / st.session_state.run_id).mkdir(parents=True, exist_ok=True)
#
# def current_run_dir() -> Path:
#     if not st.session_state.run_id:
#         new_run()
#     return OUTPUTS_DIR / st.session_state.run_id
#
# # ----------------------------
# # Helpers
# # ----------------------------
# def save_uploaded_video(file) -> Path:
#     """
#     Save the uploaded file into the current run folder with a UNIQUE name:
#     <original_stem>__<run_id>.mp4
#     This guarantees exporter isolation because it uses the video stem.
#     """
#     run_dir = current_run_dir()
#     orig = Path(file.name).stem or "input"
#     unique_name = f"{orig}__{st.session_state.run_id}.mp4"
#     dst = run_dir / unique_name  # IMPORTANT: do NOT use a fixed name like 'input.mp4'
#     with open(dst, "wb") as f:
#         f.write(file.read())
#     return dst
#
# def transcode_web_mp4(src: Path, dst: Path, width: int | None = 720):
#     """
#     Makes a browser-friendly MP4 (H.264 yuv420p). If ffmpeg not present, falls back to moviepy.
#     """
#     import shutil, subprocess
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if shutil.which("ffmpeg"):
#         cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                "-c:a","aac","-b:a","128k", str(dst)]
#         if width:
#             cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
#                    "-vf", f"scale={width}:-2",
#                    "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
#                    "-c:a","aac","-b:a","128k", str(dst)]
#         subprocess.run(cmd, check=True)
#     else:
#         from moviepy.editor import VideoFileClip
#         with VideoFileClip(str(src)) as clip:
#             if width:
#                 clip = clip.resize(width=width)
#             clip.write_videofile(str(dst), codec="libx264", audio_codec="aac", preset="veryfast", logger=None)
#
# def load_json_safe(path: Path) -> dict | None:
#     try:
#         if path.exists():
#             return json.loads(path.read_text())
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def load_csv_safe(path: Path) -> pd.DataFrame | None:
#     try:
#         if path.exists():
#             return pd.read_csv(path)
#     except Exception as e:
#         st.error(f"Failed to read {path.name}: {e}")
#     return None
#
# def zip_folder(folder: Path) -> bytes:
#     mem = io.BytesIO()
#     with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#         for p in folder.rglob("*"):
#             if p.is_file():
#                 zf.write(p, p.relative_to(folder))
#     mem.seek(0)
#     return mem.read()
#
# def show_video_autoplay(path: Path, width_css="100%", muted=True, loop=True, autoplay=True, controls=True, height_px: int | None = None):
#     if not path or not Path(path).exists():
#         return
#     try:
#         data = Path(path).read_bytes()
#         b64 = base64.b64encode(data).decode()
#         style_h = f"height:{height_px}px;" if height_px else ""
#         html = f"""
#         <video {'controls' if controls else ''} {'autoplay' if autoplay else ''} {'loop' if loop else ''} {'muted' if muted else ''} playsinline
#                style="width:{width_css}; {style_h}">
#             <source src="data:video/mp4;base64,{b64}" type="video/mp4">
#         </video>
#         """
#         st.markdown(html, unsafe_allow_html=True)
#     except Exception:
#         st.video(str(path))
#
# # ----------------------------
# # Pipelines (subprocess)
# # ----------------------------
# def run_yolo_overlay_cli(video_path: Path, weights_path: Path, out_path: Path, conf: float = 0.40, codec: str = "mp4v"):
#     cmd = [
#         sys.executable, str(YOLO_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out", str(out_path),
#         "--conf", f"{conf:.2f}",
#         "--codec", codec,
#         "--device", "auto",
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"YOLO overlay failed (exit {e.returncode}).")
#         raise
#     return out_path
#
# def run_fight_export_cli(video_path: Path, weights_path: Path, out_root: Path):
#     """
#     exporter writes into: out_root / <video_stem>
#     since our uploaded file is unique per run, the stem is unique too.
#     """
#     cmd = [
#         sys.executable, str(EXPORT_SCRIPT),
#         "--video", str(video_path),
#         "--weights", str(weights_path),
#         "--out_root", str(out_root),
#         "--seq", "16",
#         "--skip", "1",
#         "--merge_gap_s", "0.0",
#         "--min_seg_dur_s", "0.0",
#         "--pre_pad_s", "0.0",
#         "--post_pad_s", "0.0",
#         "--highlight_name", "fight_highlight.mp4",
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         st.error(f"Fight export failed (exit {e.returncode}).")
#         raise
#
#     video_stem = Path(video_path).stem
#     result_base = out_root / video_stem
#     return result_base
#
# # ----------------------------
# # Orchestrator (no infinite loops)
# # ----------------------------
# def process_pipeline():
#     if st.session_state.processing:
#         return
#     if not st.session_state.uploaded_video_path:
#         return
#
#     # Stage 1: YOLO overlay
#     if st.session_state.pipeline == "overlay_pending":
#         st.session_state.processing = True
#         with st.spinner("Running YOLO overlay…"):
#             run_dir = current_run_dir()
#             raw_path = run_dir / "yolo_overlay_raw.mp4"
#             web_path = run_dir / "yolo_overlay_web.mp4"
#             try:
#                 run_yolo_overlay_cli(
#                     video_path=Path(st.session_state.uploaded_video_path),
#                     weights_path=Path(st.session_state.yolo_weights),
#                     out_path=raw_path,
#                     conf=float(st.session_state.yolo_conf),
#                     codec=str(st.session_state.yolo_codec),
#                 )
#                 transcode_web_mp4(raw_path, web_path, width=720)
#                 st.session_state.yolo_overlay_path = str(web_path)
#                 st.session_state.step = max(st.session_state.step, 3)
#                 st.toast("YOLO overlay ready.", icon="✅")
#             finally:
#                 st.session_state.processing = False
#         st.session_state.pipeline = "fight_pending"
#         st.rerun()
#
#     # Stage 2: Fight detector
#     if st.session_state.pipeline == "fight_pending":
#         st.session_state.processing = True
#         with st.spinner("Detecting fights, exporting clips & analytics…"):
#             out_root = current_run_dir()
#             try:
#                 result_base = run_fight_export_cli(
#                     video_path=Path(st.session_state.uploaded_video_path),
#                     weights_path=Path(st.session_state.fight_weights),
#                     out_root=out_root
#                 )
#                 st.session_state.result_base = str(result_base)
#                 st.session_state.step = max(st.session_state.step, 4)
#                 st.toast(f"Fight results ready in: {result_base}", icon="🎬")
#             finally:
#                 st.session_state.processing = False
#         st.session_state.pipeline = "done"
#         st.rerun()
#
# # ----------------------------
# # UI
# # ----------------------------
# st.set_page_config(page_title="Fight Detection Dashboard", layout="wide")
# st.title("Milestone 3 — Fight Detection Dashboard")
#
# with st.sidebar:
#     st.header("Controls")
#
#     # YOLO overlay knobs
#     yolo_weights_str = st.text_input("YOLO weights (.pt)", value=str(DEFAULT_YOLO_WEIGHTS))
#     yolo_conf_val = st.slider("YOLO confidence", 0.0, 1.0, 0.40, 0.01)
#     yolo_codec_val = st.selectbox("YOLO output codec", ["mp4v", "XVID"], index=0, help="Use XVID if MP4 has playback issues")
#
#     # Persist for pipeline
#     st.session_state.yolo_weights = yolo_weights_str
#     st.session_state.yolo_conf = yolo_conf_val
#     st.session_state.yolo_codec = yolo_codec_val
#
#     st.markdown("---")
#
#     # Fight detector model
#     fight_weights_str = st.text_input("Fight model weights (.pth)", value=str(DEFAULT_FIGHT_WEIGHTS))
#     st.session_state.fight_weights = fight_weights_str
#
#     st.markdown("---")
#     if st.button("Start new analysis (reset)", use_container_width=True):
#         # clear caches
#         try: st.cache_data.clear()
#         except Exception: pass
#         try: st.cache_resource.clear()
#         except Exception: pass
#
#         # reset state
#         st.session_state.step = 1
#         st.session_state.uploaded_video_path = None
#         st.session_state.video_stem = None
#         st.session_state.result_base = None
#         st.session_state.yolo_overlay_path = None
#         st.session_state.pipeline = "idle"
#         st.session_state.processing = False
#         st.session_state.run_id = None
#
#         new_run()  # create fresh run folder
#         st.rerun()
#
# # ----- Tabs -----
# tab1, tab2, tab3, tab4 = st.tabs(["① Overview", "② Detection", "③ Analytics", "④ Clips & Labels"])
#
# # ===========================
# # TAB ① — OVERVIEW (Upload)
# # ===========================
# with tab1:
#     st.subheader("1) Upload video")
#     up = st.file_uploader("Upload an MP4", type=["mp4"], key="up_mp4")
#
#     c1, c2 = st.columns([2,1])
#     with c1:
#         if up and not st.session_state.uploaded_video_path:
#             if not st.session_state.run_id:
#                 new_run()
#             saved = save_uploaded_video(up)  # <- unique name per run
#             st.session_state.uploaded_video_path = str(saved)
#             st.session_state.video_stem = Path(saved).stem
#             st.session_state.step = max(st.session_state.step, 2)
#             st.success(f"Uploaded to: {saved}")
#             # kick pipeline
#             st.session_state.pipeline = "overlay_pending"
#             st.session_state.processing = False
#
#     with c2:
#         if st.session_state.yolo_overlay_path and Path(st.session_state.yolo_overlay_path).exists():
#             st.caption(f"Run ID: {st.session_state.run_id} • YOLO overlay preview")
#             show_video_autoplay(Path(st.session_state.yolo_overlay_path), height_px=240)
#         elif st.session_state.uploaded_video_path:
#             st.caption(f"Run ID: {st.session_state.run_id} • Original upload")
#             st.video(str(st.session_state.uploaded_video_path))
#
# # ==========================================
# # TAB ② — DETECTION (Overlay + Fight export)
# # ==========================================
# with tab2:
#     if st.session_state.step >= 2 and st.session_state.uploaded_video_path:
#         st.subheader("2) YOLO overlay (overview)")
#         c3, c4 = st.columns([1,2])
#         with c3:
#             if st.button("Re-run YOLO overlay", type="primary"):
#                 with st.spinner("Re-running YOLO overlay…"):
#                     run_dir = current_run_dir()
#                     raw_path = run_dir / "yolo_overlay_raw.mp4"
#                     web_path = run_dir / "yolo_overlay_web.mp4"
#                     try:
#                         overlay_raw = run_yolo_overlay_cli(
#                             video_path=Path(st.session_state.uploaded_video_path),
#                             weights_path=Path(st.session_state.yolo_weights),
#                             out_path=raw_path,
#                             conf=float(st.session_state.yolo_conf),
#                             codec=str(st.session_state.yolo_codec)
#                         )
#                         transcode_web_mp4(overlay_raw, web_path, width=720)
#                         st.session_state.yolo_overlay_path = str(web_path)
#                         st.session_state.step = max(st.session_state.step, 3)
#                         st.success(f"Overlay ready: {web_path.name}")
#                     except Exception:
#                         st.stop()
#         with c4:
#             p = st.session_state.yolo_overlay_path
#             if p and Path(p).exists() and Path(p).stat().st_size > 0:
#                 show_video_autoplay(Path(p), height_px=320)
#                 raw = (current_run_dir() / "yolo_overlay_raw.mp4")
#                 if raw.exists():
#                     st.download_button("Download full overlay (raw)", data=raw.read_bytes(),
#                                        file_name="yolo_overlay.mp4", mime="video/mp4")
#             else:
#                 st.info("Run the overlay to preview detections.")
#
#         st.subheader("3) Detect fights, export clips + analytics")
#         if st.button("Re-run fight detector", type="primary"):
#             with st.spinner("Building clips & analytics…"):
#                 out_root = current_run_dir()
#                 try:
#                     result_base = run_fight_export_cli(
#                         video_path=Path(st.session_state.uploaded_video_path),
#                         weights_path=Path(st.session_state.fight_weights),
#                         out_root=out_root
#                     )
#                     st.session_state.result_base = str(result_base)
#                     st.session_state.step = max(st.session_state.step, 4)
#                     st.success(f"Results ready in: {result_base}")
#                 except Exception:
#                     st.stop()
#     else:
#         st.info("Upload a video in ① Overview.")
#
# # ======================================
# # TAB ③ — ANALYTICS (Highlights/Charts)
# # ======================================
# with tab3:
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         summary_json     = base / "summary.json"
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         highlight_mp4    = base / "fight_highlight.mp4"
#
#         st.subheader("4) Highlights")
#         if highlight_mp4.exists():
#             show_video_autoplay(highlight_mp4, height_px=340)
#         else:
#             st.info("No highlight video found.")
#
#         # Metrics
#         st.markdown("### Overview")
#         summary = load_json_safe(summary_json) or {}
#         events_df = load_csv_safe(events_fight_csv)
#         c5, c6, c7, c8, c9, c10 = st.columns(6)
#
#         num_events = int(events_df["event_id"].count()) if isinstance(events_df, pd.DataFrame) else 0
#         total_fight_time = float(events_df["duration_s"].sum()) if isinstance(events_df, pd.DataFrame) else 0.0
#         mean_conf = float(events_df["avg_conf_fight"].mean()) if isinstance(events_df, pd.DataFrame) else 0.0
#         longest_event = float(events_df["duration_s"].max()) if isinstance(events_df, pd.DataFrame) else 0.0
#
#         c5.metric("Events", num_events)
#         c6.metric("Total fight time (s)", round(total_fight_time, 2))
#         c7.metric("Mean conf", round(mean_conf, 3))
#         c8.metric("Longest event (s)", round(longest_event, 2))
#         c9.metric("FPS", round(float(summary.get("fps", 0)), 2))
#         c10.metric("Duration (s)", round(float(summary.get("duration_s", 0)), 2))
#
#         # Timeline — events
#         st.markdown("### Timeline — Events")
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             df = events_df.copy()
#             df["start_dt"] = pd.to_datetime(df["start_time_s"], unit="s", origin="unix")
#             df["end_dt"] = pd.to_datetime(df["end_time_s"], unit="s", origin="unix")
#
#             fig = px.timeline(
#                 df, x_start="start_dt", x_end="end_dt", y="event_id", color="avg_conf_fight",
#                 hover_data={
#                     "event_id": True,
#                     "start_time_s": ":.3f",
#                     "end_time_s": ":.3f",
#                     "duration_s": ":.3f",
#                     "avg_conf_fight": ":.3f",
#                     "max_conf_fight": ":.3f",
#                     "start_dt": False, "end_dt": False,
#                 },
#                 color_continuous_scale="Turbo",
#             )
#             fig.update_yaxes(autorange="reversed", title="Event")
#             fig.update_xaxes(title="Time (mm:ss)", tickformat="%M:%S")
#             fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("events_fight.csv not found or empty.")
#
#         # Timeline — window strip
#         st.markdown("### Timeline — Window-level confidence")
#         tw = load_csv_safe(timeline_csv)
#         if isinstance(tw, pd.DataFrame) and not tw.empty and "fight_prob" in tw.columns:
#             tw2 = tw.copy()
#             tw2["t_center_s"] = (tw2["start_time_s"] + tw2["end_time_s"]) / 2.0
#             fig2 = px.scatter(
#                 tw2, x="t_center_s", y="fight_prob", color="label",
#                 hover_data={"start_time_s": ":.3f", "end_time_s": ":.3f", "label": True, "fight_prob": ":.3f"},
#             )
#             fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
#                                xaxis_title="Time (s)", yaxis_title="Fight prob.")
#             st.plotly_chart(fig2, use_container_width=True)
#         else:
#             st.info("timeline_windows.csv not found or missing 'fight_prob'.")
#     else:
#         st.info("Run the fight detector in ② Detection to see analytics.")
#
# # =========================================
# # TAB ④ — CLIPS & LABELS (Clips/Validate)
# # =========================================
# with tab4:
#     if st.session_state.step >= 4 and st.session_state.result_base:
#         base = Path(st.session_state.result_base)
#
#         events_fight_csv = base / "events_fight.csv"
#         timeline_csv     = base / "timeline_windows.csv"
#         clips_fight_dir  = base / "clips" / "fight"
#
#         # Clips grid
#         st.subheader("Clips")
#         if clips_fight_dir.exists():
#             clips = sorted(list(clips_fight_dir.glob("*.mp4")))[:6]
#             if clips:
#                 cols = st.columns(3)
#                 for i, clip in enumerate(clips):
#                     with cols[i % 3]:
#                         show_video_autoplay(clip, height_px=180)
#                         st.caption(clip.name)
#             with st.expander("Download all fight clips (.zip)"):
#                 data = zip_folder(clips_fight_dir)
#                 st.download_button("Download clips.zip", data=data, file_name="clips_fight.zip", mime="application/zip")
#             st.toast("Clips have been saved to the run folder.", icon="💾")
#         else:
#             st.info("No fight clips directory found.")
#
#         # Validation
#         st.markdown("### Validate events")
#         events_df = load_csv_safe(events_fight_csv)
#         if isinstance(events_df, pd.DataFrame) and not events_df.empty:
#             edit_df = events_df.copy()
#             if "validated_label" not in edit_df.columns:
#                 edit_df["validated_label"] = "fight"
#             if "note" not in edit_df.columns:
#                 edit_df["note"] = ""
#             edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed", key="editor_events")
#             if st.button("Save validation"):
#                 out_csv = base / "validated_events.csv"
#                 edited.to_csv(out_csv, index=False)
#                 st.success(f"Saved: {out_csv}")
#         else:
#             st.info("Nothing to validate yet.")
#
#         # Exports
#         st.markdown("### Exports")
#         e1, e2, e3 = st.columns(3)
#         with e1:
#             if events_fight_csv.exists():
#                 st.download_button(
#                     "Download events_fight.csv",
#                     data=events_fight_csv.read_bytes(), file_name="events_fight.csv", mime="text/csv"
#                 )
#         with e2:
#             if timeline_csv.exists():
#                 st.download_button(
#                     "Download timeline_windows.csv",
#                     data=timeline_csv.read_bytes(), file_name="timeline_windows.csv", mime="text/csv"
#                 )
#         with e3:
#             val_csv = base / "validated_events.csv"
#             if val_csv.exists():
#                 st.download_button(
#                     "Download validated_events.csv",
#                     data=val_csv.read_bytes(), file_name="validated_events.csv", mime="text/csv"
#                 )
#     else:
#         st.info("Run the fight detector in ② Detection to see clips & labels.")
#
# # Run orchestrator last
# process_pipeline()


import os, io, sys, uuid, zipfile, json, subprocess, base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# Paths (edit if your files live elsewhere)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_SCRIPT   = BASE_DIR / "Yolov8_Violence_Detection.py"
EXPORT_SCRIPT = BASE_DIR / "export_events_and_clips.py"

DEFAULT_YOLO_WEIGHTS  = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\yolo_violence_detection_model.pt")
DEFAULT_FIGHT_WEIGHTS = Path(r"D:\PycharmProject\JayantS_Fight_Dectection\fight_detection.pth")

# ----------------------------
# Session state
# ----------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "result_base" not in st.session_state:
    st.session_state.result_base = None  # outputs/<run_id>/<video_stem>/
if "yolo_overlay_path" not in st.session_state:
    st.session_state.yolo_overlay_path = None
if "step" not in st.session_state:
    st.session_state.step = 1

# Auto-pipeline flags
if "pipeline" not in st.session_state:
    st.session_state.pipeline = "idle"   # idle -> overlay_pending -> fight_pending -> done
if "processing" not in st.session_state:
    st.session_state.processing = False

# Alerts (run-scoped)
if "alerts" not in st.session_state:
    st.session_state.alerts = []  # list of dicts for current run
if "alerts_ready" not in st.session_state:
    st.session_state.alerts_ready = False
if "alert_preview_clip" not in st.session_state:
    st.session_state.alert_preview_clip = None

def new_run():
    if not st.session_state.run_id:
        st.session_state.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        (OUTPUTS_DIR / st.session_state.run_id).mkdir(parents=True, exist_ok=True)

def current_run_dir() -> Path:
    if not st.session_state.run_id:
        new_run()
    return OUTPUTS_DIR / st.session_state.run_id

# ----------------------------
# Helpers
# ----------------------------
def save_uploaded_video(file) -> Path:
    """
    Save the uploaded file into the current run folder with a UNIQUE name:
    <original_stem>__<run_id>.mp4
    This guarantees exporter isolation because it uses the video stem.
    """
    run_dir = current_run_dir()
    orig = Path(file.name).stem or "input"
    unique_name = f"{orig}__{st.session_state.run_id}.mp4"
    dst = run_dir / unique_name  # IMPORTANT: do NOT use a fixed name like 'input.mp4'
    with open(dst, "wb") as f:
        f.write(file.read())
    return dst

def transcode_web_mp4(src: Path, dst: Path, width: int | None = 720):
    """
    Makes a browser-friendly MP4 (H.264 yuv420p). If ffmpeg not present, falls back to moviepy.
    """
    import shutil, subprocess
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
               "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
               "-c:a","aac","-b:a","128k", str(dst)]
        if width:
            cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(src),
                   "-vf", f"scale={width}:-2",
                   "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","baseline","-preset","veryfast","-crf","22",
                   "-c:a","aac","-b:a","128k", str(dst)]
        subprocess.run(cmd, check=True)
    else:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(str(src)) as clip:
            if width:
                clip = clip.resize(width=width)
            clip.write_videofile(str(dst), codec="libx264", audio_codec="aac", preset="veryfast", logger=None)

def load_json_safe(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
    return None

def load_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
    return None

def zip_folder(folder: Path) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(folder))
    mem.seek(0)
    return mem.read()

def show_video_autoplay(path: Path, width_css="100%", muted=True, loop=True, autoplay=True, controls=True, height_px: int | None = None):
    if not path or not Path(path).exists():
        return
    try:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode()
        style_h = f"height:{height_px}px;" if height_px else ""
        html = f"""
        <video {'controls' if controls else ''} {'autoplay' if autoplay else ''} {'loop' if loop else ''} {'muted' if muted else ''} playsinline
               style="width:{width_css}; {style_h}">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.video(str(path))

def fmt_hms(seconds: float) -> str:
    try:
        s = float(seconds)
    except Exception:
        return ""
    if s < 0: s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    secs = s - h*3600 - m*60
    return f"{h:02d}:{m:02d}:{secs:06.3f}"

def collect_alerts(base_dir: Path):
    """
    Read events_fight.csv from base_dir and build run-scoped alerts.
    Also try to resolve the clip path for each event.
    """
    alerts = []
    events_csv = base_dir / "events_fight.csv"
    if not events_csv.exists():
        return []
    try:
        df = pd.read_csv(events_csv)
    except Exception:
        return []
    if df is None or df.empty:
        return []

    clips_dir = base_dir / "clips" / "fight"
    for _, row in df.iterrows():
        try:
            event_id = int(row.get("event_id", 0))
            st_s = float(row.get("start_time_s", 0.0))
            et_s = float(row.get("end_time_s", 0.0))
            dur_s = float(row.get("duration_s", max(0.0, et_s - st_s)))
            max_c = float(row.get("max_conf_fight", 0.0))
            avg_c = float(row.get("avg_conf_fight", 0.0))
        except Exception:
            continue

        clip_path = ""
        if clips_dir.exists():
            pattern = f"*F{event_id:03d}_fight_*.mp4"
            match = next(clips_dir.glob(pattern), None)
            if match and match.exists():
                clip_path = str(match)

        alerts.append({
            "event_id": event_id,
            "start_s": st_s,
            "end_s": et_s,
            "duration_s": dur_s,
            "max_conf": max_c,
            "avg_conf": avg_c,
            "start_hms": fmt_hms(st_s),
            "end_hms": fmt_hms(et_s),
            "clip_path": clip_path
        })
    return alerts

# ----------------------------
# Pipelines (subprocess)
# ----------------------------
def run_yolo_overlay_cli(video_path: Path, weights_path: Path, out_path: Path, conf: float = 0.40, codec: str = "mp4v"):
    cmd = [
        sys.executable, str(YOLO_SCRIPT),
        "--video", str(video_path),
        "--weights", str(weights_path),
        "--out", str(out_path),
        "--conf", f"{conf:.2f}",
        "--codec", codec,
        "--device", "auto",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"YOLO overlay failed (exit {e.returncode}).")
        raise
    return out_path

def run_fight_export_cli(video_path: Path, weights_path: Path, out_root: Path):
    """
    exporter writes into: out_root / <video_stem>
    since our uploaded file is unique per run, the stem is unique too.
    """
    cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--video", str(video_path),
        "--weights", str(weights_path),
        "--out_root", str(out_root),
        "--seq", "16",
        "--skip", "1",
        "--merge_gap_s", "0.0",
        "--min_seg_dur_s", "0.0",
        "--pre_pad_s", "0.0",
        "--post_pad_s", "0.0",
        "--highlight_name", "fight_highlight.mp4",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Fight export failed (exit {e.returncode}).")
        raise

    video_stem = Path(video_path).stem
    result_base = out_root / video_stem
    return result_base

# ----------------------------
# Orchestrator (no infinite loops)
# ----------------------------
def process_pipeline():
    if st.session_state.processing:
        return
    if not st.session_state.uploaded_video_path:
        return

    # Stage 1: YOLO overlay
    if st.session_state.pipeline == "overlay_pending":
        st.session_state.processing = True
        with st.spinner("Running YOLO overlay…"):
            run_dir = current_run_dir()
            raw_path = run_dir / "yolo_overlay_raw.mp4"
            web_path = run_dir / "yolo_overlay_web.mp4"
            try:
                run_yolo_overlay_cli(
                    video_path=Path(st.session_state.uploaded_video_path),
                    weights_path=Path(st.session_state.yolo_weights),
                    out_path=raw_path,
                    conf=float(st.session_state.yolo_conf),
                    codec=str(st.session_state.yolo_codec),
                )
                transcode_web_mp4(raw_path, web_path, width=720)
                st.session_state.yolo_overlay_path = str(web_path)
                st.session_state.step = max(st.session_state.step, 3)
                st.toast("YOLO overlay ready.", icon="✅")
            finally:
                st.session_state.processing = False
        st.session_state.pipeline = "fight_pending"
        st.rerun()

    # Stage 2: Fight detector
    if st.session_state.pipeline == "fight_pending":
        st.session_state.processing = True
        with st.spinner("Detecting fights, exporting clips & analytics…"):
            out_root = current_run_dir()
            try:
                result_base = run_fight_export_cli(
                    video_path=Path(st.session_state.uploaded_video_path),
                    weights_path=Path(st.session_state.fight_weights),
                    out_root=out_root
                )
                st.session_state.result_base = str(result_base)
                st.session_state.step = max(st.session_state.step, 4)
                # collect alerts for this run (no extra toasts)
                st.session_state.alerts = collect_alerts(Path(st.session_state.result_base))
                st.session_state.alerts_ready = True
            finally:
                st.session_state.processing = False
        st.session_state.pipeline = "done"
        st.rerun()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Fight Detection Dashboard", layout="wide")
st.title("Milestone 3 — Fight Detection Dashboard")

with st.sidebar:
    st.header("Controls")

    # YOLO overlay knobs
    yolo_weights_str = st.text_input("YOLO weights (.pt)", value=str(DEFAULT_YOLO_WEIGHTS))
    yolo_conf_val = st.slider("YOLO confidence", 0.0, 1.0, 0.40, 0.01)
    yolo_codec_val = st.selectbox("YOLO output codec", ["mp4v", "XVID"], index=0, help="Use XVID if MP4 has playback issues")

    # Persist for pipeline
    st.session_state.yolo_weights = yolo_weights_str
    st.session_state.yolo_conf = yolo_conf_val
    st.session_state.yolo_codec = yolo_codec_val

    st.markdown("---")

    # Fight detector model
    fight_weights_str = st.text_input("Fight model weights (.pth)", value=str(DEFAULT_FIGHT_WEIGHTS))
    st.session_state.fight_weights = fight_weights_str

    st.markdown("---")
    if st.button("Start new analysis (reset)", use_container_width=True):
        # clear caches
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass

        # reset state
        st.session_state.step = 1
        st.session_state.uploaded_video_path = None
        st.session_state.video_stem = None
        st.session_state.result_base = None
        st.session_state.yolo_overlay_path = None
        st.session_state.pipeline = "idle"
        st.session_state.processing = False
        st.session_state.run_id = None
        # clear alerts
        st.session_state.alerts = []
        st.session_state.alerts_ready = False
        st.session_state.alert_preview_clip = None

        new_run()  # create fresh run folder
        st.rerun()

    # ---------------- Sidebar Notifications ----------------
    st.markdown("---")
    st.subheader("Notifications")
    if st.session_state.alerts_ready and st.session_state.alerts:
        max_items = 10
        for al in st.session_state.alerts[:max_items]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<div style='width:12px;height:12px;background:#d83c7d;border-radius:3px'></div>"
                f"<div><strong>Fight {al['event_id']:03d}</strong> • {al['start_hms']}–{al['end_hms']} • "
                f"max {al['max_conf']:.2f} (avg {al['avg_conf']:.2f})</div></div>",
                unsafe_allow_html=True
            )

            clip = al.get("clip_path") or ""
            play_key = f"play_alert_{st.session_state.run_id}_{al['event_id']}"
            if clip and Path(clip).exists():
                if st.button("Play", key=play_key):
                    st.session_state.alert_preview_clip = clip

                # show preview RIGHT BELOW the clicked Play
                if st.session_state.alert_preview_clip == clip:
                    show_video_autoplay(Path(clip), height_px=180)
                    if st.button("Close", key=f"close_{play_key}"):
                        st.session_state.alert_preview_clip = None
            else:
                st.caption("Clip: unavailable")

        if len(st.session_state.alerts) > max_items:
            st.caption(f"…and {len(st.session_state.alerts) - max_items} more. See the **⑤ Alerts** tab.")
    else:
        st.caption("No alerts yet. Alerts appear here after detection finishes.")

# ----- Tabs -----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["① Overview", "② Detection", "③ Analytics", "④ Clips & Labels", "⑤ Alerts"])

# ===========================
# TAB ① — OVERVIEW (Upload)
# ===========================
with tab1:
    st.subheader("1) Upload video")
    up = st.file_uploader("Upload an MP4", type=["mp4"], key="up_mp4")

    c1, c2 = st.columns([2,1])
    with c1:
        if up and not st.session_state.uploaded_video_path:
            if not st.session_state.run_id:
                new_run()
            saved = save_uploaded_video(up)  # <- unique name per run
            st.session_state.uploaded_video_path = str(saved)
            st.session_state.video_stem = Path(saved).stem
            st.session_state.step = max(st.session_state.step, 2)
            st.success(f"Uploaded to: {saved}")
            # kick pipeline
            st.session_state.pipeline = "overlay_pending"
            st.session_state.processing = False
            # clear any stale alerts for safety
            st.session_state.alerts = []
            st.session_state.alerts_ready = False
            st.session_state.alert_preview_clip = None

    with c2:
        if st.session_state.yolo_overlay_path and Path(st.session_state.yolo_overlay_path).exists():
            st.caption(f"Run ID: {st.session_state.run_id} • YOLO overlay preview")
            show_video_autoplay(Path(st.session_state.yolo_overlay_path), height_px=240)
        elif st.session_state.uploaded_video_path:
            st.caption(f"Run ID: {st.session_state.run_id} • Original upload")
            st.video(str(st.session_state.uploaded_video_path))

# ==========================================
# TAB ② — DETECTION (Overlay + Fight export)
# ==========================================
with tab2:
    if st.session_state.step >= 2 and st.session_state.uploaded_video_path:
        st.subheader("2) YOLO overlay (overview)")
        c3, c4 = st.columns([1,2])
        with c3:
            if st.button("Re-run YOLO overlay", type="primary"):
                with st.spinner("Re-running YOLO overlay…"):
                    run_dir = current_run_dir()
                    raw_path = run_dir / "yolo_overlay_raw.mp4"
                    web_path = run_dir / "yolo_overlay_web.mp4"
                    try:
                        overlay_raw = run_yolo_overlay_cli(
                            video_path=Path(st.session_state.uploaded_video_path),
                            weights_path=Path(st.session_state.yolo_weights),
                            out_path=raw_path,
                            conf=float(st.session_state.yolo_conf),
                            codec=str(st.session_state.yolo_codec)
                        )
                        transcode_web_mp4(overlay_raw, web_path, width=720)
                        st.session_state.yolo_overlay_path = str(web_path)
                        st.session_state.step = max(st.session_state.step, 3)
                        st.success(f"Overlay ready: {web_path.name}")
                    except Exception:
                        st.stop()
        with c4:
            p = st.session_state.yolo_overlay_path
            if p and Path(p).exists() and Path(p).stat().st_size > 0:
                show_video_autoplay(Path(p), height_px=320)
                raw = (current_run_dir() / "yolo_overlay_raw.mp4")
                if raw.exists():
                    st.download_button("Download full overlay (raw)", data=raw.read_bytes(),
                                       file_name="yolo_overlay.mp4", mime="video/mp4")
            else:
                st.info("Run the overlay to preview detections.")

        st.subheader("3) Detect fights, export clips + analytics")
        if st.button("Re-run fight detector", type="primary"):
            with st.spinner("Building clips & analytics…"):
                out_root = current_run_dir()
                try:
                    result_base = run_fight_export_cli(
                        video_path=Path(st.session_state.uploaded_video_path),
                        weights_path=Path(st.session_state.fight_weights),
                        out_root=out_root
                    )
                    st.session_state.result_base = str(result_base)
                    st.session_state.step = max(st.session_state.step, 4)
                    st.success(f"Results ready in: {result_base}")
                    # refresh alerts after manual re-run
                    st.session_state.alerts = collect_alerts(Path(st.session_state.result_base))
                    st.session_state.alerts_ready = True
                except Exception:
                    st.stop()
    else:
        st.info("Upload a video in ① Overview.")

# ======================================
# TAB ③ — ANALYTICS (Highlights/Charts)
# ======================================
with tab3:
    if st.session_state.step >= 4 and st.session_state.result_base:
        base = Path(st.session_state.result_base)

        summary_json     = base / "summary.json"
        events_fight_csv = base / "events_fight.csv"
        timeline_csv     = base / "timeline_windows.csv"
        highlight_mp4    = base / "fight_highlight.mp4"

        st.subheader("4) Highlights")
        if highlight_mp4.exists():
            show_video_autoplay(highlight_mp4, height_px=340)
        else:
            st.info("No highlight video found.")

        # Metrics
        st.markdown("### Overview")
        summary = load_json_safe(summary_json) or {}
        events_df = load_csv_safe(events_fight_csv)
        c5, c6, c7, c8, c9, c10 = st.columns(6)

        num_events = int(events_df["event_id"].count()) if isinstance(events_df, pd.DataFrame) else 0
        total_fight_time = float(events_df["duration_s"].sum()) if isinstance(events_df, pd.DataFrame) else 0.0
        mean_conf = float(events_df["avg_conf_fight"].mean()) if isinstance(events_df, pd.DataFrame) else 0.0
        longest_event = float(events_df["duration_s"].max()) if isinstance(events_df, pd.DataFrame) else 0.0

        c5.metric("Events", num_events)
        c6.metric("Total fight time (s)", round(total_fight_time, 2))
        c7.metric("Mean conf", round(mean_conf, 3))
        c8.metric("Longest event (s)", round(longest_event, 2))
        c9.metric("FPS", round(float(summary.get("fps", 0)), 2))
        c10.metric("Duration (s)", round(float(summary.get("duration_s", 0)), 2))

        # Timeline — events
        st.markdown("### Timeline — Events")
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            df = events_df.copy()
            df["start_dt"] = pd.to_datetime(df["start_time_s"], unit="s", origin="unix")
            df["end_dt"] = pd.to_datetime(df["end_time_s"], unit="s", origin="unix")

            fig = px.timeline(
                df, x_start="start_dt", x_end="end_dt", y="event_id", color="avg_conf_fight",
                hover_data={
                    "event_id": True,
                    "start_time_s": ":.3f",
                    "end_time_s": ":.3f",
                    "duration_s": ":.3f",
                    "avg_conf_fight": ":.3f",
                    "max_conf_fight": ":.3f",
                    "start_dt": False, "end_dt": False,
                },
                color_continuous_scale="Turbo",
            )
            fig.update_yaxes(autorange="reversed", title="Event")
            fig.update_xaxes(title="Time (mm:ss)", tickformat="%M:%S")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("events_fight.csv not found or empty.")

        # Timeline — window strip
        st.markdown("### Timeline — Window-level confidence")
        tw = load_csv_safe(timeline_csv)
        if isinstance(tw, pd.DataFrame) and not tw.empty and "fight_prob" in tw.columns:
            tw2 = tw.copy()
            tw2["t_center_s"] = (tw2["start_time_s"] + tw2["end_time_s"]) / 2.0
            fig2 = px.scatter(
                tw2, x="t_center_s", y="fight_prob", color="label",
                hover_data={"start_time_s": ":.3f", "end_time_s": ":.3f", "label": True, "fight_prob": ":.3f"},
            )
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Time (s)", yaxis_title="Fight prob.")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("timeline_windows.csv not found or missing 'fight_prob'.")
    else:
        st.info("Run the fight detector in ② Detection to see analytics.")

# =========================================
# TAB ④ — CLIPS & LABELS (Clips/Validate)
# =========================================
with tab4:
    if st.session_state.step >= 4 and st.session_state.result_base:
        base = Path(st.session_state.result_base)

        events_fight_csv = base / "events_fight.csv"
        timeline_csv     = base / "timeline_windows.csv"
        clips_fight_dir  = base / "clips" / "fight"

        # Clips grid
        st.subheader("Clips")
        if clips_fight_dir.exists():
            clips = sorted(list(clips_fight_dir.glob("*.mp4")))[:6]
            if clips:
                cols = st.columns(3)
                for i, clip in enumerate(clips):
                    with cols[i % 3]:
                        show_video_autoplay(clip, height_px=180)
                        st.caption(clip.name)
            with st.expander("Download all fight clips (.zip)"):
                data = zip_folder(clips_fight_dir)
                st.download_button("Download clips.zip", data=data, file_name="clips_fight.zip", mime="application/zip")
        else:
            st.info("No fight clips directory found.")

        # Validation
        st.markdown("### Validate events")
        events_df = load_csv_safe(events_fight_csv)
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            edit_df = events_df.copy()
            if "validated_label" not in edit_df.columns:
                edit_df["validated_label"] = "fight"
            if "note" not in edit_df.columns:
                edit_df["note"] = ""
            edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed", key="editor_events")
            if st.button("Save validation"):
                out_csv = base / "validated_events.csv"
                edited.to_csv(out_csv, index=False)
                st.success(f"Saved: {out_csv}")
        else:
            st.info("Nothing to validate yet.")

        # Exports
        st.markdown("### Exports")
        e1, e2, e3 = st.columns(3)
        with e1:
            if events_fight_csv.exists():
                st.download_button(
                    "Download events_fight.csv",
                    data=events_fight_csv.read_bytes(), file_name="events_fight.csv", mime="text/csv"
                )
        with e2:
            if timeline_csv.exists():
                st.download_button(
                    "Download timeline_windows.csv",
                    data=timeline_csv.read_bytes(), file_name="timeline_windows.csv", mime="text/csv"
                )
        with e3:
            val_csv = base / "validated_events.csv"
            if val_csv.exists():
                st.download_button(
                    "Download validated_events.csv",
                    data=val_csv.read_bytes(), file_name="validated_events.csv", mime="text/csv"
                )
    else:
        st.info("Run the fight detector in ② Detection to see clips & labels.")

# =========================================
# TAB ⑤ — ALERTS (Run-scoped notifications)
# =========================================
with tab5:
    if st.session_state.step >= 4 and st.session_state.result_base:
        base = Path(st.session_state.result_base)
        # if alerts not ready for any reason, attempt to collect once
        if not st.session_state.alerts_ready:
            st.session_state.alerts = collect_alerts(base)
            st.session_state.alerts_ready = True if st.session_state.alerts else False

        st.subheader("Detected fight events")
        alerts = st.session_state.alerts or []
        if alerts:
            # Table view
            tbl = pd.DataFrame([{
                "Event": a["event_id"],
                "Start": a["start_hms"],
                "End": a["end_hms"],
                "Duration (s)": round(a["duration_s"], 3),
                "Max conf": round(a["max_conf"], 3),
                "Avg conf": round(a["avg_conf"], 3),
                "Clip": Path(a["clip_path"]).name if a["clip_path"] else ""
            } for a in alerts])
            st.dataframe(tbl, use_container_width=True, height=min(400, 60 + 28*len(tbl)))

            # Inline players
            st.markdown("### Preview clips")
            for a in alerts:
                title = f"Event {a['event_id']:03d} • {a['start_hms']}–{a['end_hms']} • max {a['max_conf']:.2f}"
                with st.expander(title):
                    if a["clip_path"] and Path(a["clip_path"]).exists():
                        show_video_autoplay(Path(a["clip_path"]), height_px=220)
                    else:
                        st.caption("Clip not found for this event.")
        else:
            st.info("No fight events found for this run.")
    else:
        st.info("Alerts will appear here after the fight detector finishes.")

# Run orchestrator last
process_pipeline()

