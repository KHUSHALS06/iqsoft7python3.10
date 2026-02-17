import asyncio
import websockets
import json
import socket
import threading
import uuid
import cv2
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from aiortc import RTCPeerConnection, RTCSessionDescription
import time
import traceback
from collections import deque
from queue import Queue
import warnings
from datetime import datetime
import hashlib
import os
import pickle
import sys

warnings.filterwarnings("ignore")

# ═══════════════════════ LICENSE ═════════════════════════════════════════════
MASTER_PASSWORD = "IQSOFT2026"
LICENSE_FILE    = "license.dat"

class LicenseManager:
    def __init__(self):
        self.licensed = False
        self._check()

    def _check(self):
        try:
            if os.path.exists(LICENSE_FILE):
                with open(LICENSE_FILE, "rb") as f:
                    if pickle.load(f).get("licensed"):
                        self.licensed = True
        except Exception as e:
            print(f"License check: {e}")

    def activate(self, password):
        if password == MASTER_PASSWORD:
            try:
                with open(LICENSE_FILE, "wb") as f:
                    pickle.dump({"licensed": True,
                                 "activated": datetime.now().isoformat()}, f)
                self.licensed = True
                return True
            except Exception as e:
                print(f"License save: {e}")
        return False

    def reset(self):
        try:
            if os.path.exists(LICENSE_FILE):
                os.remove(LICENSE_FILE)
            self.licensed = False
        except Exception as e:
            print(f"License reset: {e}")

license_manager = LicenseManager()
if not license_manager.licensed:
    root = tk.Tk(); root.withdraw()
    pw = simpledialog.askstring("License", "Enter License Password:", show="*", parent=root)
    if not pw or not license_manager.activate(pw):
        messagebox.showerror("License Error", "Invalid password.\nExiting.")
        root.destroy(); sys.exit(1)
    root.destroy()

# ═══════════════════════ CONFIG ═══════════════════════════════════════════════
WS_PORT        = 8888
TCP_AUDIO_PORT = 5000
SAMPLE_RATE    = 16000
CHUNK_SIZE     = 320          # 20 ms @ 16 kHz

# ── Display ───────────────────────────────────────────────────────────────────
# Display at native 720p; the canvas will scale down on smaller screens.
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
DISPLAY_DELAY_MS = 33         # ~30 fps UI refresh

# ── Stream buffer ──────────────────────────────────────────────────────────────
TARGET_FPS            = 20
# Larger buffer = smoother on high-ping; 5 s worth of frames
STREAM_BUFFER_SECONDS = 5
BUFFER_SIZE           = int(TARGET_FPS * STREAM_BUFFER_SECONDS)

# ── Freeze-frame ──────────────────────────────────────────────────────────────
# Freeze after this many seconds of no new frames
FREEZE_TIMEOUT_NORMAL = 3.0   # normal conditions
FREEZE_TIMEOUT_HIGHPING = 6.0 # give more grace when ping is high
# Ping threshold (ms) above which we consider the link "high ping"
HIGH_PING_THRESHOLD   = 800
# Ping threshold above which we freeze immediately (connection likely broken)
FREEZE_PING_THRESHOLD = 3000

# ── Adaptive quality ──────────────────────────────────────────────────────────
QUALITY_PRESETS = {
    "240p": (426,  240),
    "360p": (640,  360),
    "480p": (854,  480),
    "720p": (1280, 720),
    "1080p":(1920, 1080),
}
# Server-side receive pipeline stays at whatever the Android sends; we only
# scale *down* for display if the frame is bigger than DISPLAY_WIDTH×DISPLAY_HEIGHT.
VIDEO_WIDTH  = 1280
VIDEO_HEIGHT = 720

# ── Network quality monitor ───────────────────────────────────────────────────
NETWORK_CHECK_INTERVAL = 3.0
FRAME_DROP_THRESHOLD   = 0.12  # 12 % drop rate triggers quality reduction
QUALITY_INCREASE_DELAY = 15.0  # seconds stable before increasing

# ── Frame validation ──────────────────────────────────────────────────────────
MIN_BRIGHTNESS = 1.0
MAX_BRIGHTNESS = 254.0
MIN_STD_DEV    = 0.5  # very lenient – only reject truly solid frames

# ── Audio ──────────────────────────────────────────────────────────────────────
AUDIO_JITTER_BUFFER       = 30
AUDIO_UNDERRUN_THRESHOLD  = 5

# ═══════════════════════ GLOBAL STATE ═════════════════════════════════════════
mic_muted     = False
speaker_muted = False
is_recording  = False
video_writer  = None
recording_filename = None

latest_frame   = None
frame_lock     = threading.Lock()
frame_count    = 0
fps_display    = 0
last_fps_time  = time.time()

frame_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock  = threading.Lock()

last_good_frame      = None
last_good_frame_lock = threading.Lock()
last_frame_time      = 0.0
last_frame_time_lock = threading.Lock()
is_frozen            = False
freeze_lock          = threading.Lock()

last_valid_frame       = None
frame_validation_lock  = threading.Lock()
consecutive_bad_frames = 0

network_stats = {
    "dropped_frames":    0,
    "total_frames":      0,
    "last_quality_change": 0.0,
    "current_quality":   "720p",
    "target_width":      VIDEO_WIDTH,
    "target_height":     VIDEO_HEIGHT,
    "ping":              0,
    "last_ping_time":    0.0,
}
network_stats_lock = threading.Lock()

streams       = {}
server_running = False
streams_lock   = threading.Lock()
active_tracks  = {}
active_tracks_lock = threading.Lock()
main_loop = None

# ═══════════════════════ HELPERS ══════════════════════════════════════════════
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except:
        return "127.0.0.1"

def crop_to_aspect_ratio(frame, tw, th):
    """Crop (never stretch) to target aspect ratio, then resize."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    tr = tw / th; cr = w / h
    if abs(cr - tr) > 0.01:
        if cr > tr:
            nw = int(h * tr); x = (w - nw) // 2; frame = frame[:, x:x+nw]
        else:
            nh = int(w / tr); y = (h - nh) // 2; frame = frame[y:y+nh, :]
    if frame.shape[1] != tw or frame.shape[0] != th:
        interp = cv2.INTER_LANCZOS4 if tw > frame.shape[1] else cv2.INTER_AREA
        frame  = cv2.resize(frame, (tw, th), interpolation=interp)
    return frame

# ═══════════════════════ FRAME VALIDATOR ══════════════════════════════════════
def validate_frame(frame):
    global last_valid_frame, consecutive_bad_frames, last_good_frame
    if frame is None or frame.size == 0 or len(frame.shape) != 3:
        consecutive_bad_frames += 1
        return None
    mean = np.mean(frame)
    std  = np.std(frame)
    if mean < MIN_BRIGHTNESS or mean > MAX_BRIGHTNESS or std < MIN_STD_DEV:
        consecutive_bad_frames += 1
        with frame_validation_lock:
            return last_valid_frame.copy() if last_valid_frame is not None else None
    consecutive_bad_frames = 0
    with frame_validation_lock:
        last_valid_frame = frame.copy()
    with last_good_frame_lock:
        last_good_frame = frame.copy()
    return frame

# ═══════════════════════ AUDIO ════════════════════════════════════════════════
class AudioProcessor:
    def __init__(self):
        self.frame_size = CHUNK_SIZE

    def process(self, data):
        if isinstance(data, bytes):
            a = np.frombuffer(data, dtype=np.int16)
        else:
            a = data.astype(np.int16)
        if len(a) != self.frame_size:
            if len(a) < self.frame_size:
                a = np.pad(a, (0, self.frame_size - len(a)))
            else:
                a = a[:self.frame_size]
        return a.astype(np.int16)


class TCPAudioServer:
    def __init__(self):
        self.is_running     = False
        self.mic_queue      = Queue(maxsize=50)
        self.client_socket  = None
        self.server_socket  = None
        self.stream         = None
        self.audio_jitter   = deque(maxlen=AUDIO_JITTER_BUFFER)
        self.buf_init       = False
        self.underrun_count = 0
        self.processor      = AudioProcessor()

    def start(self):
        self.is_running = True
        threading.Thread(target=self._serve, daemon=True).start()
        print(f"TCP Audio listening on :{TCP_AUDIO_PORT}")

    def _serve(self):
        while self.is_running:
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(("0.0.0.0", TCP_AUDIO_PORT))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0)
                print("Waiting for audio client…")
                while self.is_running:
                    try:
                        cs, addr = self.server_socket.accept()
                        print(f"Audio client: {addr}")
                        self.client_socket = cs
                        self._handle(cs)
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.is_running: print(f"Accept: {e}")
                        break
            except Exception as e:
                if self.is_running: print(f"Server: {e}")
                time.sleep(1)
            finally:
                try: self.server_socket.close()
                except: pass

    def _handle(self, cs):
        def cb(indata, outdata, frames, ti, status):
            try:
                if not speaker_muted:
                    if not self.buf_init:
                        if len(self.audio_jitter) >= AUDIO_UNDERRUN_THRESHOLD:
                            self.buf_init = True
                        else:
                            outdata.fill(0); return
                    if self.audio_jitter:
                        pkt = self.audio_jitter.popleft()
                        a   = np.frombuffer(pkt, dtype=np.int16).astype(np.float32) / 32767.0
                        n   = min(len(a), frames)
                        outdata[:n, 0] = a[:n]
                        if n < frames: outdata[n:, 0] = 0
                    else:
                        self.underrun_count += 1
                        self.buf_init = False
                        outdata.fill(0)
                else:
                    outdata.fill(0)

                if not mic_muted and self.client_socket:
                    mic = (indata[:, 0] * 32767).astype(np.int16)
                    try: self.mic_queue.put_nowait(mic.tobytes())
                    except: pass
            except: outdata.fill(0)

        try:
            self.stream = sd.Stream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype="float32", callback=cb,
                                    blocksize=CHUNK_SIZE, latency="low")
            self.stream.start()

            def send():
                while self.is_running and self.client_socket:
                    try:
                        if not self.mic_queue.empty():
                            self.client_socket.sendall(self.mic_queue.get_nowait())
                    except: break
                    time.sleep(0.001)

            def recv():
                buf = b""; esz = CHUNK_SIZE * 2
                while self.is_running and self.client_socket:
                    try:
                        d = self.client_socket.recv(esz * 2)
                        if not d: break
                        buf += d
                        while len(buf) >= esz:
                            self.audio_jitter.append(buf[:esz])
                            buf = buf[esz:]
                    except socket.timeout: continue
                    except: break

            st = threading.Thread(target=send, daemon=True)
            rt = threading.Thread(target=recv, daemon=True)
            st.start(); rt.start()
            st.join(); rt.join()
        except Exception as e:
            print(f"Audio error: {e}")
        finally:
            try: self.stream.stop(); self.stream.close()
            except: pass
            try: cs.close()
            except: pass
            self.audio_jitter.clear()
            self.buf_init = False
            print("Audio client disconnected")

    def stop(self):
        self.is_running = False
        for attr in ("stream", "client_socket", "server_socket"):
            try: getattr(self, attr) and getattr(self, attr).close()
            except: pass


audio_server = TCPAudioServer()

# ═══════════════════════ CONNECTION MONITOR ════════════════════════════════════
def monitor_connection():
    global is_frozen
    while server_running:
        try:
            time.sleep(0.5)
            now = time.time()
            with last_frame_time_lock:
                delta = now - last_frame_time
            with network_stats_lock:
                ping = network_stats["ping"]

            # Pick timeout generously on high-ping links
            timeout = FREEZE_TIMEOUT_HIGHPING if ping > HIGH_PING_THRESHOLD \
                      else FREEZE_TIMEOUT_NORMAL

            should_freeze = (delta > timeout) or (ping > FREEZE_PING_THRESHOLD)

            with freeze_lock:
                was = is_frozen
                is_frozen = should_freeze
                if is_frozen and not was:
                    print(f"🔒 FREEZE  Δt={delta:.1f}s  ping={ping}ms")
                elif not is_frozen and was:
                    print("▶️  UNFREEZE")
        except Exception as e:
            print(f"monitor_connection: {e}")


# ═══════════════════════ ADAPTIVE QUALITY ═════════════════════════════════════
def adjust_quality():
    levels = ["720p", "480p", "360p", "240p"]   # server display levels (no 1080p receive-side)
    while server_running:
        try:
            time.sleep(NETWORK_CHECK_INTERVAL)
            with network_stats_lock:
                total = network_stats["total_frames"]
                if total < 60:
                    continue
                drop = network_stats["dropped_frames"] / total
                since = time.time() - network_stats["last_quality_change"]
                cur   = network_stats["current_quality"]
                idx   = levels.index(cur) if cur in levels else 0

                if drop > FRAME_DROP_THRESHOLD and idx < len(levels) - 1:
                    nq = levels[idx + 1]
                    network_stats["current_quality"]  = nq
                    network_stats["target_width"], network_stats["target_height"] = QUALITY_PRESETS[nq]
                    network_stats["last_quality_change"] = time.time()
                    print(f"⚠️  Quality ↓ → {nq}  (drop={drop:.1%})")
                elif drop < 0.03 and since > QUALITY_INCREASE_DELAY and idx > 0:
                    nq = levels[idx - 1]
                    network_stats["current_quality"]  = nq
                    network_stats["target_width"], network_stats["target_height"] = QUALITY_PRESETS[nq]
                    network_stats["last_quality_change"] = time.time()
                    print(f"✅  Quality ↑ → {nq}")

                network_stats["dropped_frames"] = 0
                network_stats["total_frames"]   = 0
        except Exception as e:
            print(f"adjust_quality: {e}")


# ═══════════════════════ WEBRTC SERVER ════════════════════════════════════════
async def handle_client(ws, path=None):
    cid = str(uuid.uuid4())[:8]
    print(f"WS connected: {cid}")
    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
                t    = data.get("type")

                if t == "register":
                    dev = data.get("device", f"dev_{cid}")
                    print(f"Registered: {dev} → {cid}")
                    with streams_lock:
                        streams[cid] = {"active": True, "pc": None,
                                        "device_name": dev, "ts": time.time()}
                    await ws.send(json.dumps({"type": "welcome",
                                              "message": f"Welcome {dev}!",
                                              "client_id": cid}))

                elif t == "offer":
                    pc = RTCPeerConnection()
                    with streams_lock:
                        if cid in streams: streams[cid]["pc"] = pc

                    @pc.on("track")
                    async def on_track(track):
                        if track.kind != "video": return
                        print(f"Video track ← {cid}")
                        with active_tracks_lock:
                            active_tracks[cid] = True

                        fc = 0
                        try:
                            while True:
                                try:
                                    frame = await asyncio.wait_for(track.recv(), timeout=8.0)
                                    fc += 1

                                    with last_frame_time_lock:
                                        global last_frame_time
                                        last_frame_time = time.time()

                                    img = frame.to_ndarray(format="bgr24")

                                    # ── High-quality resize pipeline ─────────────
                                    # We use INTER_LANCZOS4 for upscaling and
                                    # INTER_AREA for downscaling – both are much
                                    # sharper than the default INTER_LINEAR.
                                    with network_stats_lock:
                                        tw = network_stats["target_width"]
                                        th = network_stats["target_height"]

                                    processed = crop_to_aspect_ratio(img, tw, th)
                                    if processed is None:
                                        continue

                                    validated = validate_frame(processed)
                                    with network_stats_lock:
                                        network_stats["total_frames"] += 1
                                        if validated is None:
                                            network_stats["dropped_frames"] += 1
                                            continue

                                    global latest_frame, frame_count, fps_display, last_fps_time
                                    with frame_lock:
                                        latest_frame = validated.copy()
                                        frame_count += 1

                                    now = time.time()
                                    if now - last_fps_time >= 1.0:
                                        fps_display = frame_count
                                        frame_count = 0
                                        last_fps_time = now

                                    with buffer_lock:
                                        frame_buffer.append(validated.copy())

                                    if is_recording and video_writer:
                                        video_writer.write(validated)

                                    if fc % 150 == 0:
                                        print(f"{cid}: {fc} frames @ {fps_display} fps")

                                except asyncio.TimeoutError:
                                    continue
                                except Exception as e:
                                    if "track ended" in str(e).lower(): break
                                    continue
                        finally:
                            with active_tracks_lock:
                                active_tracks.pop(cid, None)

                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send(json.dumps({"type": "answer",
                                              "sdp": pc.localDescription.sdp}))
                    print(f"Answer → {cid}")

                elif t == "ice_candidate":
                    try:
                        if cid in streams and streams[cid]["pc"]:
                            c = data.get("candidate")
                            m = data.get("sdpMid")
                            i = data.get("sdpMLineIndex")
                            if c and m is not None and i is not None:
                                await streams[cid]["pc"].addIceCandidate(
                                    {"candidate": c, "sdpMid": m, "sdpMLineIndex": i})
                    except: pass

                elif t == "heartbeat":
                    await ws.send(json.dumps({"type": "pong"}))

                elif t == "ping":
                    pv = data.get("ping", 0)
                    with network_stats_lock:
                        network_stats["ping"] = pv
                        network_stats["last_ping_time"] = time.time()

                elif t == "stream_end":
                    break

            except json.JSONDecodeError: pass
            except Exception as e: print(f"msg error: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"WS closed: {cid}")
    finally:
        with streams_lock:      streams.pop(cid, None)
        with active_tracks_lock: active_tracks.pop(cid, None)


async def start_ws():
    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT,
                                ping_interval=20, ping_timeout=40,
                                max_size=10**7):
        print(f"WebRTC server: ws://{get_local_ip()}:{WS_PORT}")
        await asyncio.Future()


def run_asyncio():
    global main_loop
    while server_running:
        try:
            main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(main_loop)
            main_loop.run_until_complete(start_ws())
        except Exception as e:
            print(f"asyncio: {e}")
            time.sleep(1)
        finally:
            try: main_loop.close()
            except: pass


# ═══════════════════════ VIDEO DISPLAY ════════════════════════════════════════
class VideoDisplay(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, bg="#000000", highlightthickness=0)
        self.photo = None
        self.playback_started = False
        self.after(DISPLAY_DELAY_MS, self._tick)

    def _tick(self):
        try:
            self.delete("all")
            dw, dh = DISPLAY_WIDTH, DISPLAY_HEIGHT
            frame = None

            with freeze_lock:
                frozen = is_frozen

            if frozen:
                with last_good_frame_lock:
                    if last_good_frame is not None:
                        frame = last_good_frame.copy()
            else:
                with buffer_lock:
                    bf = len(frame_buffer)
                    if not self.playback_started:
                        if bf >= max(3, int(BUFFER_SIZE * 0.15)):
                            self.playback_started = True
                    if self.playback_started and bf > 0:
                        frame = frame_buffer[bf // 2]
                    elif bf > 0:
                        frame = frame_buffer[-1]
                    elif latest_frame is not None:
                        frame = latest_frame

            if frame is not None:
                try:
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fh, fw = rgb.shape[:2]
                    scale  = min(dw / fw, dh / fh)
                    nw, nh = int(fw * scale), int(fh * scale)
                    # INTER_LANCZOS4 for upscale, INTER_AREA for downscale
                    interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
                    disp   = cv2.resize(rgb, (nw, nh), interpolation=interp)

                    img        = Image.fromarray(disp)
                    self.photo = ImageTk.PhotoImage(img)
                    x = (dw - nw) // 2; y = (dh - nh) // 2
                    self.create_image(x, y, image=self.photo, anchor="nw")

                    # ── OSD ─────────────────────────────────────────────────
                    with network_stats_lock:
                        qs  = network_stats["current_quality"]
                        qw  = network_stats["target_width"]
                        qh  = network_stats["target_height"]
                        ping= network_stats["ping"]
                    with buffer_lock:
                        bf = len(frame_buffer)

                    ping_col = "#FF4444" if ping > HIGH_PING_THRESHOLD else "#00FF00"

                    def osd(x, y, text, color="#00FF00"):
                        # shadow for readability
                        self.create_text(x+1, y+1, text=text, fill="#000000",
                                         font=("Consolas", 10, "bold"), anchor="nw")
                        self.create_text(x,   y,   text=text, fill=color,
                                         font=("Consolas", 10, "bold"), anchor="nw")

                    osd(10, 10, f"Quality : {qs}  {qw}×{qh}")
                    osd(10, 28, f"FPS     : {fps_display}", "#FFFF00")
                    osd(10, 46, f"Buffer  : {bf}/{BUFFER_SIZE}",
                        "#00FF00" if bf > BUFFER_SIZE * 0.25 else "#FFA500")
                    osd(10, 64, f"Ping    : {ping} ms", ping_col)
                    if frozen:
                        self.create_rectangle(10, 88, 175, 112,
                                              fill="#CC0000", outline="white", width=2)
                        self.create_text(92, 100, text="⏸  FROZEN",
                                         fill="white", font=("Consolas", 12, "bold"))
                    if is_recording:
                        self.create_oval(dw-28, 12, dw-10, 30,
                                         fill="red", outline="white")
                        self.create_text(dw-70, 21, text="REC",
                                         fill="red", font=("Consolas", 12, "bold"))
                except Exception as e:
                    print(f"Display render: {e}")
            else:
                self.create_text(dw//2, dh//2,
                                 text="Waiting for video stream…",
                                 fill="#555555", font=("Arial", 16))
        except Exception:
            pass
        finally:
            self.after(DISPLAY_DELAY_MS, self._tick)


# ═══════════════════════ UI ═══════════════════════════════════════════════════
def create_ui():
    root = tk.Tk()
    root.title("IQSOFT Streaming Server – 720p / High-Ping Mode")
    root.geometry(f"{DISPLAY_WIDTH + 10}x{DISPLAY_HEIGHT + 110}")
    root.configure(bg="#1a1a1a")
    root.bind("<F11>", lambda e: root.attributes("-fullscreen",
              not root.attributes("-fullscreen")))
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    # ── Header ─────────────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg="#1a1a1a", height=55)
    hdr.pack(fill="x", padx=8, pady=4)
    hdr.pack_propagate(False)

    ip = get_local_ip()
    tk.Label(hdr,
             text=f"IP: {ip}   Audio: :{TCP_AUDIO_PORT}   Video: ws://{ip}:{WS_PORT}",
             fg="#00FF00", bg="#1a1a1a",
             font=("Consolas", 11, "bold")).pack(side="left", pady=6)

    ctrl = tk.Frame(hdr, bg="#1a1a1a")
    ctrl.pack(side="right", fill="y", padx=6)

    sstate = {"running": False}

    def start_server():
        if sstate["running"]: return
        global server_running, last_frame_time
        server_running = True; sstate["running"] = True
        with last_frame_time_lock: last_frame_time = time.time()

        audio_server.start()
        threading.Thread(target=run_asyncio,       daemon=True).start()
        threading.Thread(target=adjust_quality,    daemon=True).start()
        threading.Thread(target=monitor_connection, daemon=True).start()

        sbtn.config(state="disabled"); xbtn.config(state="normal")
        print(f"\n{'='*60}\nSERVER STARTED  –  720p native\n"
              f"IP={ip}  Audio={TCP_AUDIO_PORT}  WS={WS_PORT}\n"
              f"Display: {DISPLAY_WIDTH}×{DISPLAY_HEIGHT}\n"
              f"Buffer: {STREAM_BUFFER_SECONDS}s ({BUFFER_SIZE} frames)\n"
              f"Freeze thresholds: normal={FREEZE_TIMEOUT_NORMAL}s  "
              f"high-ping={FREEZE_TIMEOUT_HIGHPING}s\n{'='*60}\n")

    def stop_server():
        if not sstate["running"]: return
        global server_running
        server_running = False; sstate["running"] = False
        audio_server.stop()
        sbtn.config(state="normal"); xbtn.config(state="disabled")
        print("\nSERVER STOPPED\n")

    sbtn = tk.Button(ctrl, text="START", command=start_server,
                     bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                     width=7, relief="flat")
    sbtn.pack(side="left", padx=3)

    xbtn = tk.Button(ctrl, text="STOP", command=stop_server,
                     bg="#f44336", fg="white", font=("Arial", 10, "bold"),
                     width=7, relief="flat", state="disabled")
    xbtn.pack(side="left", padx=3)

    def toggle_mic():
        global mic_muted
        mic_muted = not mic_muted
        mbtn.config(text=f"Mic: {'OFF' if mic_muted else 'ON'}",
                    bg="#f44336" if mic_muted else "#4CAF50")
    mbtn = tk.Button(ctrl, text="Mic: ON", command=toggle_mic,
                     bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                     width=8, relief="flat")
    mbtn.pack(side="left", padx=2)

    def toggle_spk():
        global speaker_muted
        speaker_muted = not speaker_muted
        sbtn2.config(text=f"Spk: {'OFF' if speaker_muted else 'ON'}",
                     bg="#f44336" if speaker_muted else "#4CAF50")
    sbtn2 = tk.Button(ctrl, text="Spk: ON", command=toggle_spk,
                      bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                      width=8, relief="flat")
    sbtn2.pack(side="left", padx=2)

    def toggle_record():
        global is_recording, video_writer, recording_filename
        if not sstate["running"]:
            messagebox.showinfo("Info", "Start server first"); return
        if not is_recording:
            fn  = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fcc = cv2.VideoWriter_fourcc(*"XVID")
            with network_stats_lock:
                rw, rh = network_stats["target_width"], network_stats["target_height"]
            video_writer = cv2.VideoWriter(fn, fcc, TARGET_FPS, (rw, rh))
            if video_writer.isOpened():
                is_recording = True; recording_filename = fn
                rbtn.config(text="STOP REC", bg="#f44336")
                print(f"Recording: {fn} {rw}×{rh}")
            else:
                messagebox.showerror("Error", "Cannot start recording")
        else:
            is_recording = False
            if video_writer: video_writer.release(); video_writer = None
            rbtn.config(text="RECORD", bg="#FF9800")
            print(f"Saved: {recording_filename}")

    rbtn = tk.Button(ctrl, text="RECORD", command=toggle_record,
                     bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                     width=8, relief="flat")
    rbtn.pack(side="left", padx=4)

    # ── Video canvas ──────────────────────────────────────────────────────────
    cf = tk.Frame(root, bg="#000000", bd=1, relief="sunken",
                  width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    cf.pack_propagate(False)
    cf.pack(padx=0, pady=0)

    vd = VideoDisplay(cf)
    vd.pack(fill="both", expand=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    sb = tk.Frame(root, bg="#222222", height=28)
    sb.pack(fill="x", side="bottom")
    tk.Label(sb,
             text="F11 = fullscreen  |  720p display  |  LANCZOS4 upscale  |  adaptive quality",
             bg="#222222", fg="#888888", font=("Arial", 9)).pack(side="left", padx=8)

    def on_close():
        if sstate["running"]: stop_server()
        if video_writer: video_writer.release()
        root.quit(); root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    return root


# ═══════════════════════ MAIN ═════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print(f"IQSOFT Streaming Server  –  720p / High-Ping Mode")
    print(f"Display : {DISPLAY_WIDTH}×{DISPLAY_HEIGHT}  @  {TARGET_FPS} fps")
    print(f"Buffer  : {STREAM_BUFFER_SECONDS}s ({BUFFER_SIZE} frames)")
    print(f"License : {'✓ Active' if license_manager.licensed else '✗ Not activated'}")
    print("="*60)
    try:
        root = create_ui(); root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down…")
    except Exception as e:
        print(f"Fatal: {e}"); traceback.print_exc()
    finally:
        if video_writer: video_writer.release()
        audio_server.stop()
        print("Done.")
