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
from tkinter import messagebox
from PIL import Image, ImageTk
from aiortc import RTCPeerConnection, RTCSessionDescription
import time
from collections import deque
from queue import Queue
import warnings
from datetime import datetime
import sys
import subprocess
warnings.filterwarnings("ignore")

# ═══════════════════════ LICENSE ═════════════════════════════════════════════
AUTHORIZED_MACS = {
    "74-D8-3E-AB-9C-E9",  
    "B0-5C-DA-F4-F1-E5",  
}

def get_mac():
    try:
        result = subprocess.check_output("getmac /v /fo csv", shell=True)
        lines = result.decode(errors="ignore").strip().splitlines()[1:]
        macs = []
        for line in lines:
            parts = line.split('","')
            if len(parts) >= 3:
                macs.append(parts[2].strip().strip('"'))
        return macs
    except:
        return []

def check_license():
    macs = get_mac()
    if not any(m in AUTHORIZED_MACS for m in macs):
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("Unauthorized Device",
                             "This application is not licensed for this device.\nContact the developer.")
        r.destroy(); sys.exit(1)

check_license()

# ═══════════════════════ CONFIG ═══════════════════════════════════════════════
WS_PORT        = 8888
TCP_AUDIO_PORT = 5000
SAMPLE_RATE    = 16000
CHUNK_SIZE     = 320

DISPLAY_WIDTH    = 960
DISPLAY_HEIGHT   = 600
DISPLAY_DELAY_MS = 33

TARGET_FPS      = 20
MIN_GOOD_PIXELS = 1000

# ═══════════════════════ GLOBAL STATE ═════════════════════════════════════════
mic_muted     = False
speaker_muted = False
is_recording  = False
video_writer  = None

latest_frame = None
frame_lock   = threading.Lock()

last_good_frame      = None
last_good_frame_lock = threading.Lock()

stream_lost      = False
stream_lost_lock = threading.Lock()

frame_count   = 0
fps_display   = 0
last_fps_time = time.time()

connection_status = {"connected": False, "client_id": None}
connection_lock   = threading.Lock()

streams        = {}
server_running = False
streams_lock   = threading.Lock()

main_loop    = None
audio_server = None

# ═══════════════════════ HELPERS ══════════════════════════════════════════════
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except:
        return "127.0.0.1"

def is_good_frame(img):
    if img is None: return False
    if len(img.shape) != 3: return False
    if img.shape[0] < 10 or img.shape[1] < 10: return False
    if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < MIN_GOOD_PIXELS:
        return False
    return True

# ═══════════════════════ AUDIO ════════════════════════════════════════════════
class _ClientSession:
    def __init__(self, client_socket):
        self.cs         = client_socket
        self.is_running = True
        self.mic_queue  = Queue(maxsize=50)
        self.jitter     = deque(maxlen=30)
        self.buf_init   = False
        self.stream     = None

    def run(self):
        expected = CHUNK_SIZE * 2

        def audio_callback(indata, outdata, frames, ti, status):
            if not speaker_muted:
                if not self.buf_init:
                    if len(self.jitter) >= 5:
                        self.buf_init = True
                    else:
                        outdata.fill(0); return
                if self.jitter:
                    try:
                        pkt = self.jitter.popleft()
                        a   = np.frombuffer(pkt, dtype=np.int16).astype(np.float32) / 32767.0
                        n   = min(len(a), frames)
                        outdata[:n, 0] = a[:n]
                        if n < frames: outdata[n:, 0] = 0.0
                    except:
                        outdata.fill(0)
                else:
                    self.buf_init = False
                    outdata.fill(0)
            else:
                outdata.fill(0)

            if not mic_muted:
                try:
                    mic = (indata[:, 0] * 32767).astype(np.int16)
                    self.mic_queue.put_nowait(mic.tobytes())
                except:
                    pass

        try:
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=audio_callback, blocksize=CHUNK_SIZE, latency="low")
            self.stream.start()

            send_t = threading.Thread(target=self._send_loop, daemon=True)
            recv_t = threading.Thread(target=self._recv_loop, args=(expected,), daemon=True)
            send_t.start(); recv_t.start()
            send_t.join();  recv_t.join()
        except Exception as e:
            print(f"[Audio] Session error: {e}")
        finally:
            self._cleanup()

    def _send_loop(self):
        while self.is_running:
            try:
                if not self.mic_queue.empty():
                    self.cs.sendall(self.mic_queue.get_nowait())
            except:
                break
            time.sleep(0.001)

    def _recv_loop(self, expected):
        buf = b""
        while self.is_running:
            try:
                chunk = self.cs.recv(expected * 2)
                if not chunk: break
                buf += chunk
                while len(buf) >= expected:
                    self.jitter.append(buf[:expected])
                    buf = buf[expected:]
            except:
                break
        self.is_running = False

    def stop(self):
        self.is_running = False
        self._cleanup()

    def _cleanup(self):
        try:
            if self.stream: self.stream.stop(); self.stream.close(); self.stream = None
        except: pass
        try: self.cs.close()
        except: pass
        self.jitter.clear()
        self.buf_init = False


class TCPAudioServer:
    def __init__(self):
        self.is_running       = False
        self.server_socket    = None
        self._client_lock     = threading.Lock()
        self._current_session = None

    def start(self):
        self.is_running = True
        threading.Thread(target=self._serve, daemon=True, name="TCPAudioServe").start()

    def _serve(self):
        while self.is_running:
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(("0.0.0.0", TCP_AUDIO_PORT))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0)
                print(f"[Audio] Listening on :{TCP_AUDIO_PORT}")

                while self.is_running:
                    try:
                        cs, addr = self.server_socket.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                    print(f"[Audio] Client connected from {addr}")
                    with self._client_lock:
                        if self._current_session:
                            self._current_session.stop()
                            self._current_session = None
                        session = _ClientSession(cs)
                        self._current_session = session

                    threading.Thread(target=session.run, daemon=True, name="TCPAudioSession").start()

            except OSError as e:
                if self.is_running:
                    print(f"[Audio] Server socket error: {e} — retrying in 2s")
                    time.sleep(2)
            finally:
                try:
                    if self.server_socket: self.server_socket.close(); self.server_socket = None
                except: pass

    def stop(self):
        self.is_running = False
        with self._client_lock:
            if self._current_session: self._current_session.stop(); self._current_session = None
        try:
            if self.server_socket: self.server_socket.close(); self.server_socket = None
        except: pass

# ═══════════════════════ WEBRTC SERVER ════════════════════════════════════════
async def handle_client(ws, path=None):
    global stream_lost
    cid = str(uuid.uuid4())[:8]

    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
                t    = data.get("type")

                if t == "register":
                    dev = data.get("device", f"dev_{cid}")
                    with streams_lock:
                        streams[cid] = {"active": True, "pc": None, "device_name": dev}
                    with connection_lock:
                        connection_status["client_id"] = cid
                        connection_status["connected"] = True
                    with stream_lost_lock:
                        stream_lost = False
                    await ws.send(json.dumps({"type": "welcome", "message": f"Welcome {dev}!", "client_id": cid}))

                elif t == "offer":
                    pc = RTCPeerConnection()
                    with streams_lock:
                        if cid in streams: streams[cid]["pc"] = pc

                    @pc.on("track")
                    async def on_track(track):
                        global stream_lost
                        if track.kind != "video": return

                        with stream_lost_lock:
                            stream_lost = False

                        try:
                            while True:
                                try:
                                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                    img = frame.to_ndarray(format="bgr24")

                                    if is_good_frame(img):
                                        with last_good_frame_lock:
                                            global last_good_frame
                                            last_good_frame = img.copy()

                                    global latest_frame, frame_count, fps_display, last_fps_time
                                    with frame_lock:
                                        latest_frame = img

                                    frame_count += 1
                                    now = time.time()
                                    if now - last_fps_time >= 1.0:
                                        fps_display = frame_count
                                        frame_count = 0
                                        last_fps_time = now

                                    if is_recording and video_writer:
                                        video_writer.write(img)

                                except asyncio.TimeoutError:
                                    with stream_lost_lock:
                                        stream_lost = True
                                    print("[Video] Frame timeout — showing last good frame")
                                    continue

                                except Exception:
                                    break

                        finally:
                            with stream_lost_lock:
                                stream_lost = True

                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))

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

                elif t == "stream_end":
                    break

            except: pass

    except: pass
    finally:
        with stream_lost_lock:
            stream_lost = True
        with streams_lock:
            streams.pop(cid, None)
        with connection_lock:
            if connection_status["client_id"] == cid:
                connection_status["connected"] = False


async def start_ws():
    try:
        async with websockets.serve(
                handle_client, "0.0.0.0", WS_PORT,
                ping_interval=20, ping_timeout=40, max_size=10**7):
            await asyncio.Future()
    except: pass


def run_asyncio():
    global main_loop
    while server_running:
        try:
            main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(main_loop)
            main_loop.run_until_complete(start_ws())
        except:
            time.sleep(2)
        finally:
            try:
                if main_loop and not main_loop.is_closed():
                    main_loop.close()
            except: pass

# ═══════════════════════ VIDEO DISPLAY ════════════════════════════════════════
class VideoDisplay(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, bg="#000000", highlightthickness=0)
        self.photo = None
        self.after(DISPLAY_DELAY_MS, self._tick)

    def _tick(self):
        try:
            self.delete("all")
            dw, dh = DISPLAY_WIDTH, DISPLAY_HEIGHT

            with stream_lost_lock:
                lost = stream_lost

            if lost:
                with last_good_frame_lock:
                    frame = last_good_frame
            else:
                with frame_lock:
                    frame = latest_frame

            if frame is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(resized)
                    self.photo = ImageTk.PhotoImage(img)
                    self.create_image(0, 0, image=self.photo, anchor="nw")

                    if not lost:
                        self.create_text(10, dh - 20, text=f"FPS: {fps_display}",
                                         fill="#00FF00", font=("Arial", 10), anchor="w")
                    else:
                        self.create_text(10, dh - 20, text="Reconnecting...",
                                         fill="#FFAA00", font=("Arial", 10), anchor="w")

                    if is_recording:
                        self.create_oval(dw - 28, 12, dw - 10, 30, fill="red", outline="white")

                except: pass
            else:
                self.create_text(dw // 2, dh // 2,
                                 text="Waiting for video...",
                                 fill="#555555", font=("Arial", 16))
        except: pass
        finally:
            self.after(DISPLAY_DELAY_MS, self._tick)

# ═══════════════════════ UI ═══════════════════════════════════════════════════
def create_ui():
    root = tk.Tk()
    root.title("IQSOFT Streaming Server")
    root.geometry(f"{DISPLAY_WIDTH + 10}x{DISPLAY_HEIGHT + 80}")
    root.configure(bg="#1a1a1a")

    hdr = tk.Frame(root, bg="#1a1a1a", height=40)
    hdr.pack(fill="x", padx=8, pady=4)
    hdr.pack_propagate(False)

    ip = get_local_ip()
    tk.Label(hdr, text=f"IP: {ip}  Port: {WS_PORT}",
             fg="#00FF00", bg="#1a1a1a", font=("Consolas", 10)).pack(side="left", pady=8)

    ctrl = tk.Frame(hdr, bg="#1a1a1a")
    ctrl.pack(side="right", fill="y", padx=6)

    sstate = {"running": False}

    def start_server():
        if sstate["running"]: return
        global server_running, audio_server
        server_running = True
        sstate["running"] = True
        audio_server = TCPAudioServer()
        audio_server.start()
        threading.Thread(target=run_asyncio, daemon=True).start()
        sbtn.config(state="disabled")
        xbtn.config(state="normal")

    def stop_server():
        if not sstate["running"]: return
        global server_running, audio_server
        server_running = False
        sstate["running"] = False
        if audio_server:
            audio_server.stop()
            audio_server = None
        if main_loop and not main_loop.is_closed():
            try: main_loop.call_soon_threadsafe(main_loop.stop)
            except: pass
        sbtn.config(state="normal")
        xbtn.config(state="disabled")

    sbtn = tk.Button(ctrl, text="START", command=start_server,
                     bg="#4CAF50", fg="white", font=("Arial", 9), width=6, relief="flat")
    sbtn.pack(side="left", padx=2)

    xbtn = tk.Button(ctrl, text="STOP", command=stop_server,
                     bg="#f44336", fg="white", font=("Arial", 9), width=6, relief="flat", state="disabled")
    xbtn.pack(side="left", padx=2)

    def toggle_mic():
        global mic_muted
        mic_muted = not mic_muted
        mbtn.config(text=f"Mic: {'OFF' if mic_muted else 'ON'}",
                    bg="#f44336" if mic_muted else "#4CAF50")

    mbtn = tk.Button(ctrl, text="Mic: ON", command=toggle_mic,
                     bg="#4CAF50", fg="white", font=("Arial", 8), width=6, relief="flat")
    mbtn.pack(side="left", padx=2)

    def toggle_spk():
        global speaker_muted
        speaker_muted = not speaker_muted
        spkbtn.config(text=f"Spk: {'OFF' if speaker_muted else 'ON'}",
                      bg="#f44336" if speaker_muted else "#4CAF50")

    spkbtn = tk.Button(ctrl, text="Spk: ON", command=toggle_spk,
                       bg="#4CAF50", fg="white", font=("Arial", 8), width=6, relief="flat")
    spkbtn.pack(side="left", padx=2)

    def toggle_record():
        global is_recording, video_writer
        if not sstate["running"]: return
        if not is_recording:
            try:
                fn  = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                fcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(fn, fcc, TARGET_FPS, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                if video_writer.isOpened():
                    is_recording = True
                    rbtn.config(text="STOP", bg="#f44336")
            except: pass
        else:
            is_recording = False
            if video_writer: video_writer.release(); video_writer = None
            rbtn.config(text="REC", bg="#FF9800")

    rbtn = tk.Button(ctrl, text="REC", command=toggle_record,
                     bg="#FF9800", fg="white", font=("Arial", 8), width=4, relief="flat")
    rbtn.pack(side="left", padx=2)

    cf = tk.Frame(root, bg="#000000", bd=1, relief="sunken",
                  width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    cf.pack_propagate(False)
    cf.pack(padx=0, pady=0)

    vd = VideoDisplay(cf)
    vd.pack(fill="both", expand=True)

    def on_close():
        if sstate["running"]: stop_server()
        if video_writer: video_writer.release()
        root.quit(); root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    return root

# ═══════════════════════ MAIN ═════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        root = create_ui()
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if video_writer: video_writer.release()
        if audio_server: audio_server.stop()
