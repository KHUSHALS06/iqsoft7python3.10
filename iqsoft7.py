#!/usr/bin/env python3
"""
ENHANCED Android Streaming Server
- TCP two-way audio with ECHO CANCELLATION & VOICE CLARITY
- WebRTC video @ 15 FPS
- Speex DSP for audio processing
"""

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
from av import VideoFrame
import time
import traceback
from collections import deque
from queue import Queue
import warnings

warnings.filterwarnings("ignore")

# Try to import speexdsp for echo cancellation
try:
    import speexdsp
    SPEEX_AVAILABLE = True
    print("✅ Speex DSP available - Echo cancellation ENABLED")
except ImportError:
    SPEEX_AVAILABLE = False
    print("⚠️  Speex DSP not available - Install with: pip install speexdsp-python")
    print("    Falling back to basic noise reduction")

# ================= CONFIG =================
WS_PORT = 8888
TCP_AUDIO_PORT = 5000
SAMPLE_RATE = 16000
CHUNK_SIZE = 320  # 20ms at 16kHz for better processing
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Video quality settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Audio processing settings
ENABLE_ECHO_CANCELLATION = True
ENABLE_NOISE_REDUCTION = True
ENABLE_AGC = True  # Automatic Gain Control
ECHO_TAIL_MS = 200  # Echo tail length in milliseconds

# ================= GLOBAL STATE =================
frames = {}
frame_buffers = {}
streams = {}
server_running = False
streams_lock = threading.Lock()
active_tracks = {}
active_tracks_lock = threading.Lock()
main_loop = None

# ================= VIDEO BUFFER =================
class VideoBuffer:
    def __init__(self, client_id, maxsize=15):
        self.client_id = client_id
        self.buffer = deque(maxlen=maxsize)
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.target_interval = FRAME_INTERVAL

    def add_frame(self, frame, timestamp):
        self.buffer.append((timestamp, frame))
        self.frame_count += 1

    def get_frame(self):
        current_time = time.time()
        
        if current_time - self.last_frame_time < self.target_interval:
            return self.last_frame
        
        if self.buffer:
            timestamp, frame = self.buffer.popleft()
            try:
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            except:
                pass
            self.last_frame = frame
            self.last_frame_time = current_time
            return frame
        
        return self.last_frame

# ================= AUDIO PROCESSOR WITH ECHO CANCELLATION =================
class AudioProcessor:
    """Advanced audio processor with echo cancellation and voice clarity"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.frame_size = CHUNK_SIZE
        
        # Initialize Speex components if available
        if SPEEX_AVAILABLE and ENABLE_ECHO_CANCELLATION:
            try:
                # Echo canceller
                self.echo_state = speexdsp.EchoCanceller.create(
                    self.frame_size,
                    int(ECHO_TAIL_MS * self.sample_rate / 1000),
                    self.sample_rate
                )
                print("✅ Echo cancellation initialized")
            except Exception as e:
                print(f"⚠️  Echo cancellation init failed: {e}")
                self.echo_state = None
        else:
            self.echo_state = None
        
        if SPEEX_AVAILABLE and ENABLE_NOISE_REDUCTION:
            try:
                # Noise suppression
                self.noise_state = speexdsp.NoiseSuppression.create(
                    self.frame_size,
                    self.sample_rate
                )
                self.noise_state.suppression = -25  # Suppression in dB
                print("✅ Noise reduction initialized")
            except Exception as e:
                print(f"⚠️  Noise reduction init failed: {e}")
                self.noise_state = None
        else:
            self.noise_state = None
        
        if SPEEX_AVAILABLE and ENABLE_AGC:
            try:
                # Automatic Gain Control
                self.agc_state = speexdsp.AGC.create(
                    self.sample_rate,
                    self.frame_size
                )
                self.agc_state.level = 24000  # Target level
                self.agc_state.max_gain = 30  # Max gain in dB
                print("✅ Automatic Gain Control initialized")
            except Exception as e:
                print(f"⚠️  AGC init failed: {e}")
                self.agc_state = None
        else:
            self.agc_state = None
        
        # Simple filters as fallback
        self.use_simple_filters = not SPEEX_AVAILABLE
        if self.use_simple_filters:
            self.prev_samples = np.zeros(3)
            print("📊 Using simple audio filters")
    
    def process_microphone(self, audio_data, speaker_data=None):
        """
        Process microphone input with echo cancellation
        
        Args:
            audio_data: Microphone input (int16)
            speaker_data: What's being played on speakers (int16) - for echo cancellation
        
        Returns:
            Processed audio (int16)
        """
        try:
            # Convert to int16 numpy array
            if isinstance(audio_data, bytes):
                audio = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio = audio_data.astype(np.int16)
            
            # Ensure correct size
            if len(audio) != self.frame_size:
                if len(audio) < self.frame_size:
                    audio = np.pad(audio, (0, self.frame_size - len(audio)))
                else:
                    audio = audio[:self.frame_size]
            
            processed = audio.copy()
            
            # Apply echo cancellation if available
            if self.echo_state is not None and speaker_data is not None:
                try:
                    # Convert speaker data
                    if isinstance(speaker_data, bytes):
                        echo = np.frombuffer(speaker_data, dtype=np.int16)
                    else:
                        echo = speaker_data.astype(np.int16)
                    
                    if len(echo) != self.frame_size:
                        if len(echo) < self.frame_size:
                            echo = np.pad(echo, (0, self.frame_size - len(echo)))
                        else:
                            echo = echo[:self.frame_size]
                    
                    # Cancel echo
                    processed = self.echo_state.process(processed, echo)
                except Exception as e:
                    pass  # Fail silently, use unprocessed audio
            
            # Apply noise reduction
            if self.noise_state is not None:
                try:
                    processed = self.noise_state.process(processed)
                except:
                    pass
            
            # Apply AGC
            if self.agc_state is not None:
                try:
                    processed = self.agc_state.process(processed)
                except:
                    pass
            
            # Fallback: simple filters
            if self.use_simple_filters:
                processed = self._simple_denoise(processed)
            
            return processed.astype(np.int16)
        
        except Exception as e:
            # Return original on error
            if isinstance(audio_data, bytes):
                return audio_data
            return audio_data.astype(np.int16)
    
    def process_speaker(self, audio_data):
        """
        Process speaker output for clarity
        
        Args:
            audio_data: Speaker audio (int16)
        
        Returns:
            Processed audio (int16)
        """
        try:
            if isinstance(audio_data, bytes):
                audio = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio = audio_data.astype(np.int16)
            
            if len(audio) != self.frame_size:
                if len(audio) < self.frame_size:
                    audio = np.pad(audio, (0, self.frame_size - len(audio)))
                else:
                    audio = audio[:self.frame_size]
            
            processed = audio.copy()
            
            # Apply noise reduction to incoming audio
            if self.noise_state is not None:
                try:
                    processed = self.noise_state.process(processed)
                except:
                    pass
            
            # Simple denoise as fallback
            if self.use_simple_filters:
                processed = self._simple_denoise(processed)
            
            return processed.astype(np.int16)
        
        except Exception as e:
            if isinstance(audio_data, bytes):
                return audio_data
            return audio_data.astype(np.int16)
    
    def _simple_denoise(self, audio):
        """Simple moving average filter for noise reduction"""
        try:
            # Convert to float
            audio_float = audio.astype(np.float32)
            
            # Simple 3-point moving average
            filtered = np.zeros_like(audio_float)
            filtered[0] = (self.prev_samples[2] + audio_float[0] + audio_float[1]) / 3
            
            for i in range(1, len(audio_float) - 1):
                filtered[i] = (audio_float[i-1] + audio_float[i] + audio_float[i+1]) / 3
            
            filtered[-1] = (audio_float[-2] + audio_float[-1] + audio_float[-1]) / 3
            
            # Update history
            self.prev_samples[0] = self.prev_samples[1]
            self.prev_samples[1] = self.prev_samples[2]
            self.prev_samples[2] = audio_float[-1]
            
            return filtered.astype(np.int16)
        except:
            return audio
    
    def reset(self):
        """Reset all processors"""
        try:
            if self.echo_state:
                self.echo_state = speexdsp.EchoCanceller.create(
                    self.frame_size,
                    int(ECHO_TAIL_MS * self.sample_rate / 1000),
                    self.sample_rate
                )
        except:
            pass

# ================= ENHANCED TCP AUDIO SERVER =================
class TCPAudioServer:
    def __init__(self):
        self.is_running = False
        self.mic_queue = Queue(maxsize=30)
        self.speaker_queue = Queue(maxsize=30)
        self.client_socket = None
        self.server_socket = None
        self.stream = None
        self.processor = AudioProcessor()
        
        # Echo cancellation buffers
        self.last_speaker_data = None
        self.speaker_buffer = deque(maxlen=5)  # Keep last few speaker frames

    def start(self):
        """Start TCP audio server"""
        self.is_running = True
        threading.Thread(target=self._run_server, daemon=True).start()
        print(f"🎧 TCP Audio Server listening on port {TCP_AUDIO_PORT}")

    def _run_server(self):
        """Run server"""
        while self.is_running:
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(("0.0.0.0", TCP_AUDIO_PORT))
                self.server_socket.listen(1)
                self.server_socket.settimeout(1.0)
                
                print(f"🎧 Waiting for Android audio client...")
                
                while self.is_running:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        print(f"🎧 Android audio connected from {addr}")
                        self.client_socket = client_socket
                        self._handle_client()
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.is_running:
                            print(f"🎧 Accept error: {e}")
                        break
            
            except Exception as e:
                if self.is_running:
                    print(f"🎧 Server error: {e}")
                time.sleep(1)
            finally:
                try:
                    if self.server_socket:
                        self.server_socket.close()
                except:
                    pass

    def _handle_client(self):
        """Handle two-way audio with echo cancellation"""
        
        def audio_callback(indata, outdata, frames, time_info, status):
            """Sounddevice callback with audio processing"""
            try:
                # Process microphone input
                if self.is_running and self.client_socket:
                    # Convert float32 to int16
                    mic_data = (indata[:, 0] * 32767).astype(np.int16)
                    
                    # Apply echo cancellation using last speaker data
                    processed_mic = self.processor.process_microphone(
                        mic_data, 
                        self.last_speaker_data
                    )
                    
                    # Send to Android
                    try:
                        self.mic_queue.put_nowait(processed_mic.tobytes())
                    except:
                        pass
                
                # Process speaker output
                if not self.speaker_queue.empty():
                    try:
                        data = self.speaker_queue.get_nowait()
                        
                        # Process incoming audio for clarity
                        processed_speaker = self.processor.process_speaker(data)
                        
                        # Store for echo cancellation
                        self.last_speaker_data = processed_speaker
                        self.speaker_buffer.append(processed_speaker)
                        
                        # Convert to float32 for playback
                        audio_array = processed_speaker.astype(np.float32) / 32767.0
                        
                        # Copy to output buffer
                        samples = min(len(audio_array), frames)
                        outdata[:samples, 0] = audio_array[:samples]
                        
                        if samples < frames:
                            outdata[samples:, 0] = 0
                    except:
                        outdata.fill(0)
                else:
                    outdata.fill(0)
            
            except Exception as e:
                outdata.fill(0)
        
        try:
            # Open audio stream
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=CHUNK_SIZE
            )
            self.stream.start()
            
            mode_str = "ENHANCED" if SPEEX_AVAILABLE else "BASIC"
            features = []
            if self.processor.echo_state:
                features.append("Echo Cancel")
            if self.processor.noise_state:
                features.append("Noise Reduce")
            if self.processor.agc_state:
                features.append("AGC")
            
            print(f"🎧 Audio active! Mode: {mode_str}")
            if features:
                print(f"   Features: {', '.join(features)}")
            
            # Thread to send Windows mic to Android
            def send_audio():
                while self.is_running and self.client_socket:
                    try:
                        if not self.mic_queue.empty():
                            data = self.mic_queue.get_nowait()
                            self.client_socket.sendall(data)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    except:
                        break
                    time.sleep(0.001)
            
            # Thread to receive Android mic
            def receive_audio():
                buffer = b''
                expected_size = CHUNK_SIZE * 2  # int16 = 2 bytes per sample
                
                while self.is_running and self.client_socket:
                    try:
                        data = self.client_socket.recv(expected_size)
                        if not data:
                            break
                        
                        buffer += data
                        
                        # Process complete chunks
                        while len(buffer) >= expected_size:
                            chunk = buffer[:expected_size]
                            buffer = buffer[expected_size:]
                            
                            try:
                                self.speaker_queue.put_nowait(chunk)
                            except:
                                pass  # Queue full, drop packet
                    
                    except socket.timeout:
                        continue
                    except:
                        break
            
            send_thread = threading.Thread(target=send_audio, daemon=True)
            recv_thread = threading.Thread(target=receive_audio, daemon=True)
            
            send_thread.start()
            recv_thread.start()
            
            send_thread.join()
            recv_thread.join()
        
        except Exception as e:
            print(f"🎧 Audio error: {e}")
        finally:
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except:
                pass
            try:
                if self.client_socket:
                    self.client_socket.close()
            except:
                pass
            
            # Reset processor
            self.processor.reset()
            self.last_speaker_data = None
            self.speaker_buffer.clear()
            
            print("🎧 Audio client disconnected")

    def stop(self):
        """Stop the server"""
        self.is_running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except:
            pass
        try:
            if self.client_socket:
                self.client_socket.close()
        except:
            pass
        try:
            if self.server_socket:
                self.server_socket.close()
        except:
            pass
        print("🎧 TCP audio server stopped")

audio_server = TCPAudioServer()

# ================= WEBRTC VIDEO SERVER (UNCHANGED) =================
async def handle_client(ws, path=None):
    client_id = str(uuid.uuid4())[:8]
    print(f"📱 WebSocket connected: {client_id}")
    
    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
                msg_type = data.get("type")
                
                if msg_type == "register":
                    device_name = data.get("device", f"device_{client_id}")
                    print(f"📝 Device registered: {device_name} -> {client_id}")
                    
                    with streams_lock:
                        streams[client_id] = {
                            "active": True,
                            "pc": None,
                            "device_name": device_name,
                            "connected_at": time.time()
                        }
                    
                    frame_buffers[client_id] = VideoBuffer(client_id)
                    
                    await ws.send(json.dumps({
                        "type": "welcome",
                        "message": f"Welcome {device_name}!",
                        "client_id": client_id
                    }))
                
                elif msg_type == "offer":
                    print(f"🎬 Processing WebRTC offer from {client_id}")
                    pc = RTCPeerConnection()
                    
                    with streams_lock:
                        if client_id in streams:
                            streams[client_id]["pc"] = pc
                    
                    @pc.on("track")
                    async def on_track(track):
                        if track.kind == "video":
                            print(f"✅ Video track received from {client_id}")
                            
                            with active_tracks_lock:
                                active_tracks[client_id] = True
                            
                            frame_count = 0
                            try:
                                while True:
                                    try:
                                        frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                        frame_count += 1
                                        
                                        img = frame.to_ndarray(format="bgr24")
                                        
                                        if client_id in frame_buffers:
                                            frame_buffers[client_id].add_frame(img, time.time())
                                        
                                        frames[client_id] = img
                                        
                                        if frame_count % 100 == 0:
                                            print(f"📹 {client_id}: {frame_count} frames")
                                    
                                    except asyncio.TimeoutError:
                                        continue
                                    except Exception as e:
                                        if "track ended" in str(e).lower():
                                            break
                                        continue
                            
                            except Exception as e:
                                print(f"🎬 Video track ended: {e}")
                            finally:
                                with active_tracks_lock:
                                    if client_id in active_tracks:
                                        del active_tracks[client_id]
                    
                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    await pc.setRemoteDescription(offer)
                    
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    await ws.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))
                    
                    print(f"✅ Answer sent to {client_id}")
                
                elif msg_type == "ice_candidate":
                    try:
                        if client_id in streams and streams[client_id]["pc"]:
                            candidate = data.get("candidate")
                            sdpMid = data.get("sdpMid")
                            sdpMLineIndex = data.get("sdpMLineIndex")
                            
                            if candidate and sdpMid is not None and sdpMLineIndex is not None:
                                await streams[client_id]["pc"].addIceCandidate({
                                    "candidate": candidate,
                                    "sdpMid": sdpMid,
                                    "sdpMLineIndex": sdpMLineIndex
                                })
                    except:
                        pass
                
                elif msg_type == "heartbeat":
                    await ws.send(json.dumps({"type": "pong"}))
                
                elif msg_type == "stream_end":
                    print(f"⏹️ Stream end from {client_id}")
                    break
            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"❌ Message error: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"❌ WebSocket closed: {client_id}")
    finally:
        with streams_lock:
            if client_id in streams:
                del streams[client_id]
        if client_id in frames:
            del frames[client_id]
        if client_id in frame_buffers:
            del frame_buffers[client_id]
        with active_tracks_lock:
            if client_id in active_tracks:
                del active_tracks[client_id]

async def start_websocket_server():
    """Start WebSocket server"""
    try:
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            WS_PORT,
            ping_interval=15,
            ping_timeout=30
        ):
            print(f"🌐 WebRTC server ready: ws://{get_local_ip()}:{WS_PORT}")
            await asyncio.Future()
    except Exception as e:
        print(f"❌ WebSocket server error: {e}")

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def run_asyncio():
    """Run asyncio event loop"""
    global main_loop
    while server_running:
        try:
            main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(main_loop)
            main_loop.run_until_complete(start_websocket_server())
        except:
            time.sleep(1)
        finally:
            try:
                main_loop.close()
            except:
                pass

# ================= UI (UNCHANGED) =================
class VideoDisplay(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, bg="#000000", highlightthickness=0)
        self.photo = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        self.after(66, self.update_display)

    def update_display(self):
        """Update video display"""
        try:
            self.delete("all")
            canvas_width = max(1, self.winfo_width())
            canvas_height = max(1, self.winfo_height())
            
            current_frame = None
            if frames:
                client_id = list(frames.keys())[-1] if frames else None
                if client_id and client_id in frames:
                    current_frame = frames[client_id]
            
            if current_frame is not None:
                try:
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                    
                    frame = cv2.resize(current_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    
                    x = (canvas_width - img.width) // 2
                    y = (canvas_height - img.height) // 2
                    
                    self.photo = ImageTk.PhotoImage(img)
                    self.create_image(x, y, image=self.photo, anchor="nw")
                    
                    # Display info with audio processing status
                    audio_status = "ENHANCED" if SPEEX_AVAILABLE else "BASIC"
                    self.create_rectangle(10, 10, 200, 60, fill="#000000", outline="#333333")
                    self.create_text(15, 15, text=f"Android Video", 
                                   fill="#4CAF50", font=("Arial", 10, "bold"), anchor="nw")
                    self.create_text(15, 30, text=f"{VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {self.fps} FPS", 
                                   fill="#888888", font=("Arial", 9), anchor="nw")
                    self.create_text(15, 45, text=f"Audio: {audio_status}", 
                                   fill="#FFA726", font=("Arial", 9), anchor="nw")
                
                except Exception as e:
                    print(f"Display error: {e}")
            else:
                self.create_text(canvas_width//2, canvas_height//2,
                               text="Waiting for Android video stream...",
                               fill="#666666", font=("Arial", 14))
        
        except Exception as e:
            pass
        finally:
            self.after(66, self.update_display)

def create_ui():
    """Create main UI"""
    root = tk.Tk()
    root.title("Android Streaming Server - Enhanced Audio")
    root.geometry("800x600")
    root.configure(bg="#1a1a1a")
    
    # Header
    header = tk.Frame(root, bg="#1a1a1a")
    header.pack(fill="x", padx=20, pady=20)
    
    title = tk.Label(
        header,
        text="ANDROID STREAMING SERVER - ENHANCED",
        fg="white",
        bg="#1a1a1a",
        font=("Arial", 16, "bold")
    )
    title.pack()
    
    ip = get_local_ip()
    info = tk.Label(
        header,
        text=f"Server IP: {ip} | Audio: {TCP_AUDIO_PORT} | Video: {WS_PORT}",
        fg="#888888",
        bg="#1a1a1a",
        font=("Arial", 10)
    )
    info.pack(pady=5)
    
    # Audio features info
    features = []
    if SPEEX_AVAILABLE:
        if ENABLE_ECHO_CANCELLATION:
            features.append("Echo Cancel")
        if ENABLE_NOISE_REDUCTION:
            features.append("Noise Reduce")
        if ENABLE_AGC:
            features.append("Auto Gain")
    
    feature_text = " | ".join(features) if features else "Basic Mode"
    features_label = tk.Label(
        header,
        text=f"Audio Features: {feature_text}",
        fg="#FFA726",
        bg="#1a1a1a",
        font=("Arial", 9, "italic")
    )
    features_label.pack(pady=2)
    
    # Status
    status_frame = tk.Frame(header, bg="#1a1a1a")
    status_frame.pack(pady=10)
    
    status_led = tk.Label(status_frame, text="●", fg="red", bg="#1a1a1a", font=("Arial", 12))
    status_led.pack(side="left", padx=(0, 5))
    
    status_label = tk.Label(
        status_frame,
        text="STOPPED",
        fg="red",
        bg="#1a1a1a",
        font=("Arial", 11, "bold")
    )
    status_label.pack(side="left")
    
    # Control buttons
    btn_frame = tk.Frame(header, bg="#1a1a1a")
    btn_frame.pack(pady=10)
    
    server_state = {"running": False}
    
    def start_server():
        if server_state["running"]:
            return
        
        global server_running
        server_running = True
        server_state["running"] = True
        
        audio_server.start()
        threading.Thread(target=run_asyncio, daemon=True).start()
        
        status_led.config(fg="#4CAF50")
        status_label.config(text="RUNNING", fg="#4CAF50")
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")
        
        print("\n" + "="*60)
        print("✅ SERVER STARTED - ENHANCED AUDIO MODE")
        print("="*60)
        print(f"📱 Connect Android to: {ip}")
        print(f"🎧 TCP Audio: port {TCP_AUDIO_PORT}")
        print(f"🌐 WebRTC: ws://{ip}:{WS_PORT}")
        if features:
            print(f"🎵 Audio: {', '.join(features)}")
        print("="*60 + "\n")
    
    def stop_server():
        if not server_state["running"]:
            return
        
        global server_running
        server_running = False
        server_state["running"] = False
        
        audio_server.stop()
        frames.clear()
        frame_buffers.clear()
        
        status_led.config(fg="red")
        status_label.config(text="STOPPED", fg="red")
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")
        
        print("\n⏹️ SERVER STOPPED\n")
    
    start_btn = tk.Button(
        btn_frame,
        text="START",
        command=start_server,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 10, "bold"),
        width=10,
        relief="flat"
    )
    start_btn.pack(side="left", padx=5)
    
    stop_btn = tk.Button(
        btn_frame,
        text="STOP",
        command=stop_server,
        bg="#f44336",
        fg="white",
        font=("Arial", 10, "bold"),
        width=10,
        relief="flat",
        state="disabled"
    )
    stop_btn.pack(side="left", padx=5)
    
    # Video display
    display_frame = tk.Frame(root, bg="#000000", bd=1, relief="sunken")
    display_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    video_display = VideoDisplay(display_frame)
    video_display.pack(fill="both", expand=True)
    
    # Footer
    footer_text = "Echo Cancellation + Noise Reduction + AGC" if SPEEX_AVAILABLE else "Basic Audio Mode"
    footer = tk.Label(
        root,
        text=f"TCP Audio + WebRTC Video | 15 FPS | {footer_text}",
        fg="#666666",
        bg="#1a1a1a",
        font=("Arial", 9)
    )
    footer.pack(pady=(0, 10))
    
    def on_closing():
        if server_state["running"]:
            stop_server()
        root.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    return root

# ================= MAIN =================
if __name__ == "__main__":
    print("="*60)
    print("ANDROID STREAMING SERVER - ENHANCED AUDIO")
    print("="*60)
    
    if SPEEX_AVAILABLE:
        print("✅ Echo Cancellation - ENABLED")
        print("✅ Noise Reduction - ENABLED")
        print("✅ Automatic Gain Control - ENABLED")
        print(f"✅ Echo Tail: {ECHO_TAIL_MS}ms")
    else:
        print("⚠️  Install speexdsp-python for enhanced audio:")
        print("   pip install speexdsp-python")
        print("📊 Running with basic audio filters")
    
    print("✅ WebRTC Video @ 15 FPS")
    print("="*60)
    
    try:
        root = create_ui()
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        audio_server.stop()
        print("✅ Server shutdown")
