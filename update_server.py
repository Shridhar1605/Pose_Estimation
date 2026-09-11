import re

with open('server.py', 'r', encoding='utf-8') as f:
    code = f.read()

start_idx = code.find('    def __init__(self, stream_id:')
end_idx = code.find('    def _send_frame(self, data: dict):')

if start_idx == -1 or end_idx == -1:
    print(f"Could not find boundaries: start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

new_methods = """    def __init__(self, stream_id: str, video_path: str, model_name: str, frame_callback, target_fps=15):
        self.stream_id = stream_id
        self.video_path = video_path
        self.model_name = model_name
        self.frame_callback = frame_callback
        self.target_fps = target_fps

        self._running = False
        self._reader_thread = None
        self._processor_thread = None
        self._loop = None

        self.current_fps = 0.0
        self.person_count = 0
        self.frame_number = 0
        self.total_frames = 0
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self, loop):
        if self._running:
            return
        self._running = True
        self._loop = loop
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True, name=f"read-{self.stream_id}")
        self._processor_thread = threading.Thread(target=self._process_frames, daemon=True, name=f"proc-{self.stream_id}")
        self._reader_thread.start()
        self._processor_thread.start()

    def stop(self):
        self._running = False
        self.new_frame_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None
        if self._processor_thread:
            self._processor_thread.join(timeout=2)
            self._processor_thread = None

    def _read_frames(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._send_error(f"Cannot open video: {self.video_path}")
                return

            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = 1.0 / native_fps

            print(f"[Stream {self.stream_id}] Reader started — {self.video_path} @ {native_fps:.1f} FPS")

            while self._running:
                loop_start = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    if self.video_path.startswith(("rtsp://", "http://", "https://")):
                        self._send_error("Stream ended or disconnected.")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                h, w = frame.shape[:2]
                if h > 720 or w > 1280:
                    scale = min(1280.0 / w, 720.0 / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

                if not self.video_path.startswith(("rtsp://", "http://", "https://")):
                    total_elapsed = time.time() - loop_start
                    sleep_time = frame_interval - total_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            cap.release()
            print(f"[Stream {self.stream_id}] Reader stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

    def _process_frames(self):
        import base64
        try:
            tracker = OCSortTracker(iou_threshold=0.25, max_lost=60, min_confidence=0.25)
            track_history = {}
            track_states = {}
            tracked = []

            fps_window = []
            target_interval = 1.0 / self.target_fps
            print(f"[Stream {self.stream_id}] Processor started")

            while self._running:
                loop_start = time.time()
                
                if not self.new_frame_event.wait(timeout=1.0):
                    continue
                
                with self.frame_lock:
                    frame = self.latest_frame
                self.new_frame_event.clear()
                
                if frame is None:
                    continue

                process_start = time.time()
                
                dets = detect_persons(frame, self.model_name)
                tracked = tracker.update(dets.tolist() if len(dets) > 0 else [], frame)

                active_ids = set()
                rtmpose_count = 0 
                
                for track in tracked:
                    track_id = track["id"]
                    active_ids.add(track_id)
                    bbox = track["bbox"]

                    if track_id not in track_history:
                        track_history[track_id] = []
                    if track_id not in track_states:
                        track_states[track_id] = {"action": "STANDING", "conf": 0.5, "last_rtm_frame": 0}

                    state = track_states[track_id]
                    if self.frame_number - state.get("last_rtm_frame", 0) >= 3 and rtmpose_count < 2:
                        roi, rx1, ry1 = crop_with_padding(frame, bbox, pad=20)
                        if roi.size > 0:
                            keypoints, scores = rtmpose.infer(roi)
                            rtmpose_count += 1
                            bbox_height = bbox[3] - bbox[1]
                            track_history[track_id].append({
                                "frame": self.frame_number,
                                "kps": keypoints,
                                "rx1": rx1,
                                "ry1": ry1,
                                "conf": scores,
                            })
                            if len(track_history[track_id]) > 30:
                                track_history[track_id].pop(0)
                            action, action_conf = classify_action(keypoints, track_history[track_id], bbox_height)
                            state["action"] = action
                            state["conf"] = action_conf
                            state["last_rtm_frame"] = self.frame_number

                stale = [tid for tid in track_history if tid not in active_ids]
                for tid in stale:
                    if len(track_history.get(tid, [])) > 0:
                        last_frame = track_history[tid][-1].get("frame", 0)
                        if self.frame_number - last_frame > 120:
                            del track_history[tid]
                            track_states.pop(tid, None)
                    else:
                        del track_history[tid]
                        track_states.pop(tid, None)

                marked = frame.copy()
                draw_annotations(marked, tracked, track_states, track_history)

                model_label = f"Model: {self.model_name.upper()}"
                cv2.putText(marked, model_label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                _, buffer = cv2.imencode('.jpg', marked, encode_params)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                tracks_info = []
                for track in tracked:
                    tid = track["id"]
                    state = track_states.get(tid, {"action": "STANDING", "conf": 0.5})
                    tracks_info.append({
                        "id": tid,
                        "bbox": track["bbox"],
                        "action": state["action"],
                        "conf": round(state["conf"], 2),
                    })

                fps_window.append(time.time())
                if len(fps_window) > 30:
                    fps_window.pop(0)
                if len(fps_window) >= 2:
                    elapsed = fps_window[-1] - fps_window[0]
                    self.current_fps = (len(fps_window) - 1) / elapsed if elapsed > 0 else 0
                else:
                    self.current_fps = 0

                self.person_count = len(tracked)
                self.frame_number += 1

                frame_data = {
                    "type": "frame",
                    "stream_id": self.stream_id,
                    "frame": frame_b64,
                    "fps": round(self.current_fps, 1),
                    "person_count": self.person_count,
                    "tracks": tracks_info,
                    "frame_number": self.frame_number,
                    "total_frames": self.total_frames,
                    "model": self.model_name,
                    "infer_every_n": 1, 
                }
                self._send_frame(frame_data)

                process_elapsed = time.time() - loop_start
                sleep_time = target_interval - process_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"[Stream {self.stream_id}] Processor stopped")
        except Exception as e:
            traceback.print_exc()
            self._send_error(str(e))

"""

new_code = code[:start_idx] + new_methods + code[end_idx:]

with open('server.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
print("Updated server.py successfully.")
