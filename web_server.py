"""Local Flask server exposing the live camera feed + guidance status to a browser (Section 11).

Runs in a background thread inside inference_loop.py's own process, so it
shares in-memory state directly with the capture/inference loop - no second
process, no polling a status file on disk, no risk of the page reading a
half-written frame mid-save.
"""
import json
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, send_from_directory
import numpy as np

import config


class LiveState:
    """Thread-safe holder for the latest frame + guidance snapshot.

    Written once per loop iteration by inference_loop.py's main thread, read
    on demand by Flask's request-handling threads - the lock only ever
    guards a quick reference swap, never any per-pixel work.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._reset_requested = False

        # Load states from states.json if available
        states = []
        try:
            states_file = Path(config.STATES_PATH)
            if states_file.exists():
                with open(states_file, "r", encoding="utf-8") as f:
                    states = json.load(f)
        except Exception:
            pass

        first_step = states[0] if states else None
        self._status = {
            "message": first_step["label"] if first_step else "Starting camera...",
            "instructions": first_step.get("instructions") if first_step else None,
            "mismatch": False,
            "model": first_step.get("model") if first_step else None,
            "next_step": first_step,
            "completed_step": None,
            "confirmed_tags": [],
            "tag_probs": {},
            "tag_labels": config.TAG_LABELS,
            "states": states,
            "product_name": config.PRODUCT_NAME,
            "session_started_at": time.time(),
            "total_frames": 0,
            "mismatch_frames": 0,
        }
        self._initial_status = dict(self._status)

        # Generate an initial placeholder frame so /video_feed is ready
        try:
            placeholder = np.zeros((config.COLOR_HEIGHT, config.COLOR_WIDTH, 3), dtype=np.uint8)
            cv2.putText(placeholder, "LEGO Assembly Guidance", (220, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            cv2.putText(placeholder, "Live camera stream standby", (270, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
            ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, config.WEB_JPEG_QUALITY])
            if ok:
                self._jpeg = buf.tobytes()
        except Exception:
            pass

    def set_frame(self, bgr_image):
        ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, config.WEB_JPEG_QUALITY])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def set_status(self, status: dict):
        with self._lock:
            self._status = status

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def get_status(self):
        with self._lock:
            return dict(self._status)

    def request_reset(self):
        with self._lock:
            self._reset_requested = True
            if hasattr(self, "_initial_status") and self._initial_status:
                self._status = dict(self._initial_status)
                self._status["session_started_at"] = time.time()

    def consume_reset_request(self):
        """True at most once per request - the loop calls this every frame,
        so a plain flag would fire the reset repeatedly until it's cleared."""
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
            return requested


# This app changes shape often during development (screens, styling, JS
# logic), and viewers reload it from a bunch of different devices - without
# this, a browser that already has an old index.html/app.js/style.css cached
# can keep showing a stale layout indefinitely, which looks exactly like a
# real bug (e.g. an old build that skipped straight to the video/model view,
# before the home screen existed). Model files are left to cache normally -
# they're large, don't change per-edit, and re-fetching them every reload
# would be wasteful.
NO_CACHE_EXTENSIONS = {".html", ".js", ".css"}


def build_app(live_state: LiveState, web_dir: Path):
    app = Flask(__name__, static_folder=None)

    def _no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/")
    def index():
        return _no_cache(send_from_directory(web_dir, "index.html"))

    @app.route("/api/status")
    def status():
        # Long-lived guidance state, not sensor data - fine to let the
        # browser's own HTTP cache alone; the client polls anyway.
        return jsonify(live_state.get_status())

    @app.route("/api/reset", methods=["POST"])
    def reset():
        # Sets a flag inference_loop.py's main thread checks each frame -
        # matcher.reset() itself isn't thread-safe to call from here directly.
        live_state.request_reset()
        return jsonify({"ok": True})

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                frame = live_state.get_frame()
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(1 / 30)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/<path:filename>")
    def static_files(filename):
        # Covers style.css, app.js, and assets/models/*.glb - anything else
        # dropped under web/ is servable the same way without a new route.
        response = send_from_directory(web_dir, filename)
        if Path(filename).suffix.lower() in NO_CACHE_EXTENSIONS:
            response = _no_cache(response)
        return response

    return app


def start_server_thread(live_state: LiveState, web_dir=None, host=None, port=None):
    web_dir = Path(web_dir or config.WEB_DIR)
    app = build_app(live_state, web_dir)
    thread = threading.Thread(
        target=lambda: app.run(host=host or config.WEB_HOST, port=port or config.WEB_PORT,
                                threaded=True, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Lego guidance web server")
    parser.add_argument("--host", default=config.WEB_HOST, help="Host to bind to (default: %(default)s)")
    parser.add_argument("--port", type=int, default=config.WEB_PORT, help="Port to listen on (default: %(default)s)")
    parser.add_argument("--web-dir", default=config.WEB_DIR, help="Static web directory (default: %(default)s)")
    args = parser.parse_args()

    state = LiveState()
    print(f"Starting LEGO guidance web server on http://localhost:{args.port} (host: {args.host})")
    app = build_app(state, Path(args.web_dir))
    app.run(host=args.host, port=args.port, debug=False)

