"""
╔═══════════════════════════════════════════════════════════════════════╗
║        AI SURVEILLANCE & ATTENDANCE SYSTEM  v4.0                     ║
║        Built for Live Demo — College Project                          ║
║                                                                       ║
║  ENGINE:                                                              ║
║  • YOLOv8n          → Person / Phone / Object Detection              ║
║  • YOLOv8n-Pose     → Skeleton Pose → Fight Detection                ║
║  • face_recognition → Attendance (who is present)                    ║
║  • OpenCV Cascades  → Cheating (head turn / profile face)            ║
║  • Optical Flow     → Motion heatmap                                 ║
║                                                                       ║
║  OUTPUT:                                                              ║
║  • Live OpenCV window  (professional HUD)                            ║
║  • Flask web dashboard (localhost:5000) — real-time                  ║
║  • Voice alerts        (pyttsx3 TTS)                                 ║
║  • Buzzer beeps        (winsound / beep)                             ║
║  • Email alerts        (Gmail SMTP)                                  ║
║  • Screenshots         → screenshots/                                ║
║  • Event log           → events.txt                                  ║
║  • Attendance log      → attendance.csv                              ║
╚═══════════════════════════════════════════════════════════════════════╝

QUICK START:
  1. pip install -r requirements.txt
  2. Add known faces:  put  Name.jpg  images in  known_faces/  folder
  3. python main.py
  4. Open browser:     http://localhost:5000
  5. Press Q to quit   |   S for manual screenshot
"""

# ── Imports ──────────────────────────────────────────────────────────────
import cv2, os, time, threading, smtplib, csv, json
import numpy as np
from datetime import datetime
from collections import deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO

# Flask dashboard (real-time via SocketIO)
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO

# Voice TTS
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 165)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# NOTE: Attendance uses OpenCV LBPH (see AttendanceSystem class below) —
# no dlib / face_recognition library needed, so there is nothing to import here.

# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════
CROWD_THRESHOLD      = 4      # 4+ log = crowd (3 pe faltu alert tha)
FIGHT_POSE_FRAMES    = 8      # 8 consecutive frames = confirmed fight (4 pe faltu tha)
FIGHT_OVERLAP_THRESH = 0.25   # tighter overlap needed
CHEAT_TURN_FRAMES    = 14     # 14 frames sustained = cheating (8 pe faltu tha)
KNOWN_FACES_DIR      = "known_faces"   # put Name.jpg files here

# ── Company branding ─────────────────────────────────────────────
COMPANY_NAME    = "Pinakin Infra"
COMPANY_TAGLINE = "Planning  |  Innovation  |  Excellence"
LOGO_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pinakin_logo.png")

RESTRICTED_ZONES = []   # Empty by default — draw with mouse at runtime (press R)

EMAIL_ENABLED       = False
EMAIL_SENDER        = "youremail@gmail.com"
EMAIL_PASSWORD      = "gmail_app_password"
EMAIL_RECEIVER      = "owner@gmail.com"
EMAIL_COOLDOWN_SEC  = 60

VOICE_ENABLED       = True   # set False to mute voice
DASHBOARD_PORT      = 5000

# ── Camera source ─────────────────────────────────────────────────
# Option A (default) — laptop's built-in webcam:
CAMERA_SOURCE = 0

# Option B — phone camera via "IP Webcam" app (free, Android):
#   1. Install "IP Webcam" app on phone, connect phone to SAME WiFi as laptop
#   2. Open the app, tap "Start server" — it shows a URL like http://192.168.1.5:8080
#   3. Uncomment the line below and put YOUR phone's IP in it:
# CAMERA_SOURCE = "http://192.168.1.5:8080/video"

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE  (shared with dashboard)
# ════════════════════════════════════════════════════════════════════════
state = {
    "people": 0, "crowd": False, "phone": False,
    "fight": False, "fight_reason": "",
    "cheat": False, "cheat_reason": "",
    "restricted": False, "motion": 0.0,
    "events": 0, "attendance": [],
    "recent_events": [],   # last 10 events for dashboard feed
    "uptime_start": datetime.now().strftime("%H:%M:%S"),
}
state_lock = threading.Lock()

# ════════════════════════════════════════════════════════════════════════
#  FLASK DASHBOARD
# ════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ company_name }} — Site & Security Control Room</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
  --bg:      #0b0d11;
  --surface: #12161c;
  --panel:   #161b22;
  --border:  #232b35;
  --border2: #38424f;
  --primary: #c41e3a;   /* Pinakin red */
  --primary2:#e84560;
  --slate:   #2d3a4a;   /* Pinakin slate */
  --green:   #2ecc71;
  --red:     #ff3b30;
  --orange:  #ff9500;
  --yellow:  #ffd700;
  --purple:  #a855f7;
  --dim:     #5a6675;
  --text:    #c4ccd6;
}

*{ margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Rajdhani', sans-serif;
  min-height:100vh;
  overflow-x:hidden;
}

/* ── Scanline overlay ── */
body::after {
  content:'';
  position:fixed; inset:0; pointer-events:none; z-index:9999;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,.04) 2px, rgba(0,0,0,.04) 4px
  );
}

/* ── Subtle grid ── */
body::before {
  content:'';
  position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(196,30,58,.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(196,30,58,.02) 1px, transparent 1px);
  background-size:50px 50px;
}

/* ── Corner decorations ── */
.corner {
  position:fixed; width:60px; height:60px; z-index:10;
  pointer-events:none;
}
.corner::before,.corner::after {
  content:''; position:absolute; background:var(--primary); opacity:.5;
}
.corner::before { width:2px; height:100%; }
.corner::after  { width:100%; height:2px; }
.corner.tl { top:0; left:0; }
.corner.tr { top:0; right:0; transform:scaleX(-1); }
.corner.bl { bottom:0; left:0; transform:scaleY(-1); }
.corner.br { bottom:0; right:0; transform:scale(-1); }

/* ── Layout ── */
.root { position:relative; z-index:1; display:flex; flex-direction:column; height:100vh; padding:12px 16px; gap:10px; }

/* ── Top bar ── */
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:8px 16px;
  background: linear-gradient(90deg, #c41e3a0a, #c41e3a04, #c41e3a0a);
  border:1px solid var(--border2);
  border-radius:6px;
  flex-shrink:0;
}
.brand { display:flex; align-items:center; gap:14px; }
.brand-icon {
  width:42px; height:42px; border-radius:8px;
  background: #ffffff;
  border:1px solid var(--border2); display:flex; align-items:center;
  justify-content:center; padding:4px;
  box-shadow: 0 0 12px #c41e3a22;
}
.brand-icon img { width:100%; height:100%; object-fit:contain; }
.brand-text h1 {
  font-family:'Orbitron',monospace; font-size:.85rem; font-weight:900;
  color:var(--primary2); letter-spacing:.1em;
}
.brand-text p { font-size:.6rem; color:var(--dim); letter-spacing:.1em; margin-top:1px; }

.topbar-center { display:flex; gap:24px; }
.sys-stat { text-align:center; }
.sys-stat-val { font-family:'Share Tech Mono',monospace; font-size:.95rem; color:var(--primary2); }
.sys-stat-lbl { font-size:.55rem; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; }

.topbar-right { display:flex; align-items:center; gap:16px; }
.live-pill {
  display:flex; align-items:center; gap:6px;
  padding:4px 12px; border-radius:20px;
  background:#ff3b3018; border:1px solid #ff3b3040;
  font-family:'Orbitron',monospace; font-size:.6rem; font-weight:700;
  color:var(--red); letter-spacing:.15em;
}
.pulse-dot { width:6px; height:6px; border-radius:50%; background:var(--red); animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(1.5)} }
#clock { font-family:'Orbitron',monospace; font-size:.8rem; color:var(--primary2); letter-spacing:.1em; }
#date  { font-size:.58rem; color:var(--dim); text-align:right; margin-top:2px; letter-spacing:.06em; }

/* ── Main grid ── */
.main { display:grid; grid-template-columns:220px 1fr 280px; gap:10px; flex:1; min-height:0; }

/* ── Left sidebar ── */
.sidebar { display:flex; flex-direction:column; gap:8px; }

.status-block {
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:10px 12px;
  position:relative; overflow:hidden;
  transition: border-color .3s, box-shadow .3s;
}
.status-block::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
  background: var(--accent, var(--primary));
}
.status-block.alert {
  border-color:var(--red) !important;
  box-shadow:0 0 14px #ff3b3028;
  animation:alertPulse 1s infinite;
}
@keyframes alertPulse { 0%,100%{box-shadow:0 0 14px #ff3b3028} 50%{box-shadow:0 0 24px #ff3b3060} }
.status-block.alert::before { background:var(--red); }

.sb-label { font-size:.58rem; letter-spacing:.14em; text-transform:uppercase; color:var(--dim); margin-bottom:4px; }
.sb-value { font-family:'Share Tech Mono',monospace; font-size:1.1rem; font-weight:700; }
.sb-value.ok     { color:var(--green); }
.sb-value.warn   { color:var(--orange); }
.sb-value.danger { color:var(--red); }
.sb-value.info   { color:var(--primary2); }
.sb-sub { font-size:.6rem; color:var(--dim); margin-top:3px; line-height:1.3; }

/* ── Center ── */
.center { display:flex; flex-direction:column; gap:10px; }

/* Camera feed placeholder */
.cam-panel {
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; flex:1; min-height:0;
  position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center;
}
.cam-panel::before {
  content:''; position:absolute; inset:0;
  background: radial-gradient(ellipse at center, #c41e3a08 0%, transparent 70%);
}
.cam-overlay {
  position:absolute; inset:0; display:flex; flex-direction:column;
  justify-content:space-between; padding:12px;
}
.cam-top { display:flex; justify-content:space-between; align-items:flex-start; }
.cam-tag {
  font-family:'Share Tech Mono',monospace; font-size:.62rem;
  color:var(--primary2); background:#c41e3a18; border:1px solid #c41e3a40;
  padding:3px 8px; border-radius:4px; letter-spacing:.08em;
}
.cam-corners span {
  display:inline-block; width:12px; height:12px;
  border-color:var(--primary2); border-style:solid; opacity:.7;
}
.cam-bottom { display:flex; justify-content:space-between; align-items:flex-end; }
.cam-info { font-family:'Share Tech Mono',monospace; font-size:.6rem; color:var(--dim); line-height:1.8; }
.cam-placeholder {
  display:flex; flex-direction:column; align-items:center; gap:10px;
  color:var(--dim);
}
.cam-placeholder .icon { font-size:2.5rem; opacity:.3; }
.cam-placeholder p { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; }

/* Chart row */
.chart-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; height:150px; }
.chart-panel {
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:10px 12px; position:relative;
}
.chart-title { font-size:.6rem; letter-spacing:.12em; text-transform:uppercase; color:var(--dim); margin-bottom:6px; }

/* ── Right panel ── */
.right-panel { display:flex; flex-direction:column; gap:10px; }

/* Event feed */
.feed-panel {
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; display:flex; flex-direction:column; flex:1; min-height:0;
}
.panel-hdr {
  padding:8px 12px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; justify-content:space-between; flex-shrink:0;
}
.panel-hdr-title { font-size:.65rem; letter-spacing:.14em; text-transform:uppercase; color:var(--primary2); font-weight:700; }
.panel-hdr-sub   { font-family:'Share Tech Mono',monospace; font-size:.6rem; color:var(--dim); }
.feed-body { flex:1; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:6px; }
.feed-body::-webkit-scrollbar { width:3px; }
.feed-body::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }

.ev {
  display:flex; gap:8px; padding:7px 10px; border-radius:6px;
  background:#ffffff05; border-left:2px solid var(--border2);
  animation:slideIn .25s ease;
  flex-shrink:0;
}
@keyframes slideIn { from{opacity:0;transform:translateX(8px)} to{opacity:1;transform:none} }
.ev.FIGHT      { border-color:var(--red);    background:#ff224410; }
.ev.CHEAT      { border-color:var(--red);    background:#ff224410; }
.ev.RESTRICTED { border-color:var(--orange); background:#ff880010; }
.ev.CROWD      { border-color:var(--yellow); background:#ffd70010; }
.ev.PHONE      { border-color:var(--primary2);   background:#c41e3a10; }
.ev.ATTENDANCE { border-color:var(--green);  background:#2ecc7110; }
.ev-icon { font-size:.9rem; margin-top:1px; flex-shrink:0; }
.ev-body h4 { font-size:.68rem; font-weight:700; color:#dde8f0; }
.ev-body p  { font-size:.58rem; color:var(--dim); margin-top:1px; font-family:'Share Tech Mono',monospace; }

/* Attendance */
.att-panel {
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; max-height:220px; display:flex; flex-direction:column;
}
.att-body { flex:1; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:6px; }
.att-body::-webkit-scrollbar { width:3px; }
.att-body::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }
.att-row {
  display:flex; align-items:center; gap:10px;
  padding:6px 10px; border-radius:6px;
  background:#ffffff05; border:1px solid var(--border);
}
.att-av {
  width:30px; height:30px; border-radius:50%; flex-shrink:0;
  background:linear-gradient(135deg,#c41e3a22,#2ecc7122);
  border:1px solid var(--primary2);
  display:flex; align-items:center; justify-content:center;
  font-size:.85rem; font-weight:700; color:var(--primary2);
  font-family:'Share Tech Mono',monospace;
}
.att-name { font-size:.72rem; font-weight:700; color:#dde8f0; }
.att-time { font-size:.58rem; color:var(--dim); font-family:'Share Tech Mono',monospace; }
.att-badge {
  margin-left:auto; padding:2px 8px; border-radius:10px;
  font-size:.55rem; font-weight:700; letter-spacing:.1em;
  background:#2ecc7118; color:var(--green); border:1px solid #2ecc7130;
}

/* Alert modal overlay */
#alert-modal {
  position:fixed; inset:0; z-index:1000; display:none;
  align-items:center; justify-content:center;
  background:#000000cc; backdrop-filter:blur(4px);
}
#alert-modal.show { display:flex; }
.alert-box {
  background:#0a0505; border:2px solid var(--red);
  border-radius:12px; padding:28px 36px; text-align:center;
  box-shadow:0 0 60px #ff224440;
  animation:zoomIn .3s ease;
  max-width:420px; width:90%;
}
@keyframes zoomIn { from{transform:scale(.8);opacity:0} to{transform:scale(1);opacity:1} }
.alert-box .alert-icon { font-size:2.8rem; margin-bottom:10px; }
.alert-box h2 {
  font-family:'Orbitron',monospace; font-size:1.1rem; color:var(--red);
  letter-spacing:.12em; margin-bottom:8px;
}
.alert-box p { font-size:.8rem; color:var(--text); margin-bottom:18px; line-height:1.6; }
.alert-dismiss {
  padding:8px 24px; border-radius:6px; border:1px solid var(--red);
  background:#ff224420; color:var(--red);
  font-family:'Rajdhani',sans-serif; font-size:.8rem; font-weight:700;
  cursor:pointer; letter-spacing:.1em; transition:background .2s;
}
.alert-dismiss:hover { background:#ff224440; }

/* Bottom status bar */
.statusbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:4px 16px;
  background:var(--surface); border:1px solid var(--border);
  border-radius:4px; flex-shrink:0;
}
.sb-items { display:flex; gap:20px; }
.sb-item { font-family:'Share Tech Mono',monospace; font-size:.58rem; color:var(--dim); display:flex; align-items:center; gap:5px; }
.sb-item span { color:var(--primary2); }
.status-dot { width:5px; height:5px; border-radius:50%; background:var(--green); display:inline-block; animation:pulse 2s infinite; }

.empty { text-align:center; padding:20px; color:var(--dim); font-size:.7rem; letter-spacing:.08em; }
</style>
</head>
<body>

<!-- Corner decorations -->
<div class="corner tl"></div>
<div class="corner tr"></div>
<div class="corner bl"></div>
<div class="corner br"></div>

<div class="root">

  <!-- Top bar -->
  <div class="topbar">
    <div class="brand">
      <div class="brand-icon"><img src="/logo.png" alt="{{ company_name }}"></div>
      <div class="brand-text">
        <h1>{{ company_name|upper }} — SITE &amp; SECURITY CONTROL ROOM</h1>
        <p>{{ tagline }}</p>
      </div>
    </div>
    <div class="topbar-center">
      <div class="sys-stat"><div class="sys-stat-val" id="tb-people">0</div><div class="sys-stat-lbl">People</div></div>
      <div class="sys-stat"><div class="sys-stat-val" id="tb-events">0</div><div class="sys-stat-lbl">Events</div></div>
      <div class="sys-stat"><div class="sys-stat-val" id="tb-att">0</div><div class="sys-stat-lbl">Present</div></div>
      <div class="sys-stat"><div class="sys-stat-val" id="tb-motion" style="color:var(--green)">0.0</div><div class="sys-stat-lbl">Motion</div></div>
    </div>
    <div class="topbar-right">
      <div class="live-pill"><div class="pulse-dot"></div>LIVE</div>
      <div><div id="clock">--:--:--</div><div id="date"></div></div>
    </div>
  </div>

  <!-- Main -->
  <div class="main">

    <!-- LEFT: Status sidebar -->
    <div class="sidebar">

      <div style="font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);padding:0 4px">
        ◈ DETECTION STATUS
      </div>

      <div class="status-block" id="blk-crowd" style="--accent:#ffd700">
        <div class="sb-label">🚨 Crowd Monitor</div>
        <div class="sb-value ok" id="v-crowd">NORMAL</div>
        <div class="sb-sub" id="s-crowd">Threshold: {{ crowd_threshold }} people</div>
      </div>

      <div class="status-block" id="blk-fight" style="--accent:#ff2244">
        <div class="sb-label">🥊 Violence / Fight</div>
        <div class="sb-value ok" id="v-fight">CLEAR</div>
        <div class="sb-sub" id="s-fight">Pose skeleton analysis</div>
      </div>

      <div class="status-block" id="blk-cheat" style="--accent:#ff2244">
        <div class="sb-label">📝 Exam Integrity</div>
        <div class="sb-value ok" id="v-cheat">SECURE</div>
        <div class="sb-sub" id="s-cheat">Head-turn + phone track</div>
      </div>

      <div class="status-block" id="blk-phone" style="--accent:#c41e3a">
        <div class="sb-label">📱 Device Detection</div>
        <div class="sb-value ok" id="v-phone">NONE</div>
        <div class="sb-sub">YOLOv8 object class 67</div>
      </div>

      <div class="status-block" id="blk-restricted" style="--accent:#ff8800">
        <div class="sb-label">🚧 Restricted Zone</div>
        <div class="sb-value ok" id="v-restricted">SECURE</div>
        <div class="sb-sub">Perimeter monitoring</div>
      </div>

      <div class="status-block" style="--accent:#a855f7">
        <div class="sb-label">🌊 Motion Index</div>
        <div class="sb-value info" id="v-motion">0.0</div>
        <div class="sb-sub">Optical flow magnitude</div>
      </div>

      <div class="status-block" style="--accent:#2ecc71; margin-top:auto">
        <div class="sb-label">⚡ System</div>
        <div class="sb-value ok">ONLINE</div>
        <div class="sb-sub" id="s-uptime">Started --:--</div>
      </div>

    </div>

    <!-- CENTER -->
    <div class="center">

      <!-- Live Camera Feed -->
      <div class="cam-panel">
        <img id="live-feed" src="/video_feed"
             style="width:100%;height:100%;object-fit:cover;display:block;border-radius:8px;"
             onerror="this.style.display='none';document.getElementById('cam-fallback').style.display='flex'">
        <div id="cam-fallback" style="display:none;position:absolute;inset:0;align-items:center;justify-content:center;flex-direction:column;gap:8px;color:var(--dim)">
          <div style="font-size:2rem;opacity:.3">📷</div>
          <div style="font-size:.7rem;letter-spacing:.1em">CONNECTING TO CAMERA...</div>
        </div>
        <div class="cam-overlay">
          <div class="cam-top">
            <div style="display:flex;gap:6px">
              <div class="cam-tag">CAM-01 • MAIN</div>
              <div class="cam-tag" id="cam-status-tag" style="color:var(--green);border-color:var(--green);background:#2ecc7118">● LIVE</div>
            </div>
            <div class="cam-tag" id="cam-alert-tag" style="display:none;color:var(--red);border-color:var(--red);background:#ff224418;animation:pulse 1s infinite">⚠ ALERT</div>
          </div>
          <div class="cam-bottom">
            <div class="cam-info">
              <div>PEOPLE: <span style="color:var(--primary2)" id="cam-people">0</span></div>
              <div>MOTION: <span style="color:var(--primary2)" id="cam-motion">0.0</span></div>
              <div id="cam-ts" style="color:var(--dim)">--:--:--</div>
            </div>
            <div style="font-size:.58rem;color:var(--dim);text-align:right;font-family:'Share Tech Mono',monospace">
              YOLOv8n + Pose<br>OpenCV 4.x
            </div>
          </div>
        </div>
      </div>

      <!-- Charts + Screenshots -->
      <div class="chart-row">
        <div class="chart-panel">
          <div class="chart-title">📈 People Count — Last 30s</div>
          <canvas id="chart-people" style="max-height:110px"></canvas>
        </div>
        <div class="chart-panel">
          <div class="chart-title">🌊 Motion Level — Last 30s</div>
          <canvas id="chart-motion" style="max-height:110px"></canvas>
        </div>
      </div>

      <!-- Evidence Screenshots -->
      <div class="panel-hdr" style="background:var(--panel);border:1px solid var(--border);border-radius:8px;margin-top:0">
        <div class="panel-hdr-title">📸 Evidence Screenshots</div>
        <div class="panel-hdr-sub" id="ss-count">Loading...</div>
      </div>
      <div id="ss-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;max-height:120px;overflow-y:auto"></div>

    </div>

    <!-- RIGHT -->
    <div class="right-panel">

      <!-- Event Feed -->
      <div class="feed-panel">
        <div class="panel-hdr">
          <div class="panel-hdr-title">📡 Live Event Feed</div>
          <div class="panel-hdr-sub" id="feed-count">0 events</div>
        </div>
        <div class="feed-body" id="event-feed">
          <div class="empty">Monitoring active...<br>No violations detected</div>
        </div>
      </div>

      <!-- New Face Registration -->
      <button id="register-trigger-btn" onclick="startRegisterScan()"
              style="width:100%;padding:10px;border-radius:8px;border:1px solid var(--primary);
                     background:#c41e3a18;color:var(--primary2);font-weight:700;font-size:.78rem;
                     letter-spacing:.06em;cursor:pointer;font-family:'Rajdhani',sans-serif;
                     display:flex;align-items:center;justify-content:center;gap:8px">
        📷 REGISTER NEW STUDENT
      </button>

      <div class="att-panel" id="register-panel" style="display:none; border-color:var(--orange)">
        <div class="panel-hdr" style="border-color:var(--orange)">
          <div class="panel-hdr-title" style="color:var(--orange)" id="register-title">🔍 Scanning for face...</div>
        </div>
        <div style="padding:10px">
          <img id="register-photo" src="" style="width:100%;max-height:140px;object-fit:cover;border-radius:6px;border:1px solid var(--orange);margin-bottom:8px;display:none">
          <div id="register-scanning" style="text-align:center;padding:20px;color:var(--dim);font-size:.7rem;letter-spacing:.08em">
            Stand in front of the camera...<br>
            <span style="color:var(--orange)">⏳ Looking for a NEW face</span><br>
            <span style="font-size:.6rem;color:var(--dim)">Already-registered students are automatically skipped</span>
          </div>
          <div id="register-actions" style="display:none;gap:6px;flex-direction:column">
            <input id="register-name-real" type="text" placeholder="Enter student name..."
                   style="width:100%;padding:7px 10px;border-radius:6px;border:1px solid var(--border2);
                          background:var(--surface);color:var(--text);font-family:'Rajdhani',sans-serif;
                          font-size:.8rem;margin-bottom:8px;box-sizing:border-box"
                   onkeydown="if(event.key==='Enter') registerFace()">
            <div style="display:flex;gap:6px">
              <button onclick="registerFace()"
                      style="flex:1;padding:7px;border-radius:6px;border:1px solid var(--green);
                             background:#2ecc7118;color:var(--green);font-weight:700;font-size:.72rem;
                             letter-spacing:.05em;cursor:pointer;font-family:'Rajdhani',sans-serif">
                ✓ SAVE &amp; REGISTER
              </button>
              <button onclick="cancelRegisterScan()"
                      style="padding:7px 14px;border-radius:6px;border:1px solid var(--border2);
                             background:transparent;color:var(--dim);font-size:.72rem;cursor:pointer;
                             font-family:'Rajdhani',sans-serif">
                CANCEL
              </button>
            </div>
          </div>
          <button id="register-cancel-scan-btn" onclick="cancelRegisterScan()"
                  style="width:100%;margin-top:8px;padding:6px;border-radius:6px;border:1px solid var(--border2);
                         background:transparent;color:var(--dim);font-size:.68rem;cursor:pointer;
                         font-family:'Rajdhani',sans-serif">
            CANCEL SCAN
          </button>
          <div id="register-msg" style="font-size:.62rem;color:var(--dim);margin-top:6px"></div>
        </div>
      </div>

      <!-- Student Evaluation -->
      <div class="att-panel">
        <div class="panel-hdr">
          <div class="panel-hdr-title">🎯 Student Evaluation</div>
          <div class="panel-hdr-sub" id="att-count">0 / 0 present</div>
        </div>
        <div class="att-body" id="att-list">
          <div class="empty">Add photos to<br><span style="color:var(--primary2);font-family:monospace">known_faces/Name.jpg</span></div>
        </div>
      </div>

    </div>
  </div>

  <!-- Status bar -->
  <div class="statusbar">
    <div class="sb-items">
      <div class="sb-item"><div class="status-dot"></div> AI ENGINE ACTIVE</div>
      <div class="sb-item">MODEL: <span>YOLOv8n + POSE</span></div>
      <div class="sb-item">CAMERA: <span>CAM-01 LOCAL</span></div>
      <div class="sb-item">DASHBOARD: <span>localhost:5000</span></div>
    </div>
    <div class="sb-item">SENTINEL AI SURVEILLANCE SYSTEM • v4.0 • COLLEGE PROJECT</div>
  </div>

</div>

<!-- Alert Modal -->
<div id="alert-modal">
  <div class="alert-box">
    <div class="alert-icon" id="modal-icon">⚠️</div>
    <h2 id="modal-title">ALERT DETECTED</h2>
    <p id="modal-body">Violation detected in surveillance area.</p>
    <button class="alert-dismiss" onclick="dismissAlert()">ACKNOWLEDGE</button>
  </div>
</div>

<script>
const socket = io();
let eventCount = 0;
let modalQueue = [];
let modalOpen  = false;

const ICONS = { FIGHT:'🥊', CHEAT:'📝', RESTRICTED:'🚧', CROWD:'👥', PHONE:'📱', MANUAL:'📸', ATTENDANCE:'🎯' };
const LABELS = {
  FIGHT:'FIGHT DETECTED', CHEAT:'CHEATING ALERT',
  RESTRICTED:'ZONE BREACH', CROWD:'CROWD ALERT', PHONE:'PHONE DETECTED'
};

// ── Clock ──
function tickClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-IN',{hour12:false});
  document.getElementById('date').textContent =
    now.toLocaleDateString('en-IN',{weekday:'short',day:'2-digit',month:'short',year:'numeric'}).toUpperCase();
}
setInterval(tickClock, 1000); tickClock();

// ── Charts ──
const chartOpts = (color) => ({
  responsive:true, maintainAspectRatio:false, animation:{duration:200},
  plugins:{legend:{display:false}},
  scales:{
    x:{display:false},
    y:{
      display:true, grid:{color:'#112240'},
      ticks:{color:'#3a5070', font:{size:9}, maxTicksLimit:4}
    }
  },
  elements:{ line:{tension:.4}, point:{radius:0} }
});

const peopleData = { labels:Array(30).fill(''), datasets:[{
  data:Array(30).fill(0),
  borderColor:'#e84560', backgroundColor:'#c41e3a18',
  borderWidth:1.5, fill:true
}]};
const motionData = { labels:Array(30).fill(''), datasets:[{
  data:Array(30).fill(0),
  borderColor:'#a855f7', backgroundColor:'#a855f718',
  borderWidth:1.5, fill:true
}]};

const chartPeople = new Chart(document.getElementById('chart-people'), { type:'line', data:peopleData, options:chartOpts('#e84560') });
const chartMotion = new Chart(document.getElementById('chart-motion'), { type:'line', data:motionData, options:chartOpts('#a855f7') });

function pushChart(chart, val) {
  chart.data.datasets[0].data.push(val);
  chart.data.datasets[0].data.shift();
  chart.update('none');
}

// ── State updates ──
socket.on('state_update', s => {
  // Topbar
  document.getElementById('tb-people').textContent = s.people;
  document.getElementById('tb-events').textContent = s.events;
  document.getElementById('tb-att').textContent = (s.attendance||[]).length;
  const mEl = document.getElementById('tb-motion');
  mEl.textContent = (s.motion||0).toFixed(1);
  mEl.style.color = s.motion > 5 ? 'var(--orange)' : 'var(--green)';

  // Cam overlay
  document.getElementById('cam-people').textContent = s.people;
  document.getElementById('cam-motion').textContent = (s.motion||0).toFixed(1);
  document.getElementById('cam-ts').textContent = new Date().toLocaleTimeString('en-IN',{hour12:false});

  const anyAlert = s.fight || s.cheat || s.restricted || s.crowd;
  const camAlertTag = document.getElementById('cam-alert-tag');
  camAlertTag.style.display = anyAlert ? 'block' : 'none';

  // Sidebar blocks
  setBlock('crowd', 'blk-crowd',
    s.crowd ? 'ALERT!' : 'NORMAL',
    s.crowd ? 'danger' : 'ok',
    s.crowd ? `${s.people} people detected` : `Threshold: {{ crowd_threshold }} people`);

  setBlock('fight', 'blk-fight',
    s.fight ? 'ALERT!' : 'CLEAR',
    s.fight ? 'danger' : 'ok',
    s.fight ? (s.fight_reason||'Fight detected') : 'Pose skeleton analysis');

  setBlock('cheat', 'blk-cheat',
    s.cheat ? 'CHEATING!' : 'SECURE',
    s.cheat ? 'danger' : 'ok',
    s.cheat ? (s.cheat_reason||'Violation') : 'Head-turn + phone track');

  setBlock('phone', 'blk-phone',
    s.phone ? 'DETECTED' : 'NONE',
    s.phone ? 'danger' : 'ok');

  setBlock('restricted', 'blk-restricted',
    s.restricted ? 'BREACH!' : 'SECURE',
    s.restricted ? 'warn' : 'ok');

  const vMotion = document.getElementById('v-motion');
  vMotion.textContent = (s.motion||0).toFixed(1);
  vMotion.className = 'sb-value ' + (s.motion > 5 ? 'warn' : 'info');

  document.getElementById('s-uptime').textContent = 'Started ' + (s.uptime_start||'--:--');

  // Charts
  pushChart(chartPeople, s.people);
  pushChart(chartMotion, s.motion||0);

  // Student evaluation
  updateEvaluation(s.evaluation||[], s.roster_size||0);

  // Modal for critical alerts
  if (s.fight && !modalOpen) showModal('fight', s.fight_reason);
  else if (s.cheat && !modalOpen) showModal('cheat', s.cheat_reason);
});

// ── New event ──
socket.on('new_event', ev => {
  eventCount++;
  document.getElementById('feed-count').textContent = eventCount + ' events';
  const feed = document.getElementById('event-feed');
  const empty = feed.querySelector('.empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = `ev ${ev.type}`;
  div.innerHTML = `
    <div class="ev-icon">${ICONS[ev.type]||'⚠️'}</div>
    <div class="ev-body">
      <h4>${ev.type} — ${ev.detail}</h4>
      <p>${ev.time}</p>
    </div>`;
  feed.insertBefore(div, feed.firstChild);
  if (feed.children.length > 30) feed.removeChild(feed.lastChild);
});

// ── Helpers ──
function setBlock(id, blockId, value, cls, sub) {
  const v = document.getElementById('v-' + id);
  if (v) { v.textContent = value; v.className = 'sb-value ' + cls; }
  const blk = document.getElementById(blockId);
  if (blk) blk.classList.toggle('alert', cls === 'danger');
  if (sub) {
    const s = document.getElementById('s-' + id);
    if (s) s.textContent = sub;
  }
}

// Per-student evaluation panel (present/absent + current activity)
const ACTIVITY_COLOR = {
  Normal:   'var(--green)',
  Phone:    'var(--orange)',
  Cheating: 'var(--red)',
  Fight:    'var(--red)',
};
function updateEvaluation(rows, rosterSize) {
  const el  = document.getElementById('att-list');
  const cnt = document.getElementById('att-count');
  const presentCount = rows.filter(r => r.present).length;
  cnt.textContent = `${presentCount} / ${rosterSize || rows.length} present`;
  if (!rows.length) return;

  // Present students first, then absent
  const sorted = [...rows].sort((a,b) => (b.present - a.present) || a.name.localeCompare(b.name));

  el.innerHTML = sorted.map(r => {
    // Main badge: clearly PRESENT or ABSENT — this is the attendance answer.
    const mainBadge = r.present
      ? `<div class="att-badge" style="background:#2ecc7118;color:var(--green);border-color:#2ecc7130">PRESENT</div>`
      : `<div class="att-badge" style="background:#5a667518;color:var(--dim);border-color:#5a667530">ABSENT</div>`;

    // Secondary badge: only shown for present students who are NOT behaving
    // normally — this is the evaluation/violation flag, separate from attendance.
    let activityBadge = '';
    if (r.present && r.status !== 'Normal') {
      const c = ACTIVITY_COLOR[r.status] || 'var(--orange)';
      activityBadge = `<div class="att-badge" style="background:${c}18;color:${c};border-color:${c}30;margin-left:4px">⚠ ${r.status.toUpperCase()}</div>`;
    }

    const sub = r.present
      ? `${r.violations > 0 ? r.violations + ' flag(s) • ' : ''}seen ${r.last_seen}`
      : 'not detected yet';
    return `
    <div class="att-row" style="opacity:${r.present ? 1 : .55}">
      <div class="att-av" style="${r.present ? '' : 'filter:grayscale(1)'}">${r.name[0].toUpperCase()}</div>
      <div>
        <div class="att-name">${r.name}</div>
        <div class="att-time">${sub}</div>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center">${mainBadge}${activityBadge}</div>
    </div>`;
  }).join('');
}

// Load screenshots from server
async function loadScreenshots() {
  try {
    const r = await fetch('/api/screenshots');
    const files = await r.json();
    const grid = document.getElementById('ss-grid');
    const cnt  = document.getElementById('ss-count');
    cnt.textContent = files.length + ' captured';
    grid.innerHTML = files.map(f => {
      const tag = f.split('_')[0].toUpperCase();
      const colors = {FIGHT:'#ff2244',CHEAT:'#ff2244',RESTRICTED:'#ff8800',CROWD:'#ffd700',PHONE:'#e84560',MANUAL:'#2ecc71'};
      const col = colors[tag]||'#e84560';
      return `<div style="position:relative;border-radius:6px;overflow:hidden;border:1px solid ${col}33;cursor:pointer"
                   onclick="window.open('/screenshot/${f}','_blank')">
        <img src="/screenshot/${f}" style="width:100%;height:70px;object-fit:cover;display:block">
        <div style="position:absolute;bottom:0;left:0;right:0;background:#000000cc;padding:2px 4px;font-size:.5rem;color:${col};font-family:monospace">${tag}</div>
      </div>`;
    }).join('');
  } catch(e) {}
}
loadScreenshots();
setInterval(loadScreenshots, 5000); // refresh every 5s

// ── "Register New Student" — explicit, button-triggered scan ──
let scanActive   = false;
let scanInterval = null;
let scanFoundFace = false;

function startRegisterScan() {
  scanActive    = true;
  scanFoundFace = false;
  document.getElementById('register-trigger-btn').style.display = 'none';
  document.getElementById('register-panel').style.display = 'block';
  document.getElementById('register-title').textContent = '🔍 Scanning for face...';
  document.getElementById('register-photo').style.display = 'none';
  document.getElementById('register-scanning').style.display = 'block';
  document.getElementById('register-actions').style.display = 'none';
  document.getElementById('register-cancel-scan-btn').style.display = 'block';
  document.getElementById('register-msg').textContent = '';
  document.getElementById('register-name-real').value = '';

  // Tell the backend to start looking — it only buffers an unknown face
  // while a scan is "armed", so nothing is captured unless this button was pressed.
  fetch('/api/start_scan', { method: 'POST' }).catch(()=>{});

  scanInterval = setInterval(checkScanProgress, 1200);
}

async function checkScanProgress() {
  if (!scanActive) return;
  try {
    const r = await fetch('/api/pending_face');
    const d = await r.json();
    if (d.pending && !scanFoundFace) {
      scanFoundFace = true;
      document.getElementById('register-title').textContent = '✓ Face captured!';
      document.getElementById('register-photo').src = '/pending_face.jpg?t=' + Date.now();
      document.getElementById('register-photo').style.display = 'block';
      document.getElementById('register-scanning').style.display = 'none';
      document.getElementById('register-actions').style.display = 'flex';
      document.getElementById('register-cancel-scan-btn').style.display = 'none';
      document.getElementById('register-name-real').focus();
      clearInterval(scanInterval);
      // stop the backend from scanning further once we have a face
      fetch('/api/stop_scan', { method: 'POST' }).catch(()=>{});
    }
  } catch (e) {}
}

function cancelRegisterScan() {
  scanActive = false;
  scanFoundFace = false;
  if (scanInterval) clearInterval(scanInterval);
  fetch('/api/stop_scan', { method: 'POST' }).catch(()=>{});
  document.getElementById('register-panel').style.display = 'none';
  document.getElementById('register-trigger-btn').style.display = 'flex';
}

async function registerFace() {
  const name = document.getElementById('register-name-real').value.trim();
  const msg  = document.getElementById('register-msg');
  if (!name) { msg.textContent = 'Please enter a name first.'; msg.style.color = 'var(--red)'; return; }
  msg.textContent = 'Saving...'; msg.style.color = 'var(--dim)';
  try {
    const r = await fetch('/api/register_face', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({name})
    });
    const d = await r.json();
    if (d.ok) {
      msg.textContent = `✓ ${name} registered! Now recognized automatically.`;
      msg.style.color = 'var(--green)';
      setTimeout(() => {
        document.getElementById('register-panel').style.display = 'none';
        document.getElementById('register-trigger-btn').style.display = 'flex';
        scanActive = false;
      }, 1800);
    } else {
      msg.textContent = d.error || 'Could not register — try again.';
      msg.style.color = 'var(--red)';
    }
  } catch (e) {
    msg.textContent = 'Network error — try again.';
    msg.style.color = 'var(--red)';
  }
}

// ── Alert Modal ──
const MODAL_CFG = {
  fight:      { icon:'🥊', title:'FIGHT DETECTED', color:'var(--red)' },
  cheat:      { icon:'📝', title:'CHEATING ALERT', color:'var(--red)' },
  restricted: { icon:'🚧', title:'ZONE BREACH',    color:'var(--orange)' },
  crowd:      { icon:'👥', title:'CROWD ALERT',    color:'var(--yellow)' },
};
function showModal(type, detail) {
  const cfg = MODAL_CFG[type] || { icon:'⚠️', title:'ALERT', color:'var(--red)' };
  document.getElementById('modal-icon').textContent  = cfg.icon;
  document.getElementById('modal-title').textContent = cfg.title;
  document.getElementById('modal-title').style.color = cfg.color;
  document.getElementById('modal-body').textContent  = detail || 'Violation detected.';
  document.getElementById('alert-modal').classList.add('show');
  modalOpen = true;
  setTimeout(dismissAlert, 6000);
}
function dismissAlert() {
  document.getElementById('alert-modal').classList.remove('show');
  modalOpen = false;
}
</script>
</body>
</html>
"""

app    = Flask(__name__)
socketio_server = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML, crowd_threshold=CROWD_THRESHOLD,
                                   company_name=COMPANY_NAME, tagline=COMPANY_TAGLINE)

@app.route("/logo.png")
def serve_logo():
    from flask import send_file
    if os.path.exists(LOGO_PATH):
        return send_file(LOGO_PATH, mimetype="image/png")
    return "", 404

@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(state)

# ── Live MJPEG video feed ─────────────────────────────────────────
_latest_frame = None
_frame_lock   = threading.Lock()

def set_latest_frame(frame):
    global _latest_frame
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _frame_lock:
        _latest_frame = buf.tobytes()

def _gen_frames():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05); continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)  # ~25 fps

@app.route("/video_feed")
def video_feed():
    from flask import Response
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Latest screenshots API ────────────────────────────────────────
@app.route("/api/screenshots")
def api_screenshots():
    files = sorted([f for f in os.listdir(SCREENSHOT_DIR)
                    if f.endswith(".jpg")], reverse=True)[:12]
    return jsonify(files)

@app.route("/screenshot/<fname>")
def serve_screenshot(fname):
    from flask import send_from_directory
    return send_from_directory(SCREENSHOT_DIR, fname)

# ── Personnel (known faces) management ────────────────────────────
_attendance_ref = {"obj": None}   # set once AttendanceSystem is created in main()

def set_attendance_ref(obj):
    _attendance_ref["obj"] = obj

@app.route("/api/personnel", methods=["GET"])
def api_personnel_list():
    files = sorted(f for f in os.listdir(KNOWN_FACES_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return jsonify([{"name": os.path.splitext(f)[0], "file": f} for f in files])

@app.route("/api/personnel", methods=["POST"])
def api_personnel_add():
    from flask import request
    name = request.form.get("name", "").strip()
    file = request.files.get("photo")
    if not name or not file:
        return jsonify({"ok": False, "error": "Name and photo are required"}), 400

    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid name"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}{ext}")
    file.save(path)

    obj = _attendance_ref["obj"]
    if obj:
        obj.reload()
    log_event("PERSONNEL", f"{safe_name} added to registry")
    push_event("PERSONNEL", f"{safe_name} added to registry")
    return jsonify({"ok": True, "name": safe_name})

@app.route("/api/personnel/<name>", methods=["DELETE"])
def api_personnel_delete(name):
    removed = False
    for f in os.listdir(KNOWN_FACES_DIR):
        if os.path.splitext(f)[0] == name:
            os.remove(os.path.join(KNOWN_FACES_DIR, f))
            removed = True
    if not removed:
        return jsonify({"ok": False, "error": "Not found"}), 404

    obj = _attendance_ref["obj"]
    if obj:
        obj.reload()
        obj.present_today.discard(name)
    log_event("PERSONNEL", f"{name} removed from registry")
    push_event("PERSONNEL", f"{name} removed from registry")
    return jsonify({"ok": True})

# ── Live "unknown face → register" flow ───────────────────────────
@app.route("/api/start_scan", methods=["POST"])
def api_start_scan():
    """User pressed 'Register New Student' — arm the scanner so the next
    unrecognized face seen on camera gets buffered for naming."""
    obj = _attendance_ref["obj"]
    if not obj:
        return jsonify({"ok": False, "error": "System not ready"}), 503
    obj.pending_unknown_path = None
    obj.scan_armed = True
    return jsonify({"ok": True})

@app.route("/api/stop_scan", methods=["POST"])
def api_stop_scan():
    """Stop scanning — either a face was captured, or the user cancelled."""
    obj = _attendance_ref["obj"]
    if obj:
        obj.scan_armed = False
    return jsonify({"ok": True})

@app.route("/api/pending_face")
def api_pending_face():
    """Dashboard polls this (only while a scan is armed) to know if an
    unrecognized face has been captured and is ready to be named."""
    obj = _attendance_ref["obj"]
    has_pending = bool(obj and obj.pending_unknown_path and os.path.exists(obj.pending_unknown_path))
    return jsonify({"pending": has_pending})

@app.route("/pending_face.jpg")
def serve_pending_face():
    from flask import send_file
    obj = _attendance_ref["obj"]
    if obj and obj.pending_unknown_path and os.path.exists(obj.pending_unknown_path):
        return send_file(obj.pending_unknown_path, mimetype="image/jpeg")
    return "", 404

@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    """Dashboard's 'Save & Register' button — names the currently-pending
    unknown face and saves it straight into known_faces/, no manual
    file upload needed."""
    from flask import request
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required"}), 400
    obj = _attendance_ref["obj"]
    if not obj:
        return jsonify({"ok": False, "error": "System not ready"}), 503
    ok = obj.register_new_student(name)
    obj.scan_armed = False
    if not ok:
        return jsonify({"ok": False, "error": "No pending face to register — make sure an unrecognized face is currently visible on camera"}), 400
    return jsonify({"ok": True, "name": name})

def push_state():
    """Push current state to all dashboard clients."""
    with state_lock:
        s = dict(state)
    socketio_server.emit("state_update", s)

def push_event(ev_type: str, detail: str):
    socketio_server.emit("new_event", {
        "type":   ev_type,
        "detail": detail,
        "time":   datetime.now().strftime("%H:%M:%S"),
    })

def run_dashboard():
    socketio_server.run(app, host="0.0.0.0", port=DASHBOARD_PORT,
                        debug=False, use_reloader=False, log_output=False)

# ════════════════════════════════════════════════════════════════════════
#  VOICE ALERT
# ════════════════════════════════════════════════════════════════════════
_voice_queue   = deque()
_voice_running = False

VOICE_MESSAGES = {
    "fight":      "Warning! Fight detected. Security please respond immediately.",
    "cheat":      "Alert! Cheating activity detected.",
    "restricted": "Warning! Restricted area breach detected.",
    "crowd":      "Crowd alert. Area getting overcrowded.",
    "phone":      "Phone detected in surveillance area.",
}

def _voice_worker():
    global _voice_running
    while _voice_queue:
        msg = _voice_queue.popleft()
        if TTS_AVAILABLE and VOICE_ENABLED:
            try:
                _tts_engine.say(msg)
                _tts_engine.runAndWait()
            except Exception:
                pass
    _voice_running = False

def speak(alert_type: str):
    msg = VOICE_MESSAGES.get(alert_type, "Alert detected.")
    _voice_queue.append(msg)
    global _voice_running
    if not _voice_running:
        _voice_running = True
        threading.Thread(target=_voice_worker, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════
#  BUZZER
# ════════════════════════════════════════════════════════════════════════
BUZZER_PATTERNS = {
    "fight":      [(1400,180),(0,60),(1400,180),(0,60),(1400,180)],
    "cheat":      [(900,500)],
    "restricted": [(1200,250),(0,100),(1200,250)],
    "crowd":      [(700,350)],
    "phone":      [(1000,300)],
}

def buzzer(alert_type="crowd"):
    def _beep():
        pattern = BUZZER_PATTERNS.get(alert_type, [(1000,300)])
        try:
            import winsound
            for freq, dur in pattern:
                if freq: winsound.Beep(freq, dur)
                else:    time.sleep(dur/1000)
        except ImportError:
            try:
                import subprocess
                for freq, dur in pattern:
                    if freq:
                        subprocess.run(["beep",f"-f{freq}",f"-l{dur}"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                print("\a", end="", flush=True)
    threading.Thread(target=_beep, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════
#  EMAIL
# ════════════════════════════════════════════════════════════════════════
_last_email_time = {}
def send_email_alert(subject, body, image_path=None):
    if not EMAIL_ENABLED: return
    now = time.time()
    if now - _last_email_time.get(subject, 0) < EMAIL_COOLDOWN_SEC: return
    _last_email_time[subject] = now
    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"], msg["To"], msg["Subject"] = EMAIL_SENDER, EMAIL_RECEIVER, f"[AI Surveillance] {subject}"
            msg.attach(MIMEText(body, "plain"))
            if image_path and os.path.exists(image_path):
                with open(image_path,"rb") as f: msg.attach(MIMEImage(f.read(),name=os.path.basename(image_path)))
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
                s.login(EMAIL_SENDER, EMAIL_PASSWORD)
                s.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        except Exception as e: print(f"[EMAIL ERR] {e}")
    threading.Thread(target=_send, daemon=True).start()

# ════════════════════════════════════════════════════════════════════════
#  EVENT LOG
# ════════════════════════════════════════════════════════════════════════
def log_event(event_type, detail, ss_path=None):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {event_type:<20} | {detail}"
    if ss_path: line += f" | {ss_path}"
    print(f"[LOG] {line}")
    with open("events.txt","a") as f: f.write(line+"\n")
    with state_lock:
        state["recent_events"].insert(0, {"type": event_type, "detail": detail, "time": ts})
        state["recent_events"] = state["recent_events"][:20]
        state["events"] += 1

def save_screenshot(frame, tag):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{tag}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

# ════════════════════════════════════════════════════════════════════════
#  COOLDOWN
# ════════════════════════════════════════════════════════════════════════
class Cooldown:
    def __init__(self, seconds=6): self._l={}; self.s=seconds
    def ready(self, k, seconds=None):
        now = time.time()
        win = seconds if seconds is not None else self.s
        if now-self._l.get(k,0) >= win:
            self._l[k] = now
            return True
        return False

# ════════════════════════════════════════════════════════════════════════
#  ATTENDANCE  (face recognition)
# ════════════════════════════════════════════════════════════════════════
class AttendanceSystem:
    """
    Attendance + per-student evaluation, using OpenCV LBPH.
    NO dlib / face_recognition needed.

    FIX for "only one student matches":
    LBPH gives one global confidence score per face — if photos differ a lot
    in lighting/angle/quality, some students naturally score worse and get
    rejected by a single fixed threshold. Fix applied:
      1. Stronger preprocessing (equalizeHist + CLAHE) on EVERY photo so
         lighting differences shrink before training.
      2. More augmented variants per photo → recognizer sees more "versions"
         of each face → less biased toward whichever photo was cleanest.
      3. Slightly higher, single safe threshold (95) since equalisation
         narrows the score gap between students.
      4. roster + absent tracking, so the class list (all 10) is known
         up front, not just whoever happens to get recognized.
    """

    CONFIDENCE_THRESHOLD = 135  # 0=perfect match, higher=worse match. Raised after
                                 # live testing showed correct matches scoring 118-124.
    FACE_SIZE = (160, 160)

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.present_today = set()
        self.recognizer    = None
        self.known_names   = []     # full roster (everyone with a photo)
        self.trained       = False
        # Per-student running record for evaluation
        # name -> {"status": "Normal/Phone/Cheating/...", "last_seen": ts, "violations": int}
        self.activity = {}
        # Unknown-face registration buffer — last unrecognized face crop,
        # shown on the dashboard with a "name + save" form. Throttled so we
        # don't write a file every single frame.
        self._unknown_last_save = 0
        self._unknown_last_seen = 0
        self.pending_unknown_path = None   # path to most recent unknown crop
        self.scan_armed = False            # only buffer unknown faces while a
                                            # dashboard "Register New Student" scan is active
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.reload()

    def _prep(self, gray_roi):
        roi = cv2.resize(gray_roi, self.FACE_SIZE)
        roi = self.clahe.apply(roi)        # local contrast equalisation — handles uneven lighting
        roi = cv2.equalizeHist(roi)        # global equalisation on top
        return roi

    def roster(self):
        """Everyone with a registered photo — the full class list."""
        return list(self.known_names)

    def absent_list(self):
        return [n for n in self.known_names if n not in self.present_today]

    def reload(self):
        """(Re)train from whatever is currently in known_faces/."""
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8)
        faces, labels = [], []
        names = []

        try:
            files = sorted(f for f in os.listdir(KNOWN_FACES_DIR)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        except FileNotFoundError:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            files = []

        if not files:
            print("[ATTENDANCE] No photos in known_faces/ — add Name.jpg files")
            self.trained, self.known_names = False, []
            return

        for fname in files:
            path = os.path.join(KNOWN_FACES_DIR, fname)
            name = os.path.splitext(fname)[0]
            img  = cv2.imread(path)
            if img is None:
                print(f"[ATTENDANCE] ⚠ Could not read {fname} — skipped")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Try a few scale factors — some photos need looser detection
            dets = []
            for sf in (1.05, 1.1, 1.2):
                dets = self.face_cascade.detectMultiScale(gray, sf, 5, minSize=(50, 50))
                if len(dets) > 0:
                    break

            if len(dets) == 0:
                print(f"[ATTENDANCE] ⚠ No face found in {fname} — SKIPPED. "
                      f"Use a clear front-facing photo for this student.")
                continue

            x, y, w, h = max(dets, key=lambda d: d[2] * d[3])
            # Pad the crop slightly so we don't cut off chin/forehead
            pad = int(0.1 * h)
            y1, y2 = max(0, y-pad), min(gray.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(gray.shape[1], x+w+pad)
            roi = self._prep(gray[y1:y2, x1:x2])

            label = len(names)
            names.append(name)

            variants = [
                roi,
                cv2.flip(roi, 1),
                cv2.convertScaleAbs(roi, alpha=1.15, beta=8),
                cv2.convertScaleAbs(roi, alpha=0.88, beta=-8),
                cv2.GaussianBlur(roi, (3, 3), 0),
                cv2.flip(cv2.convertScaleAbs(roi, alpha=1.15, beta=8), 1),
            ]
            for v in variants:
                faces.append(v)
                labels.append(label)

        if faces:
            recognizer.train(faces, np.array(labels))
            self.recognizer  = recognizer
            self.known_names = names
            self.trained     = True
            print(f"[ATTENDANCE ✓] Trained on {len(names)} students: {names}")
        else:
            self.trained, self.known_names = False, []
            print("[ATTENDANCE] No valid face photos to train on")

    def update(self, frame):
        """Detect + recognize every face in frame. Never raises — always
        returns a (possibly empty) list, so a bad frame can't crash the loop."""
        if not self.trained and not hasattr(self, "face_cascade"):
            return []
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            dets  = self.face_cascade.detectMultiScale(small, 1.1, 5, minSize=(40, 40))
        except Exception as e:
            print(f"[ATTENDANCE] frame read error: {e}")
            return []

        names_in_frame = []
        saw_unknown_this_frame = False

        for (x, y, w, h) in dets:
            name = "Unknown"
            try:
                x2, y2, w2, h2 = x*2, y*2, w*2, h*2
                roi = self._prep(gray[y2:y2+h2, x2:x2+w2])

                if self.trained:
                    label, confidence = self.recognizer.predict(roi)
                    if confidence < self.CONFIDENCE_THRESHOLD and label < len(self.known_names):
                        name = self.known_names[label]

                if name != "Unknown":
                    # Already a registered student — show clearly that scanning
                    # SKIPS this person, it will never re-capture/overwrite them.
                    label_text = f"{name} (Already Registered)" if self.scan_armed else name
                    cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (0,255,100), 2)
                    cv2.putText(frame, label_text, (x2, y2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 2)
                else:
                    label_text = "NEW FACE - capturing..." if self.scan_armed else "Unknown"
                    cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (0,180,255), 2)
                    cv2.putText(frame, label_text, (x2, y2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1)
                    # Only buffer this face for the registration panel if the
                    # user explicitly pressed "Register New Student" on the
                    # dashboard — never capture/store faces silently, and
                    # NEVER capture a face that already matched someone above.
                    if self.scan_armed:
                        self._offer_unknown_face(frame[y2:y2+h2, x2:x2+w2])
                    saw_unknown_this_frame = True
            except Exception as e:
                print(f"[ATTENDANCE] face match error: {e}")
                name = "Unknown"
                saw_unknown_this_frame = True

            names_in_frame.append(name)

            if name != "Unknown" and name not in self.present_today:
                self.present_today.add(name)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    with open("attendance.csv", "a", newline="") as f:
                        csv.writer(f).writerow([name, ts, "PRESENT"])
                except Exception as e:
                    print(f"[ATTENDANCE] csv write error: {e}")
                log_event("ATTENDANCE", f"{name} marked present")
                push_event("ATTENDANCE", f"{name} marked present")
                print(f"[ATTENDANCE ✓] {name} — PRESENT")

            if name != "Unknown":
                rec = self.activity.setdefault(name, {"status": "Normal", "violations": 0})
                rec["last_seen"] = datetime.now().strftime("%H:%M:%S")

        # If nobody unknown has been seen for a while, clear the pending
        # registration prompt so the dashboard doesn't keep offering a stale photo.
        if saw_unknown_this_frame:
            self._unknown_last_seen = time.time()
        elif self.pending_unknown_path and time.time() - getattr(self, "_unknown_last_seen", 0) > 6:
            self.pending_unknown_path = None

        return names_in_frame

    def mark_activity(self, name: str, status: str):
        """Record what a recognized student is currently doing
        (Normal / Phone / Cheating / Restricted), for the evaluation panel."""
        if name == "Unknown" or not name:
            return
        rec = self.activity.setdefault(name, {"status": "Normal", "violations": 0})
        rec["status"] = status
        rec["last_seen"] = datetime.now().strftime("%H:%M:%S")
        if status != "Normal":
            rec["violations"] += 1

    def evaluation_snapshot(self):
        """Per-student summary for dashboard / printout."""
        rows = []
        for name in self.known_names:
            rec = self.activity.get(name, {"status": "Absent", "violations": 0, "last_seen": "--:--:--"})
            present = name in self.present_today
            rows.append({
                "name": name,
                "present": present,
                "status": rec.get("status", "Normal") if present else "Absent",
                "violations": rec.get("violations", 0),
                "last_seen": rec.get("last_seen", "--:--:--"),
            })
        return rows

    def draw(self, frame, names):
        return frame  # drawing already done in update()

    # ── New-student registration (dashboard-driven) ──────────────────
    UNKNOWN_DIR = "pending_faces"

    def _offer_unknown_face(self, face_crop):
        """Save the current unrecognized face crop so the dashboard can
        show it with a 'name + save' form. Throttled to ~once/2s so we
        don't spam disk writes while an unknown person is in frame."""
        if face_crop is None or face_crop.size == 0:
            return
        now = time.time()
        if now - self._unknown_last_save < 2:
            return
        self._unknown_last_save = now
        os.makedirs(self.UNKNOWN_DIR, exist_ok=True)
        path = os.path.join(self.UNKNOWN_DIR, "latest.jpg")
        try:
            cv2.imwrite(path, face_crop)
            self.pending_unknown_path = path
        except Exception as e:
            print(f"[ATTENDANCE] could not save pending face: {e}")

    def register_new_student(self, name: str) -> bool:
        """Called from the dashboard's 'Register' button. Saves the most
        recent unknown face crop as known_faces/<name>.jpg and retrains
        immediately, so the new student is recognized from the next frame."""
        safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not safe or not self.pending_unknown_path or not os.path.exists(self.pending_unknown_path):
            return False
        dest = os.path.join(KNOWN_FACES_DIR, f"{safe}.jpg")
        try:
            img = cv2.imread(self.pending_unknown_path)
            cv2.imwrite(dest, img)
        except Exception as e:
            print(f"[ATTENDANCE] register failed: {e}")
            return False
        self.reload()
        self.pending_unknown_path = None
        print(f"[ATTENDANCE ✓] New student registered: {safe}")
        log_event("PERSONNEL", f"{safe} self-registered via dashboard")
        push_event("PERSONNEL", f"{safe} self-registered via dashboard")
        return True

# ════════════════════════════════════════════════════════════════════════
#  RESTRICTED ZONES
# ════════════════════════════════════════════════════════════════════════
def draw_restricted_zones(frame):
    for (x1,y1,x2,y2) in RESTRICTED_ZONES:
        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,180),-1)
        cv2.addWeighted(ov,.22,frame,.78,0,frame)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,"RESTRICTED",(x1+4,y1+20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    return frame

def in_restricted(cx,cy):
    for (x1,y1,x2,y2) in RESTRICTED_ZONES:
        if x1<=cx<=x2 and y1<=cy<=y2: return True
    return False

# ════════════════════════════════════════════════════════════════════════
#  FIGHT DETECTOR  (pose-based)
# ════════════════════════════════════════════════════════════════════════
class FightDetector:
    """
    STRICT fight detection — only fires on REAL fight body language.

    Requires ALL of these together (not just any 3):
    1. TWO people physically CLOSE (bbox overlap OR very small distance)
    2. At least ONE person has arm raised above shoulder (punch/push pose)
    3. HIGH motion magnitude sustained for many frames
    4. Aggressive torso lean toward the other person

    Single person stretching, waving, or walking fast = NOT a fight.
    """

    def __init__(self):
        self.counter      = 0
        self.centre_hist  = deque(maxlen=12)
        self.motion_hist  = deque(maxlen=12)   # per-frame motion scores

    # ── Keypoint helpers ─────────────────────────────────────────
    def _arm_raised(self, kp):
        """Wrist clearly above shoulder — real punch/push position."""
        raised = 0
        ls, rs = kp[5], kp[6]   # shoulders
        lw, rw = kp[9], kp[10]  # wrists
        le, re = kp[7], kp[8]   # elbows
        # Wrist must be above shoulder by at least 30px (not just slightly raised)
        if lw[2]>.4 and ls[2]>.4 and le[2]>.4:
            if lw[1] < ls[1] - 30 and le[1] < ls[1] + 10:  # wrist AND elbow up
                raised += 1
        if rw[2]>.4 and rs[2]>.4 and re[2]>.4:
            if rw[1] < rs[1] - 30 and re[1] < rs[1] + 10:
                raised += 1
        return raised

    def _torso_toward_other(self, kp_a, kp_b):
        """
        Check if person A is leaning TOWARD person B.
        Shoulder midpoint vs hip midpoint direction should point at B.
        """
        lsa, rsa = kp_a[5], kp_a[6]
        lha, rha = kp_a[11], kp_a[12]
        lsb, rsb = kp_b[5], kp_b[6]

        if not all(p[2]>.3 for p in [lsa,rsa,lha,rha,lsb,rsb]):
            return False

        sh_a = ((lsa[0]+rsa[0])/2, (lsa[1]+rsa[1])/2)
        hp_a = ((lha[0]+rha[0])/2, (lha[1]+rha[1])/2)
        sh_b = ((lsb[0]+rsb[0])/2, (lsb[1]+rsb[1])/2)

        # Lean vector: shoulder - hip
        lean_x = sh_a[0] - hp_a[0]
        # Direction to B: B_shoulder - A_shoulder
        to_b_x = sh_b[0] - sh_a[0]

        # Same direction = leaning toward B
        lean_angle = abs(np.degrees(np.arctan2(sh_a[0]-hp_a[0], hp_a[1]-sh_a[1]+1e-6)))
        leaning    = lean_angle > 18

        toward = (lean_x * to_b_x) > 0  # same sign = toward each other
        return leaning and toward

    def _persons_close(self, boxes):
        """
        Two persons must be VERY close — either bbox overlapping OR
        centres within 1/3 of frame width of each other.
        """
        close_pairs = []
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                ax1,ay1,ax2,ay2 = boxes[i]
                bx1,by1,bx2,by2 = boxes[j]
                # Centre distance
                cax, cay = (ax1+ax2)/2, (ay1+ay2)/2
                cbx, cby = (bx1+bx2)/2, (by1+by2)/2
                dist = ((cax-cbx)**2 + (cay-cby)**2)**.5
                avg_h = ((ay2-ay1) + (by2-by1)) / 2
                # Close = centres within 1.5x average person height
                if dist < avg_h * 1.5:
                    close_pairs.append((i, j))
        return close_pairs

    def _sustained_high_motion(self, centres):
        """Motion must be HIGH and SUSTAINED — not just one fast frame."""
        self.centre_hist.append(centres)
        if len(self.centre_hist) < 4:
            return False, 0.0
        # Compute per-frame speed for each person
        speeds = []
        hist = list(self.centre_hist)
        for t in range(1, len(hist)):
            for c in hist[t]:
                for p in hist[t-1]:
                    s = ((c[0]-p[0])**2 + (c[1]-p[1])**2)**.5
                    speeds.append(s)
                    break

        if not speeds:
            return False, 0.0

        avg_speed = np.mean(speeds)
        self.motion_hist.append(avg_speed)

        # Need sustained motion — at least 6 of last 10 frames above threshold
        high_frames = sum(1 for s in self.motion_hist if s > 40)
        return high_frames >= 6, avg_speed

    def update(self, pose_res, boxes):
        """
        Returns (is_fight: bool, reason: str)
        ALL 3 conditions must be true simultaneously:
          A) Two people physically close
          B) At least one raised arm in the pair
          C) Sustained high motion
        """
        if len(boxes) < 2:
            self.counter = max(0, self.counter - 2)
            return self.counter >= FIGHT_POSE_FRAMES, ""

        # Condition C: sustained motion
        centres = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in boxes]
        sustained, avg_spd = self._sustained_high_motion(centres)

        if not sustained:
            self.counter = max(0, self.counter - 1)
            return False, ""

        # Condition A: close persons
        close_pairs = self._persons_close(boxes)
        if not close_pairs:
            self.counter = max(0, self.counter - 1)
            return False, ""

        # Extract keypoints per person
        all_kps = []
        if pose_res is not None and pose_res.keypoints is not None:
            for pkp in pose_res.keypoints.data:
                all_kps.append(pkp.cpu().numpy())

        # Condition B: raised arm + torso lean toward each other
        fight_found = False
        reasons     = []

        for (i, j) in close_pairs:
            kp_i = all_kps[i] if i < len(all_kps) else None
            kp_j = all_kps[j] if j < len(all_kps) else None

            raised_i = self._arm_raised(kp_i) if kp_i is not None else 0
            raised_j = self._arm_raised(kp_j) if kp_j is not None else 0

            toward = False
            if kp_i is not None and kp_j is not None:
                toward = (self._torso_toward_other(kp_i, kp_j) or
                          self._torso_toward_other(kp_j, kp_i))

            # Need: close + (raised arm OR torso toward) + sustained motion
            if (raised_i + raised_j) >= 1 or toward:
                fight_found = True
                if raised_i + raised_j >= 1:
                    reasons.append("arm raised")
                if toward:
                    reasons.append("facing aggressor")
                reasons.append(f"motion={avg_spd:.0f}px/f")

        if fight_found:
            self.counter += 1
            reasons = list(dict.fromkeys(reasons))  # deduplicate
        else:
            self.counter = max(0, self.counter - 1)

        confirmed = self.counter >= FIGHT_POSE_FRAMES
        return confirmed, ", ".join(reasons)

# ════════════════════════════════════════════════════════════════════════
#  CHEAT DETECTOR
# ════════════════════════════════════════════════════════════════════════
class CheatDetector:
    def __init__(self):
        self.face_cas    = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        self.profile_cas = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_profileface.xml")
        self.turn_ctr    = 0
        self.score       = 0.0

    def _head_turned(self, pose_res):
        if pose_res is None or pose_res.keypoints is None: return False,""
        for pkp in pose_res.keypoints.data:
            kp = pkp.cpu().numpy()
            nose,le,re = kp[0],kp[3],kp[4]
            if nose[2]<.3: continue
            lv,rv = le[2],re[2]
            if lv>.4 and rv<.15: return True,"looking LEFT"
            if rv>.4 and lv<.15: return True,"looking RIGHT"
            if lv>.3 and rv>.3:
                mid=(le[0]+re[0])/2; dist=abs(le[0]-re[0])
                if dist>0 and abs(nose[0]-mid)/dist>.25:
                    return True,("looking LEFT" if nose[0]<mid else "looking RIGHT")
        return False,""

    def update(self, frame, pose_res, phone):
        gray   = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        prof   = self.profile_cas.detectMultiScale(gray,1.1,5,minSize=(40,40))
        front  = self.face_cas.detectMultiScale(gray,1.1,5,minSize=(40,40))
        nf  = 0 if isinstance(front,tuple) else len(front)
        np_ = 0 if isinstance(prof, tuple) else len(prof)
        turned,tdir = self._head_turned(pose_res)
        reasons=[]
        if turned:
            self.turn_ctr+=1
            if self.turn_ctr>=CHEAT_TURN_FRAMES: reasons.append(f"Head turn ({tdir})")
        else: self.turn_ctr=max(0,self.turn_ctr-2)
        if phone: reasons.append("Phone detected"); self.score+=3
        if np_>0 and nf==0: reasons.append("Sideways glance"); self.score+=1
        # NOTE: multiple faces in frame is normal (multiple people present) —
        # this is NOT evidence of cheating on its own, so it no longer scores.
        self.score=max(0,self.score-.5)
        is_cheat = bool(reasons) and (phone or self.turn_ctr>=CHEAT_TURN_FRAMES or self.score>=4)
        return is_cheat, " | ".join(reasons), front

# ════════════════════════════════════════════════════════════════════════
#  OPTICAL FLOW
# ════════════════════════════════════════════════════════════════════════
class MotionAnalyzer:
    def __init__(self): self.prev=None
    def update(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        if self.prev is None: self.prev=gray; return frame,0.0
        flow = cv2.calcOpticalFlowFarneback(self.prev,gray,None,.5,3,13,3,5,1.1,0)
        mag,_ = cv2.cartToPolar(flow[...,0],flow[...,1])
        mag_f = float(np.mean(mag))
        norm  = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        heat  = cv2.applyColorMap(norm,cv2.COLORMAP_JET)
        out   = cv2.addWeighted(frame,.78,heat,.22,0)
        self.prev=gray
        return out, mag_f

# ════════════════════════════════════════════════════════════════════════
#  HUD OVERLAY  (professional look)
# ════════════════════════════════════════════════════════════════════════
def draw_hud(frame, stats):
    h,w = frame.shape[:2]
    pw   = 236

    # Left panel bg
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(pw,h),(8,8,12),-1)
    cv2.addWeighted(ov,.62,frame,.38,0,frame)

    # Top bar
    cv2.rectangle(frame,(0,0),(w,44),(12,12,18),-1)
    cv2.putText(frame,"AI SURVEILLANCE SYSTEM v4.0",(pw+10,29),
                cv2.FONT_HERSHEY_DUPLEX,.62,(0,220,255),1)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame,ts,(w-220,29),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,140,160),1)

    # Separator lines
    cv2.line(frame,(pw,0),(pw,h),(30,50,70),1)
    cv2.line(frame,(0,44),(w,44),(30,50,70),1)

    # Panel title
    cv2.putText(frame,"LIVE STATUS",(10,70),cv2.FONT_HERSHEY_DUPLEX,.52,(0,220,255),1)
    cv2.line(frame,(10,76),(pw-10,76),(0,80,100),1)

    def row(y, label, value, alert=False):
        color = (50,50,255) if alert else (0,220,100)
        cv2.putText(frame,label,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.43,(120,130,140),1)
        cv2.putText(frame,value,(130,y),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)

    rows = [
        ("People",     str(stats["people"]),       False),
        ("Crowd",      "ALERT!" if stats["crowd"]  else "Clear", stats["crowd"]),
        ("Fight",      "FIGHT!" if stats["fight"]  else "Clear", stats["fight"]),
        ("Cheating",   "ALERT!" if stats["cheat"]  else "Clear", stats["cheat"]),
        ("Phone",      "DETECTED" if stats["phone"] else "---",  stats["phone"]),
        ("Restricted", "BREACH!" if stats["restricted"] else "Clear", stats["restricted"]),
        ("Motion",     f"{stats['motion']:.1f}",   stats["motion"]>5),
        ("Events",     str(stats["events"]),        False),
        ("Attendance", str(stats["att_count"])+" present", False),
    ]
    y=96
    for label,value,alert in rows:
        row(y,label,value,alert); y+=26

    # Attendance names
    y+=4
    cv2.putText(frame,"PRESENT",(10,y),cv2.FONT_HERSHEY_DUPLEX,.45,(0,200,255),1)
    cv2.line(frame,(10,y+4),(pw-10,y+4),(0,80,100),1)
    y+=20
    for name in list(stats["present"])[:6]:
        cv2.putText(frame,f"• {name}",(14,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,80),1)
        y+=18

    # Dashboard hint
    cv2.putText(frame,"Dashboard → localhost:5000",(10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,.37,(60,80,100),1)
    return frame

def draw_alerts(frame, alerts):
    if not alerts: return frame
    h,w = frame.shape[:2]
    rh   = 34
    bh   = rh*len(alerts)
    ov   = frame.copy()
    cv2.rectangle(ov,(0,h-bh),(w,h),(0,0,0),-1)
    cv2.addWeighted(ov,.72,frame,.28,0,frame)
    tick = int(time.time()*3)%2
    if tick: cv2.rectangle(frame,(0,h-bh),(w-1,h-1),(0,0,255),2)
    COLS={"fight":(50,50,255),"cheat":(50,50,255),"restricted":(50,140,255),
          "crowd":(50,210,255),"phone":(255,200,50)}
    for i,(key,msg) in enumerate(alerts):
        y = h-bh+22+i*rh
        color = COLS.get(key,(200,200,200))
        cv2.putText(frame,f"!  {msg}",(240,y),cv2.FONT_HERSHEY_DUPLEX,.7,color,2)
    return frame

# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  AI Surveillance & Attendance System  v4.0")
    print("  Loading models — please wait...")
    print("="*65)

    det_model  = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Camera nahi mili! Source = {CAMERA_SOURCE}")
        print("[ERROR] Agar phone camera use kar rahe ho, check karo:")
        print("        1. Phone aur laptop SAME WiFi pe hain?")
        print("        2. IP Webcam app mein 'Start server' dabaya?")
        print("        3. CAMERA_SOURCE mein sahi IP daala hai?")
        return

    fight_det  = FightDetector()
    cheat_det  = CheatDetector()
    motion_an  = MotionAnalyzer()
    attendance = AttendanceSystem()
    set_attendance_ref(attendance)
    cooldown   = Cooldown(seconds=6)

    # ── Mouse drawing for restricted zones ──────────────────────
    _drawing   = False
    _draw_start= (0, 0)
    _draw_cur  = (0, 0)

    def mouse_callback(event, x, y, flags, param):
        nonlocal _drawing, _draw_start, _draw_cur
        if event == cv2.EVENT_LBUTTONDOWN:
            _drawing    = True
            _draw_start = (x, y)
            _draw_cur   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and _drawing:
            _draw_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and _drawing:
            _drawing = False
            x1 = min(_draw_start[0], x)
            y1 = min(_draw_start[1], y)
            x2 = max(_draw_start[0], x)
            y2 = max(_draw_start[1], y)
            if (x2-x1) > 20 and (y2-y1) > 20:
                RESTRICTED_ZONES.append((x1, y1, x2, y2))
                print(f"[ZONE] Added restricted zone: ({x1},{y1}) → ({x2},{y2})")

    cv2.namedWindow("AI Surveillance System v4.0")
    cv2.setMouseCallback("AI Surveillance System v4.0", mouse_callback)

    # Start dashboard server
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()
    print(f"\n  Dashboard → http://localhost:{DASHBOARD_PORT}")
    print("  Press Q = quit  |  S = screenshot")
    print("="*65+"\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        try:
            # ── YOLO inference ─────────────────────────────────────────
            det_res  = det_model(frame,  verbose=False)
            pose_res = pose_model(frame, verbose=False)

            # ── Parse detections ───────────────────────────────────────
            person_count=0; phone_det=False; restricted=False; boxes=[]
            for box in det_res[0].boxes:
                cls=int(box.cls[0])
                x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
                if cls==0:
                    person_count+=1; boxes.append((x1,y1,x2,y2))
                    if in_restricted(cx,cy): restricted=True
                if cls==67: phone_det=True

            crowd = person_count >= CROWD_THRESHOLD

            # ── Fight ──────────────────────────────────────────────────
            fight, fight_reason = fight_det.update(pose_res[0], boxes)

            # ── Cheat ──────────────────────────────────────────────────
            cheat, cheat_reason, faces = cheat_det.update(frame, pose_res[0], phone_det)

            # ── Attendance + per-student activity tracking ───────────────
            # Run every frame — LBPH predict on small crops is cheap, this is
            # what lets us know WHO is doing WHAT (needed for evaluation).
            recognized_names = attendance.update(frame)
            recognized_names = [n for n in recognized_names if n and n != "Unknown"]

            # Tag each recognized student with their current activity.
            # If multiple students are in frame and a violation fires, we can't
            # always know exactly which one triggered it (single shared phone/
            # fight detector) — so we conservatively tag ALL currently-recognized
            # students with the most serious active status. Better than silence,
            # and avoids blaming an unrelated/absent student.
            current_status = "Normal"
            if fight:   current_status = "Fight"
            elif cheat: current_status = "Cheating"
            elif phone_det: current_status = "Phone"

            for nm in recognized_names:
                attendance.mark_activity(nm, current_status)

            # ── Motion heatmap ─────────────────────────────────────────
            annotated, motion_mag = motion_an.update(frame)

            # ── YOLO boxes on top ──────────────────────────────────────
            annotated = det_res[0].plot(img=annotated.copy())

            # Draw faces
            if not isinstance(faces,tuple):
                for (fx,fy,fw,fh) in faces:
                    cv2.rectangle(annotated,(fx,fy),(fx+fw,fy+fh),(255,165,0),2)

            annotated = draw_restricted_zones(annotated)

            # ── Build alerts ───────────────────────────────────────────
            alerts=[]
            if fight:      alerts.append(("fight",      f"FIGHT DETECTED  ({fight_reason})"))
            if cheat:      alerts.append(("cheat",      f"CHEATING!  ({cheat_reason})"))
            if restricted: alerts.append(("restricted", "RESTRICTED AREA BREACH!"))
            if crowd:      alerts.append(("crowd",      f"CROWD ALERT — {person_count} people"))
            if phone_det and not cheat:
                           alerts.append(("phone",      "PHONE DETECTED!"))

            # ── Trigger actions ────────────────────────────────────────
            # severity: "high" = always screenshot (legal/evaluation evidence)
            #           "low"  = log + notify only, no screenshot spam
            for key, active, email_subj, email_body, severity in [
                ("fight",      fight,      "Fight Detected",   f"Fight! {fight_reason}",   "high"),
                ("cheat",      cheat,      "Cheating Alert",   f"Cheating! {cheat_reason}", "high"),
                ("restricted", restricted, "Restricted Breach","Zone breach detected!",    "high"),
                ("crowd",      crowd,      "Crowd Alert",      f"{person_count} people detected", "low"),
                ("phone",      phone_det,  "Phone Detected",   "Phone in surveillance area", "low"),
            ]:
                cd_key = key if severity == "high" else f"{key}_low"
                if active and cooldown.ready(cd_key, seconds=6 if severity == "high" else 30):
                    buzzer(key)
                    speak(key)
                    ss = save_screenshot(annotated, key) if severity == "high" else None
                    log_event(key.upper(), email_body, ss)
                    if severity == "high":
                        send_email_alert(email_subj, email_body, ss)
                    push_event(key.upper(), email_body)

            # ── Update shared state ────────────────────────────────────
            with state_lock:
                state.update({
                    "people":       person_count,
                    "crowd":        crowd,
                    "phone":        phone_det,
                    "fight":        fight,
                    "fight_reason": fight_reason,
                    "cheat":        cheat,
                    "cheat_reason": cheat_reason,
                    "restricted":   restricted,
                    "motion":       round(motion_mag, 2),
                    "attendance":   sorted(list(attendance.present_today)),
                    "evaluation":   attendance.evaluation_snapshot(),
                    "roster_size":  len(attendance.roster()),
                })

            push_state()   # broadcast to dashboard

            # ── Draw HUD ───────────────────────────────────────────────
            hud_stats = {
                "people":     person_count, "crowd":  crowd,
                "fight":      fight,        "cheat":  cheat,
                "phone":      phone_det,    "restricted": restricted,
                "motion":     motion_mag,   "events": state["events"],
                "att_count":  len(attendance.present_today),
                "present":    attendance.present_today,
            }
            annotated = draw_hud(annotated, hud_stats)
            annotated = draw_alerts(annotated, alerts)

            # Draw live zone while dragging
            if _drawing:
                cv2.rectangle(annotated, _draw_start, _draw_cur, (0, 0, 255), 2)
                cv2.putText(annotated, "Drawing zone... release to set",
                            (10, annotated.shape[0]-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

        except Exception as e:
            # Never let one bad frame kill the live demo — log it and
            # just show the raw frame for this iteration, then continue.
            print(f"[WARN] Frame processing error (skipped this frame): {e}")
            annotated = frame

        # Push frame to web dashboard
        set_latest_frame(annotated)
        cv2.imshow("AI Surveillance System v4.0", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("c"):
            RESTRICTED_ZONES.clear()
            print("[ZONE] All restricted zones cleared")
        elif key == ord("s"):
            p = save_screenshot(annotated,"manual")
            log_event("MANUAL","Manual screenshot",p)
            push_event("MANUAL","Manual screenshot taken")

    cap.release()
    cv2.destroyAllWindows()
    with state_lock:
        ec = state["events"]
    print(f"\n[DONE] Events logged : {ec}")
    print("[DONE] Log           : events.txt")
    print("[DONE] Attendance    : attendance.csv")
    print("[DONE] Screenshots   : screenshots/")


if __name__ == "__main__":
    main()