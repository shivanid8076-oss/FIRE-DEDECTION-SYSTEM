# TRINETRA AEGIS: Forest Fire & Smoke Detection System 🛡️🔥

**TRINETRA AEGIS** ("Three-Eyed Shield") is an advanced, AI-powered command center designed for real-time forest fire detection, smoke analysis, and tactical response coordination. This system integrates ground-level computer vision with orbital satellite telemetry to provide a comprehensive situational awareness platform for emergency responders.

---

## 🚀 Key Features

### 1. 📹 Real-Time Threat Detection
- **Multi-Source Input:** Supports Webcam, IP Cameras, Drone Streams, and Video Uploads.
- **AI Vision:** Utilizes `YOLOv8` (and experimental `YOLO11`) models to detect **Fire**, **Smoke**, and **Human Presence** in real-time.
- **Ensemble Detection:** Combines multiple detection strategies for high confidence and reduced false positives.
- **Thermal Vision Mode:** Simulated thermal imaging overlay for better visualization in low-visibility conditions.
- **CLAHE Enhancement:** Adaptive histogram equalization to improve visibility in smoky footage.

### 2. 🛰️ Global Satellite Surveillance
- **NASA FIRMS Integration:** Fetches near real-time fire data from **MODIS** and **VIIRS** satellites.
- **Heatmap Visualization:** Interactive global map showing active thermal anomalies.
- **Proximity Alerts:** Automatically alerts command if satellite data indicates fires within a configurable radius (e.g., 50km).

### 3. 🧠 AI Tactical Advisor (GenAI)
- **Incident Commander:** Powered by **OpenAI GPT-4o**, the system analyzes visual data and environmental metrics to generate instant tactical response plans.
- **Visual Intelligence (VQA):** Operators can ask natural language questions about the live video feed (e.g., *"Is the smoke black or white?"*, *"Are there civilians near the fire?"*).
- **Mission Reporting:** Automatically generates professional "After-Action Reports" summarizing the detection session.

### 4. 🚨 Automated Alert System
- **Multi-Channel Notifications:**
    - **SMS Alerts:** Immediate alerts via **Twilio** with location coordinates.
    - **Discord Integration:** Posts rich embeds with detection details to emergency channels.
    - **Voice Announcer:** Browser-based text-to-speech warnings for the command center.
- **Evidence Locker:** Automatically records video footage of detected threats and generates incident reports.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (Accessible, responsive dashboard)
- **Computer Vision:** [OpenCV](https://opencv.org/), [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Geospatial:** [Folium](https://python-visualization.github.io/folium/), [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium)
- **GenAI:** [OpenAI API](https://openai.com/) (GPT-4o)
- **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Alerts:** [Twilio](https://www.twilio.com/), [Requests](https://pypi.org/project/requests/)

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shivanid8076-oss/dedectionsystem.git
   cd dedectionsystem
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create a `.streamlit/secrets.toml` file to enable AI and Alert features:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "sk-..."
   TWILIO_SID = "AC..."
   TWILIO_AUTH = "..."
   TWILIO_FROM = "+1..."
   TO_PHONE = "+91..."
   DISCORD_WEBHOOK = "https://discord.com/api/webhooks/..."
   ```

4. **Run the Application**
   ```bash
   streamlit run aap.py
   ```

---

## 📂 Project Structure

- `aap.py`: Main application entry point containing the Streamlit dashboard and logic.
- `best.pt`, `yolov8n.pt`: Trained YOLO models for fire detection.
- `requirements.txt`: List of Python dependencies.
- `evidence_locker/`: Folder where recorded incidents and reports are saved.
- `logs/`: System logs for tracking detections and performance.

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
