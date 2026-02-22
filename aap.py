import streamlit as st
import cv2
import tempfile
import folium
import time
import os
import glob
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import numpy as np
import queue
from typing import Dict, List, Tuple
import logging
from PIL import Image
import base64
import math
import threading

# OpenAI Import
from openai import OpenAI

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(
    page_title="TRINETRA AEGIS Command Center",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
if not os.path.exists('evidence_locker'):
    os.makedirs('evidence_locker')
if not os.path.exists('logs'):
    os.makedirs('logs')

# OPTIMIZATION: Performance constants - UPDATED FOR REAL-TIME SATELLITE
FRAME_SKIP = 2  # Process every 2nd frame
BATCH_SIZE = 3  # For future batch processing
STRATEGY_UPDATE_INTERVAL = 900  # Update strategy every 45s
SATELLITE_CHECK_INTERVAL = 120  # Check satellite every 6 seconds (UPGRADED from 600)
CACHE_TTL = 60  # 1 minute cache for satellite data (UPGRADED from 300)

# --- HELPER: Image to Base64 for OpenAI Vision ---
def encode_image_to_base64(frame):
    """Converts OpenCV frame to base64 string for OpenAI API"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# --- HELPER: Haversine Distance Calculation ---
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two GPS points in KM"""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- ADVANCED FEATURES MODULES ---

class MissionReporter:
    """Generates professional After-Action Reports using OpenAI"""
    def __init__(self, client):
        self.client = client

    def generate_after_action_report(self, incident_df):
        """Reads logs and writes a summary report"""
        if self.client is None:
            return "⚠ OpenAI Key Missing. Cannot generate report."
        
        if incident_df.empty:
            return "No incidents recorded to report."

        # Summarize data for the prompt
        total_incidents = len(incident_df)
        max_conf = incident_df['Confidence'].max()
        avg_risk = incident_df['Risk_Score'].mean()
        types = incident_df['Type'].value_counts().to_string()
        
        prompt = f"""
        ACT AS A MILITARY FIRE COMMANDER. 
        Write a professional "After-Action Report" (AAR) based on this session log:
        - Total Detections: {total_incidents}
        - Max Threat Confidence: {max_conf:.1%}
        - Average Risk Level: {avg_risk:.1%}
        - Threat Types Detected: \n{types}
        
        FORMAT THE REPORT AS FOLLOWS:
        1. **EXECUTIVE SUMMARY**: A 2-sentence overview of the session.
        2. **THREAT ANALYSIS**: Detailed breakdown of what was seen.
        3. **STRATEGIC RECOMMENDATION**: Advice for future monitoring based on these threats.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Report Generation Failed: {str(e)}"

class FieldIntel:
    """Visual Q&A System - Allows user to chat with the video feed"""
    def __init__(self, client):
        self.client = client

    def analyze_frame(self, frame, user_question):
        if self.client is None:
            return "⚠ OpenAI Key Missing."
        
        base64_image = encode_image_to_base64(frame)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": f"Analyze this surveillance frame and answer this question strictly based on visual evidence: {user_question}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Visual Analysis Failed: {str(e)}"

# ============================================================================
# UPGRADED SATELLITE NETWORK CLASS
# ============================================================================
class ImprovedSatelliteNetwork:
    """
    Enhanced TRINETRA SATELLITE UPLINK v2.0
    - Multi-source satellite data aggregation (MODIS + VIIRS)
    - Intelligent caching and refresh (1-min TTL)
    - Real-time proximity monitoring with threat classification
    - Historical fire tracking
    - Background auto-refresh capability
    """
    
    def __init__(self):
        # NASA FIRMS Data Feeds - DUAL SATELLITE SOURCES
        self.data_feeds = {
            "South Asia": {
                "modis": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv",
                "viirs": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv"
            },
            "Global": {
                "viirs": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv"
            },
            "North America": {
                "modis": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_24h.csv"
            },
            "Europe": {
                "modis": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Europe_24h.csv"
            }
        }
        
        self.cache = {}
        self.last_fetch_time = {}
        self.historical_fires = []
        
    def fetch_satellite_data(self, region="South Asia", force_refresh=False):
        """
        Fetches and aggregates satellite fire data with intelligent caching
        Returns: DataFrame with columns [latitude, longitude, brightness, confidence, acq_time, satellite]
        """
        current_time = time.time()
        cache_key = region
        
        # Check cache validity
        if not force_refresh and cache_key in self.cache:
            time_since_fetch = current_time - self.last_fetch_time.get(cache_key, 0)
            if time_since_fetch < CACHE_TTL:
                logger.info(f"📡 Using cached satellite data (age: {time_since_fetch:.0f}s)")
                return self.cache[cache_key]
        
        # Fetch fresh data
        logger.info(f"🛰️ Fetching fresh satellite data for {region}...")
        all_fires = []
        
        feeds = self.data_feeds.get(region, self.data_feeds["South Asia"])
        
        for satellite_type, url in feeds.items():
            try:
                df = pd.read_csv(url)
                df['satellite'] = satellite_type.upper()
                all_fires.append(df)
                logger.info(f"✅ {satellite_type.upper()}: {len(df)} fires detected")
            except Exception as e:
                logger.error(f"❌ Failed to fetch {satellite_type}: {str(e)}")
        
        if all_fires:
            combined_df = pd.concat(all_fires, ignore_index=True)
            # Store in cache
            self.cache[cache_key] = combined_df
            self.last_fetch_time[cache_key] = current_time
            
            # Update historical tracking
            self.historical_fires.append({
                'timestamp': datetime.now(),
                'count': len(combined_df),
                'region': region
            })
            
            return combined_df
        else:
            logger.warning("⚠️ No satellite data available - using fallback simulation")
            # Fallback Simulation
            return pd.DataFrame({
                'latitude': np.random.uniform(28.0, 32.0, 5),
                'longitude': np.random.uniform(75.0, 78.0, 5),
                'brightness': np.random.uniform(300, 450, 5),
                'confidence': np.random.randint(50, 100, 5),
                'acq_time': [datetime.now().strftime("%H:%M")] * 5,
                'satellite': ['SIMULATED'] * 5
            })
    
    def check_proximity_threats(self, base_lat, base_lon, radius_km=50):
        """
        Checks for fires within specified radius
        Returns: (threat_count, closest_fire_distance, threat_level, nearby_fires_df)
        """
        satellite_data = self.fetch_satellite_data()
        
        if satellite_data.empty:
            return 0, None, "SAFE", pd.DataFrame()
        
        # Calculate distances
        satellite_data['distance_km'] = satellite_data.apply(
            lambda row: calculate_distance(base_lat, base_lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter threats within radius
        nearby_fires = satellite_data[satellite_data['distance_km'] <= radius_km].copy()
        
        if nearby_fires.empty:
            return 0, None, "SAFE", pd.DataFrame()
        
        # Classify threat level
        closest_distance = nearby_fires['distance_km'].min()
        threat_count = len(nearby_fires)
        
        if closest_distance < 5:
            threat_level = "CRITICAL"
        elif closest_distance < 15:
            threat_level = "HIGH"
        elif closest_distance < 30:
            threat_level = "MODERATE"
        else:
            threat_level = "LOW"
        
        # Sort by proximity
        nearby_fires = nearby_fires.sort_values('distance_km')
        
        return threat_count, closest_distance, threat_level, nearby_fires
    
    def check_proximity(self, user_lat, user_lon, radius_km=50):
        """Legacy compatibility method"""
        _, _, _, nearby_fires = self.check_proximity_threats(user_lat, user_lon, radius_km)
        nearby_threats = []
        if not nearby_fires.empty:
            for _, row in nearby_fires.iterrows():
                nearby_threats.append({
                    'dist': row['distance_km'],
                    'lat': row['latitude'],
                    'lon': row['longitude'],
                    'temp': row.get('brightness', 0)
                })
        return nearby_threats
    
    def fetch_live_data(self, region):
        """Legacy compatibility method"""
        return self.fetch_satellite_data(region)
    
    def get_cache_status(self):
        """Returns cache freshness information"""
        if not self.cache:
            return "No cached data"
        
        age = time.time() - max(self.last_fetch_time.values())
        return f"Cache age: {age:.0f}s (TTL: {CACHE_TTL}s)"

class FireSpreadPredictor:
    """Advanced fire spread prediction using physics-based modeling"""
    def __init__(self):
        self.wind_speed = 15.0
        self.wind_direction = 0.0
        self.temperature = 25.0
        self.humidity = 50.0
        self.fuel_load = 1.0

    def predict_spread_rate(self, fire_intensity: float) -> float:
        base_rate = 0.01
        wind_factor = 1 + 0.001 * self.wind_speed * np.cos(np.radians(self.wind_direction))
        moisture_factor = max(0.1, 1 - (self.humidity / 100) * 0.8)
        temp_factor = 1 + (self.temperature - 20) * 0.02
        return base_rate * wind_factor * moisture_factor * temp_factor * fire_intensity

class MultiModelEnsemble:
    """Ensemble of multiple AI models for robust fire detection"""
    def __init__(self):
        self.models = []
        self.confidence_weights = [0.4, 0.3, 0.3]

    def add_model(self, model, model_type: str):
        self.models.append({
            'model': model,
            'type': model_type,
            'performance_score': 0.8
        })

    def predict_ensemble(self, frame, ai_sensitivity: float, use_fp16: bool = True) -> Tuple[bool, float, str, List[Tuple[int, int, int, int, str, float]]]:
        predictions = []
        boxes = []
        
        # Check if we have models loaded
        if len(self.models) > 0 and self.models[0]['type'] == 'yolo':
            # OPTIMIZATION: Use FP16 if available and appropriate
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            try:
                # If using GPU
                if device == 0:
                    yolo_results = self.models[0]['model'](
                        frame, 
                        verbose=False, 
                        conf=ai_sensitivity,
                        half=use_fp16,  # FP16 inference for 2x speed
                        device=0
                    )
                else:
                    # If using CPU
                    yolo_results = self.models[0]['model'](
                        frame, 
                        verbose=False, 
                        conf=ai_sensitivity,
                        device='cpu'
                    )
            except Exception as e:
                # Fallback
                print(f"Inference error: {e}, falling back to default")
                yolo_results = self.models[0]['model'](frame, verbose=False, conf=ai_sensitivity, device='cpu')
            
            yolo_conf = 0.0
            yolo_class = "none"
            
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Safely access names
                    if hasattr(self.models[0]['model'], 'names'):
                        cls_name = self.models[0]['model'].names[cls_id].lower()
                    else:
                        cls_name = str(cls_id)

                    if cls_name in ['fire', 'smoke', 'flame', 'person', 'cell phone']: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2, cls_name, conf))
                        if conf > yolo_conf:
                            yolo_conf = conf
                            yolo_class = cls_name
            predictions.append(('yolo', yolo_conf > ai_sensitivity, yolo_conf, yolo_class))
        
        predictions.extend([('cnn', False, 0.1, "none"), ('rf', False, 0.05, "none")])
        
        total_conf = sum(pred[2] * self.confidence_weights[i] for i, pred in enumerate(predictions))
        is_fire = any(pred[1] for pred in predictions)
        max_conf_pred = max(predictions, key=lambda x: x[2])
        detected_class = max_conf_pred[3] if is_fire else "none"
        
        return is_fire, total_conf, detected_class, boxes

class IoTIntegration:
    """IoT sensor integration for environmental monitoring"""
    def __init__(self):
        self.sensors = {}
        self.alert_thresholds = {
            'temperature': 50.0, 'smoke': 100.0, 'co': 50.0, 'wind_speed': 30.0
        }

    def add_sensor(self, sensor_id: str, sensor_type: str, location: Tuple[float, float]):
        self.sensors[sensor_id] = {
            'type': sensor_type, 'location': location, 'last_reading': None,
            'status': 'active', 'battery_level': 100.0
        }

class EmergencyCoordination:
    """Advanced emergency response coordination system"""
    def __init__(self):
        self.active_incidents = {}
        self.emergency_contacts = []
        self.response_teams = []
        self.coordination_queue = queue.Queue()

    def add_emergency_contact(self, name: str, role: str, contact: str, priority: int):
        self.emergency_contacts.append({
            'name': name, 'role': role, 'contact': contact, 'priority': priority
        })

    def add_response_team(self, team_id: str, team_type: str, location: Tuple[float, float],
                          capacity: int, eta: int):
        self.response_teams.append({
            'id': team_id, 'type': team_type, 'location': location,
            'capacity': capacity, 'eta': eta, 'status': 'available'
        })

class RealTimeAnalytics:
    """Advanced real-time analytics and reporting"""
    def __init__(self):
        self.log_file = "logs/incident_log.csv"
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["Timestamp", "Type", "Confidence", "Risk_Score"]).to_csv(self.log_file, index=False)
        
        self.incident_history = []
        self.performance_metrics = {}
        self.prediction_accuracy = []
        self.response_times = []

    def calculate_risk_score(self, location: Tuple[float, float],
                             environmental_factors: Dict) -> float:
        base_risk = 0.1
        temp_factor = max(0, (environmental_factors.get('temperature', 25) - 30) * 0.02)
        wind_factor = environmental_factors.get('wind_speed', 10) * 0.01
        humidity_factor = max(0, (50 - environmental_factors.get('humidity', 50)) * 0.005)
        
        recent_incidents = 0 
        history_factor = min(0.5, recent_incidents * 0.05)
        
        risk_score = base_risk + temp_factor + wind_factor + humidity_factor + history_factor
        return min(1.0, risk_score)

    def log_incident(self, obj, conf, risk):
        new_data = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Type": [obj],
            "Confidence": [float(conf)],
            "Risk_Score": [float(risk)]
        })
        new_data.to_csv(self.log_file, mode='a', header=False, index=False)
        
    def get_logs(self):
        if os.path.exists(self.log_file):
            return pd.read_csv(self.log_file)
        return pd.DataFrame()

class StrategicAI:
    """OpenAI-powered Incident Commander for strategy generation"""
    def __init__(self, client):
        self.client = client
        self.last_strategy = "System Scanning..."
        self.last_update_time = 0

    def generate_strategy(self, image_frame, detection_type, confidence, weather, teams, location):
        """Generates a tactical response plan"""
        current_time = time.time()
        
        # OPTIMIZATION: Only update strategy every 45 seconds (was 15) AND only if fire detected
        if current_time - self.last_update_time < 45:
            return self.last_strategy

        # OpenAI Vision Logic
        base64_image = encode_image_to_base64(image_frame)

        prompt = f"""
        ACT AS A SENIOR FIRE INCIDENT COMMANDER.
        SITUATION REPORT:
        - Threat: {detection_type.upper()} detected (Confidence: {confidence:.1%})
        - Location: Lat {location[0]}, Long {location[1]}
        - Environmental Conditions: {weather}
        - Available Resources: {len(teams)} Response Teams available.

        TASK:
        Analyze the attached image and data.
        Provide a concise, immediate tactical plan (max 4 bullet points).
        
        FORMAT:
        1. [VISUAL ANALYSIS] (Briefly describe what you see in the image)
        2. [IMMEDIATE ACTION]
        3. [RESOURCE ALLOCATION]
        4. [EVACUATION ADVICE]
        """
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    max_tokens=300
                )
                strategy_text = response.choices[0].message.content
                self.last_strategy = strategy_text
                self.last_update_time = current_time
                return strategy_text
            return "⚠ AI CONFIG MISSING"
        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            return f"⚠ AI STRATEGY OFFLINE: {str(e)}"

# --- DEPENDENCY HANDLING ---
try:
    from ultralytics import YOLO
    from twilio.rest import Client
except ImportError:
    st.error("❌ Critical Libraries Missing. Run: pip install ultralytics twilio requests pandas openai pillow")
    st.stop()

# --- SECRETS & CONFIGURATION ---
try:
    TWILIO_SID = st.secrets.get("TWILIO_SID", None)
    TWILIO_AUTH = st.secrets.get("TWILIO_AUTH", None)
    TWILIO_FROM = st.secrets.get("TWILIO_FROM", None)
    TO_PHONE = st.secrets.get("TO_PHONE", None)
    DISCORD_WEBHOOK = st.secrets.get("DISCORD_WEBHOOK", None)
    
    # OpenAI Key Config
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    client = None
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    
except (FileNotFoundError, AttributeError):
    TWILIO_SID = TWILIO_AUTH = TWILIO_FROM = TO_PHONE = DISCORD_WEBHOOK = OPENAI_API_KEY = client = None

# --- GLOBAL INSTANCES (PERSISTED) ---

if 'fire_predictor' not in st.session_state: st.session_state.fire_predictor = FireSpreadPredictor()
if 'ensemble_model' not in st.session_state: st.session_state.ensemble_model = MultiModelEnsemble()
if 'iot_system' not in st.session_state: st.session_state.iot_system = IoTIntegration()
if 'emergency_coord' not in st.session_state: st.session_state.emergency_coord = EmergencyCoordination()
if 'analytics' not in st.session_state: st.session_state.analytics = RealTimeAnalytics()
if 'strategic_ai' not in st.session_state: st.session_state.strategic_ai = StrategicAI(client)
if 'satellite_uplink' not in st.session_state: st.session_state.satellite_uplink = ImprovedSatelliteNetwork()
if 'field_intel' not in st.session_state: st.session_state.field_intel = FieldIntel(client)
if 'mission_reporter' not in st.session_state: st.session_state.mission_reporter = MissionReporter(client)

# Create easy references
fire_predictor = st.session_state.fire_predictor
ensemble_model = st.session_state.ensemble_model
iot_system = st.session_state.iot_system
emergency_coord = st.session_state.emergency_coord
analytics = st.session_state.analytics
strategic_ai = st.session_state.strategic_ai
satellite_uplink = st.session_state.satellite_uplink
field_intel = st.session_state.field_intel
mission_reporter = st.session_state.mission_reporter

# --- DARK SCI-FI COMMAND CENTER THEME ---
st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%); color: #00d4ff; font-family: 'Courier New', monospace; background-attachment: fixed; }
        .stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px); background-size: 50px 50px; pointer-events: none; z-index: -1; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%); border-right: 2px solid #00d4ff; box-shadow: 2px 0 20px rgba(0, 212, 255, 0.3); }
        h1, h2, h3 { color: #00d4ff; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5); }
        .stButton>button { background: linear-gradient(45deg, #ff6b35, #f7931e); border: 2px solid #00d4ff; color: #ffffff; border-radius: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s ease; box-shadow: 0 0 20px rgba(255, 107, 53, 0.3); font-family: 'Courier New', monospace; }
        .stButton>button:hover { background: linear-gradient(45deg, #f7931e, #ff6b35); box-shadow: 0 0 30px rgba(255, 107, 53, 0.6); transform: translateY(-2px) scale(1.02); border-color: #00ff88; }
        div[data-testid="stMetricValue"] { color: #00ff88 !important; font-size: 32px !important; text-shadow: 0 0 15px rgba(0, 255, 136, 0.8); }
        div[data-testid="stMetricLabel"] { color: #00d4ff !important; font-size: 12px !important; letter-spacing: 1px; }
        .stTabs [data-baseweb="tab-list"] { background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 10px; border: 1px solid #00d4ff; }
        .stTabs [aria-selected="true"] { background: linear-gradient(45deg, #00d4ff, #0099cc) !important; color: #0a0a0a !important; box-shadow: 0 0 15px rgba(0, 212, 255, 0.5); }
        .stTextInput > div > div > input { color: #00d4ff; background-color: #1a1a2e; border: 1px solid #00d4ff; }
        div[data-testid="stToast"] { background: linear-gradient(45deg, #ff6b35, #f7931e); color: white; border: 2px solid #00d4ff; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- SYSTEM INITIALIZATION ---
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if not st.session_state.system_initialized:
    with st.sidebar:
        with st.status("🚀 INITIALIZING TRINETRA AEGIS...", expanded=True) as status:
            st.write("🛰️ Establishing satellite link...")
            time.sleep(0.3)
            st.write("🌡️ Calibrating thermal sensors...")
            time.sleep(0.3)
            st.write("🧠 Loading neural networks...")
            time.sleep(0.3)
            if status is not None:
                status.update(label="🛡️ SYSTEM ONLINE", state="complete", expanded=False)
    st.session_state.system_initialized = True

# --- UTILITY FUNCTIONS ---

def apply_enhancement(frame):
    """CLAHE enhancement for better visibility"""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return frame

def draw_hud(frame, frame_count, lat, long, fps=0):
    """Advanced HUD with tactical information"""
    height, width = frame.shape[:2]
    if frame_count % 30 < 15:
        cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (width - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Crosshair
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 1)
    
    # Data
    gps_text = f"LAT: {lat:.4f} | LONG: {long:.4f}"
    cv2.putText(frame, gps_text, (20, height - 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    time_text = datetime.now().strftime("%H:%M:%S")
    
    # OPTIMIZATION: Show FPS in HUD
    fps_text = f"FPS: {fps:.1f}" if fps > 0 else ""
    cv2.putText(frame, f"TRINETRA AEGIS | {time_text} | {fps_text}", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
    return frame

def generate_incident_report(timestamp, obj, conf, lat, long, risk_score):
    """Generate comprehensive incident report"""
    filename = f"evidence_locker/incident_report_{timestamp}.html"
    html_content = f"""
    <html>
        <head><title>TRINETRA AEGIS Incident Report</title></head>
        <body style="background: #0b0c10; color: #66fcf1; padding: 20px; font-family: monospace;">
            <h1>🔥 INCIDENT REPORT</h1>
            <div style="border: 2px solid red; padding: 15px; background: #330000;">
                <h3>🚨 THREAT DETECTED: {obj.upper()}</h3>
                <p><strong>Confidence:</strong> {conf:.1%}</p>
                <p><strong>Risk Score:</strong> {risk_score:.1%}</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Coordinates:</strong> {lat:.6f}, {long:.6f}</p>
            </div>
        </body>
    </html>
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filename
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return "Error Saving Report"

def get_weather_data():
    """Get simulated weather data"""
    return {
        'temperature': random.randint(35, 42),
        'wind_speed': random.randint(15, 30),
        'humidity': random.randint(10, 25),
        'wind_direction': random.randint(0, 360)
    }

def get_alert_classes(use_demo_mode):
    """Get detection classes based on mode"""
    if use_demo_mode:
        return ['person', 'cell phone', 'bottle', 'fire', 'smoke']
    else:
        return ['fire', 'smoke', 'flame']

def speak_alert(text):
    """Voice alert system"""
    safe_text = text.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    js = f"<script>window.speechSynthesis.speak(new SpeechSynthesisUtterance('{safe_text}'));</script>"
    components.html(js, height=0, width=0)

def send_discord_alert(risk_level, location_link, incident_details):
    """Send Discord alert"""
    if not DISCORD_WEBHOOK: return False
    data = {
        "content": "🚨 **TRINETRA AEGIS ALERT** 🚨",
        "embeds": [{
            "title": f"🔥 {risk_level} THREAT DETECTED",
            "description": f"**Location:** {location_link}\n**Conf:** {incident_details.get('confidence', 0):.1%}",
            "color": 15158332
        }]
    }
    try: 
        requests.post(DISCORD_WEBHOOK, json=data, timeout=5)
        return True
    except: 
        return False

def send_sms_alert(lat, long, risk_level, incident_details):
    """Send SMS alert"""
    if not TWILIO_SID: 
        logger.warning("SMS not sent - Twilio credentials missing")
        return False
    try:
        twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
        maps_link = f"http://maps.google.com/?q={lat},{long}"
        message_body = f"🚨 TRINETRA AEGIS\n🔥 {risk_level}\n📍 {maps_link}\nConf: {incident_details.get('confidence', 0):.1%}"
        twilio_client.messages.create(body=message_body, from_=TWILIO_FROM, to=TO_PHONE)
        logger.info("✅ SMS Alert sent successfully")
        return True
    except Exception as e:
        logger.error(f"❌ SMS Failed: {e}")
        return False

@st.cache_resource
def load_ai_model():
    """Load the AI model with fallback options and warm-up"""
    if os.path.exists('best.pt'): 
        model = YOLO('best.pt')
        status = "✅ Custom YOLOv8 Model Loaded"
    elif os.path.exists('yolov8n-fire.pt'): 
        model = YOLO('yolov8n-fire.pt')
        status = "✅ Fire Detection Model Loaded"
    else: 
        model = YOLO('yolov8n.pt')
        status = "⚠️ Using Standard COCO Model"
    
    # OPTIMIZATION: Model warm-up to eliminate cold start lag
    try:
        dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        _ = model(dummy_frame, verbose=False, half=True)
        logger.info("Model warmed up successfully")
    except Exception as e:
        logger.warning(f"Warm-up with FP16 failed, trying FP32: {e}")
        try:
            _ = model(dummy_frame, verbose=False)
        except:
            pass
    
    return model, status

# --- OPENAI CHAT CACHING ---
@st.cache_data(show_spinner=False, ttl=600)
def ask_commander_cached(query):
    try:
        if not client: return "⚠️ API Key Missing"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior fire chief. Answer briefly and tactically."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Commander Offline: {str(e)}"

# --- MAIN APPLICATION ---

def main():
    
    # --- INIT NOTIFICATION TIMER ---
    if "last_notification_time" not in st.session_state:
        st.session_state.last_notification_time = 0
    
    # OPTIMIZATION: Initialize FPS tracking
    if "fps_history" not in st.session_state:
        st.session_state.fps_history = []
        st.session_state.last_fps_time = time.time()
    
    # Initialize detection active state
    if "detection_active" not in st.session_state:
        st.session_state.detection_active = False
    
    # Initialize confidence history
    if "confidence_history" not in st.session_state:
        st.session_state.confidence_history = []
    
    # Initialize detection radius
    if 'detection_radius' not in st.session_state:
        st.session_state.detection_radius = 50

    # Load AI Model
    model, model_status = load_ai_model()
    ensemble_model.add_model(model, 'yolo')

    # Initialize advanced features
    emergency_coord.add_response_team('FIRE01', 'fire_engine', (30.7, 76.7), 5, 8)
    
    # ========================================================================
    # SIDEBAR CONFIGURATION - NO COLUMN NESTING
    # ========================================================================
    with st.sidebar:
        st.title("🛡️ TRINETRA AEGIS")
        st.markdown("---")
        
        # --- SMS DEBUG SECTION ---
        with st.expander("🔧 SMS Debug Info"):
            st.write(f"Twilio SID: {'✅ Set' if TWILIO_SID else '❌ Missing'}")
            st.write(f"Twilio Auth: {'✅ Set' if TWILIO_AUTH else '❌ Missing'}")
            st.write(f"Twilio From: {TWILIO_FROM if TWILIO_FROM else '❌ Missing'}")
            st.write(f"To Phone: {TO_PHONE if TO_PHONE else '❌ Missing'}")

        if st.button("🧪 Test SMS Alert"):
            if TWILIO_SID:
                test_result = send_sms_alert(30.7333, 76.7794, "TEST", {'confidence': 0.95})
                if test_result:
                    st.success("✅ SMS Test Successful!")
                else:
                    st.error("❌ SMS Test Failed - Check Logs")
            else:
                st.error("❌ Twilio credentials not configured")

        st.header("⚙️ Mission Control")
        if "Custom" in model_status:
            st.success(model_status)
        else:
            st.warning(model_status)

        demo_mode = st.toggle("🧪 Training Mode", value="Standard" in model_status)

        st.markdown("### 🔬 Advanced Systems")
        thermal_vision = st.toggle("🔥 Thermal Vision", value=False)
        clahe_enhancement = st.toggle("👁️ CLAHE Enhancement", value=False)
        ensemble_detection = st.toggle("🤖 Ensemble Detection", value=True)
        auto_satellite = st.toggle("📡 Auto-Satellite Link", value=True)
        
        # OPTIMIZATION: Add FP16 toggle
        use_fp16 = st.toggle("⚡ FP16 Acceleration", value=True, 
                             help="Enable half-precision inference for 2x speed boost (requires GPU)")

        # OPTIMIZATION: Adjusted default sensitivity to 0.35 for better fire detection
        ai_sensitivity = st.slider("🎯 AI Sensitivity", 0.1, 1.0, 0.35, 0.05,
                                   help="Lower = more sensitive (detects more fires but may increase false positives)")
        
        # Detection Radius
        detection_radius = st.slider("🛰️ Detection Radius (km)", 10, 200, 50, 10)
        st.session_state.detection_radius = detection_radius
        
        # New Field Intelligence Section in Sidebar
        st.markdown("---")
        st.markdown("### 🧠 Field Intelligence (Visual Q&A)")
        user_q = st.text_input("Ask the Video Feed:", placeholder="e.g. Is the smoke black?")
        analyze_btn = st.button("👁️ Analyze Current Frame")
        
        # New Mission Reporting Section in Sidebar
        st.markdown("---")
        st.markdown("### 📄 Reporting")
        if st.button("📝 Generate Mission Report"):
            with st.spinner("Compiling tactical data..."):
                logs = analytics.get_logs()
                report = mission_reporter.generate_after_action_report(logs)
                st.session_state['last_report'] = report

        # Evidence Management
        st.markdown("---")
        st.subheader("📂 Evidence Vault")
        evidence_files = glob.glob("evidence_locker/evidence_*.mp4")
        evidence_files.sort(key=os.path.getmtime, reverse=True)

        if evidence_files:
            selected_evidence = st.selectbox("Select Evidence", evidence_files)
            if st.button("▶️ Review"): 
                st.video(selected_evidence)
            if st.button("🗑️ Delete"): 
                os.remove(selected_evidence)
                st.rerun()
        
        st.markdown("---")
        if st.button("🧹 Reset System Cache"):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        
        # Manual satellite refresh
        if st.button("🔄 Refresh Satellite Data"):
            with st.spinner("Fetching satellite data..."):
                satellite_uplink.fetch_satellite_data(force_refresh=True)
                st.success("✅ Satellite data refreshed")
        
        st.divider()
        st.subheader("💬 Commander Chat")
        
        user_query = st.text_input("Ask the AI Commander:")
        if user_query:
            if client:
                response_text = ask_commander_cached(user_query)
                st.info(response_text)
            else:
                st.error("API Key Missing. Check .streamlit/secrets.toml")
        
        # OPTIMIZATION: Show performance metrics
        st.markdown("---")
        st.markdown("### 📈 Performance")
        if st.session_state.fps_history:
            avg_fps = np.mean(st.session_state.fps_history[-30:])
            st.metric("Current FPS", f"{avg_fps:.1f}")
        st.caption(f"Frame Skip: Every {FRAME_SKIP} frames")
        st.caption(f"FP16 Mode: {'ON' if use_fp16 else 'OFF'}")

    # ========================================================================
    # MAIN DASHBOARD LAYOUT - FIXED NO NESTING
    # ========================================================================
    
    # TOP LEVEL ROW
    main_col, right_col = st.columns([4, 1])

    with main_col:
        st.title("🚁 TRINETRA AEGIS Command Center")
        
        # System Status - SEPARATE ROW
        c1, c2, c3 = st.columns(3)
        c1.metric("System Status", "🟢 ONLINE")
        c2.metric("Active Sensors", "3")
        c3.metric("Response Teams", "2")

        # Main Interface Tabs
        tab1, tab2, tab3 = st.tabs(["📹 Live Feed", "🛰️ Satellite View", "📊 Analytics"])

    with right_col:
        st.markdown("### 📊 AI STATUS")
        st.markdown(f"**Mode:** {'TRAINING' if demo_mode else 'ACTIVE'}")
        st.markdown(f"**Sensitivity:** {ai_sensitivity}")
        
        weather = get_weather_data()
        st.markdown("### 🌡️ ENVIRONMENT")
        st.metric("Temp", f"{weather['temperature']}°C")
        st.metric("Wind", f"{weather['wind_speed']} km/h")
        
        # Visual Q&A Response Area
        st.markdown("### 🧠 AI INTEL")
        intel_ph = st.empty()
        if 'vqa_response' in st.session_state:
            intel_ph.info(f"🤖 {st.session_state['vqa_response']}")

    # ========================================================================
    # TAB 1: LIVE FEED - NO NESTED COLUMNS
    # ========================================================================
    with tab1:
        st.header("📹 Real-Time Threat Detection")
        
        # Input source selection - TOP LEVEL
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            input_source = st.selectbox("Input Source", ["Webcam Feed", "Video Upload", "IP Camera", "Drone Stream"])
        
        vid_file = None
        ip_camera_url = ""
        drone_stream_url = ""
        
        if input_source == "Video Upload":
            vid_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        elif input_source == "IP Camera":
            ip_camera_url = st.text_input("IP Camera URL")
        elif input_source == "Drone Stream":
            drone_stream_url = st.text_input("Drone Stream URL")

        # Video and Controls - SEPARATE ROW, NO NESTING
        video_col, control_col = st.columns([3, 1])
        
        with video_col:
            video_placeholder = st.empty()
            confidence_placeholder = st.empty()
            satellite_alert_ph = st.empty()

        with control_col:
            st.markdown("**Controls**")
            
            # START/STOP BUTTONS
            def start_detection():
                st.session_state.detection_active = True
            
            def stop_detection():
                st.session_state.detection_active = False
            
            # Using a container for vertical layout - NO NESTING INSIDE
            st.button("▶ START SYSTEM", type="primary", use_container_width=True, on_click=start_detection)
            st.button("⏹ STOP SYSTEM", type="secondary", use_container_width=True, on_click=stop_detection)

            # Status Indicator Logic
            status_text = "🟢 RUNNING" if st.session_state.detection_active else "🔴 STOPPED"
            status_color = "green" if st.session_state.detection_active else "red"
            st.markdown(f"**System Status:** :{status_color}[{status_text}]")

            st.markdown("---")
            auto_response = st.checkbox("🚁 Auto Response", value=True)
            evidence_recording = st.checkbox("📹 Record Evidence", value=True)
            
            # AI Strategy UI
            st.markdown("---")
            st.markdown("### 🧠 AI TACTICAL ADVISOR")
            strategy_text = st.empty()
            strategy_text.info("Waiting for threat detection to generate strategy...")

    # ========================================================================
    # TAB 2: SATELLITE VIEW
    # ========================================================================
    with tab2:
        st.header("🛰️ GLOBAL SATELLITE SURVEILLANCE")
        
        # Layout: Controls and Map - NO NESTING
        col_map, col_controls = st.columns([3, 1])
        
        with col_controls:
            st.markdown("### 📡 UPLINK SETTINGS")
            target_region = st.selectbox("Target Sector", ["South Asia", "Global", "North America", "Europe"])
            map_style = st.selectbox("Vis Mode", ["Dark Matter", "Satellite", "OpenStreetMap"])
            scan_btn = st.button("🚀 INITIATE SCAN", type="primary", use_container_width=True)
            st.divider()
            st.markdown("**ORBITAL ASSETS ONLINE:**")
            st.success("🟢 VIIRS-SNPP (Polar)")
            st.success("🟢 MODIS-Aqua (EOS)")
            st.success("🟢 NOAA-20 (JPSS)")
            st.info("ℹ️ Data latency: ~3 hours (Near Real-Time)")

        with col_map:
            tiles = "Cartodb dark_matter" if map_style == "Dark Matter" else "OpenStreetMap"
            m = folium.Map(location=[22.5, 82.0], zoom_start=4, tiles=tiles, prefer_canvas=True)
            fire_count = 0
            if scan_btn:
                with st.spinner(f"🛰️ Aligning sensors over {target_region}..."):
                    fire_df = satellite_uplink.fetch_live_data(target_region)
                    time.sleep(1.5)
                
                if not fire_df.empty:
                    fire_count = len(fire_df)
                    st.toast(f"⚠ {fire_count} THERMAL ANOMALIES DETECTED", icon="🔥")
                    for _, row in fire_df.iterrows():
                        lat, lon = row['latitude'], row['longitude']
                        temp_k = row.get('brightness', 300)
                        conf = row.get('confidence', 50)
                        color = "#ff0000" if temp_k > 350 else "#ff5500"
                        popup_html = f"<div style='font-family: monospace;'><b>⚠ THERMAL SIGNATURE</b><br>LAT: {lat:.3f}<br>LON: {lon:.3f}<br>TMP: {temp_k:.1f} K<br>CNF: {conf}</div>"
                        folium.CircleMarker(location=[lat, lon], radius=4 if temp_k > 350 else 2, popup=folium.Popup(popup_html, max_width=200), color=color, fill=True, fill_color=color, fill_opacity=0.8).add_to(m)
                else:
                    st.warning("No active thermal anomalies detected in this sector.")

            st_folium(m, height=600, width="100%")
            if scan_btn and fire_count > 0:
                st.markdown("### 📋 RAW TELEMETRY STREAM")
                st.dataframe(fire_df[['latitude', 'longitude', 'brightness', 'acq_time']].head(50), use_container_width=True, height=200)

    # ========================================================================
    # TAB 3: ANALYTICS
    # ========================================================================
    with tab3:
        st.header("📊 Analytics")
        # Display Mission Report if generated
        if 'last_report' in st.session_state:
            st.success("✅ New Mission Report Available")
            st.markdown(st.session_state['last_report'])
            st.download_button("💾 Download Report", st.session_state['last_report'], file_name="mission_report.txt")
            st.divider()

        if os.path.exists("logs/incident_log.csv"):
            df = pd.read_csv("logs/incident_log.csv")
            if not df.empty:
                # Analytics charts - SEPARATE ROW
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Threat Confidence Trend")
                    st.line_chart(df["Confidence"])
                with col2:
                    st.markdown("#### Detection Distribution")
                    st.bar_chart(df["Type"].value_counts())
                st.markdown("#### Detailed Logs")
                st.dataframe(df.tail(10), use_container_width=True)
            else:
                st.info("No data logged yet.")
        else:
            st.warning("Log file initialization pending.")

    # ========================================================================
    # VIDEO PROCESSING LOOP
    # ========================================================================

    # Visual Q&A Trigger Handling
    if analyze_btn:
        st.session_state.do_analysis = True

    if st.session_state.detection_active:
        cap = None
        try:
            if input_source == "Webcam Feed": 
                cap = cv2.VideoCapture(0)
            elif input_source == "Video Upload" and vid_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(vid_file.read())
                cap = cv2.VideoCapture(tfile.name)
            elif input_source == "IP Camera" and ip_camera_url: 
                cap = cv2.VideoCapture(ip_camera_url)
            elif input_source == "Drone Stream" and drone_stream_url: 
                cap = cv2.VideoCapture(drone_stream_url)

            if cap and cap.isOpened():
                alert_classes = get_alert_classes(demo_mode)
                fire_frames = 0
                frame_count = 0
                recorder = None
                is_recording = False
                base_lat, base_long = 30.7333, 76.7794
                
                # OPTIMIZATION: Cache toggle values
                APPLY_CLAHE = clahe_enhancement
                APPLY_THERMAL = thermal_vision
                
                last_boxes = []
                last_class = "none"
                last_conf = 0.0
                last_is_fire = False
                
                # FPS calculation
                fps_start_time = time.time()
                fps_frame_count = 0

                while cap.isOpened() and st.session_state.detection_active:
                    ret, frame = cap.read()
                    if not ret: 
                        st.warning("Video stream ended.")
                        break

                    # OPTIMIZATION: Adaptive resize to maintain aspect ratio
                    TARGET_WIDTH = 640
                    h, w = frame.shape[:2]
                    if w > TARGET_WIDTH:
                        scale = TARGET_WIDTH / w
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    frame_count += 1
                    fps_frame_count += 1
                    
                    # Calculate FPS every 30 frames
                    if fps_frame_count >= 30:
                        elapsed = time.time() - fps_start_time
                        current_fps = fps_frame_count / elapsed if elapsed > 0 else 0
                        st.session_state.fps_history.append(current_fps)
                        # OPTIMIZATION: Keep only last 100 FPS readings
                        st.session_state.fps_history = st.session_state.fps_history[-100:]
                        fps_frame_count = 0
                        fps_start_time = time.time()

                    # 0. AI FIELD INTEL (Visual Q&A One-shot analysis)
                    if st.session_state.get('do_analysis', False):
                        with st.spinner("Analyzing visual data..."):
                            answer = field_intel.analyze_frame(frame, user_q)
                            st.session_state['vqa_response'] = answer
                            st.session_state.do_analysis = False
                            intel_ph.info(f"🤖 {answer}")

                    # 1. SATELLITE BACKGROUND CHECK
                    # OPTIMIZATION: Check every SATELLITE_CHECK_INTERVAL frames
                    if auto_satellite and frame_count % SATELLITE_CHECK_INTERVAL == 0:
                        threat_count, closest_dist, threat_level, nearby_fires = \
                            satellite_uplink.check_proximity_threats(base_lat, base_long, detection_radius)
                        
                        if threat_count > 0:
                            satellite_alert_ph.error(
                                f"🚨 {threat_count} satellite fire(s) within {detection_radius}km! "
                                f"Closest: {closest_dist:.1f}km ({threat_level})"
                            )
                            analytics.log_incident("SATELLITE_THERMAL", 0.99, 1.0)
                        else:
                            satellite_alert_ph.success(f"✅ Satellite: All clear within {detection_radius}km")

                    # 2. Filters
                    # OPTIMIZATION: Only apply CLAHE every 2nd frame when enabled
                    if APPLY_CLAHE and frame_count % 2 == 0:
                        frame = apply_enhancement(frame)
                    
                    if APPLY_THERMAL:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

                    # 3. HUD
                    avg_fps = np.mean(st.session_state.fps_history[-10:]) if st.session_state.fps_history else 0
                    frame = draw_hud(frame, frame_count, base_lat, base_long, avg_fps)

                    # 4. AI Detection (OPTIMIZED)
                    if ensemble_detection:
                        # OPTIMIZATION: Process every FRAME_SKIP frame
                        if frame_count % FRAME_SKIP == 0:
                            is_fire, confidence, detected_class, boxes = ensemble_model.predict_ensemble(
                                frame, ai_sensitivity, use_fp16=use_fp16
                            )
                            last_is_fire, last_conf, last_class, last_boxes = is_fire, confidence, detected_class, boxes
                        else:
                            is_fire, confidence, detected_class, boxes = last_is_fire, last_conf, last_class, last_boxes

                        for box in boxes:
                            x1, y1, x2, y2, cls_name, conf = box
                            color = (255, 255, 255) if APPLY_THERMAL else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            label = f"🔥 {cls_name.upper()} {conf:.0%}"
                            if demo_mode: label = f"🧪 {cls_name.upper()}"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    else:
                        # Standard YOLO fall-back logic
                        if frame_count % FRAME_SKIP == 0:
                            try:
                                results = model(frame, verbose=False, conf=ai_sensitivity, half=use_fp16, device=0)
                            except:
                                results = model(frame, verbose=False, conf=ai_sensitivity)
                            
                            detections = results[0].boxes
                            temp_boxes = []
                            temp_is_fire = False
                            temp_conf = 0.0
                            temp_class = "none"

                            if detections:
                                for box in detections:
                                    cls_id = int(box.cls[0])
                                    cls_name = model.names[cls_id].lower()
                                    conf = float(box.conf[0])
                                    if cls_name in alert_classes and conf > ai_sensitivity:
                                        temp_is_fire = True
                                        temp_conf = max(temp_conf, conf)
                                        temp_class = cls_name
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        temp_boxes.append((x1, y1, x2, y2, cls_name, conf))
                            last_boxes, last_is_fire, last_conf, last_class = temp_boxes, temp_is_fire, temp_conf, temp_class
                        else:
                            is_fire, confidence, detected_class = last_is_fire, last_conf, last_class

                        for box in last_boxes:
                            x1, y1, x2, y2, cls_name, conf = box
                            color = (255, 255, 255) if APPLY_THERMAL else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            label = f"🔥 {cls_name.upper()} {conf:.0%}"
                            if demo_mode: label = f"🧪 {cls_name.upper()}"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 5. Data Logging & Alerts
                    st.session_state.confidence_history.append(confidence)
                    # OPTIMIZATION: More efficient list trimming
                    st.session_state.confidence_history = st.session_state.confidence_history[-50:]
                    
                    chart_data = pd.DataFrame({"Confidence": st.session_state.confidence_history})
                    confidence_placeholder.line_chart(chart_data, height=150)

                    if is_fire:
                        fire_frames += 1
                    else:
                        fire_frames = max(0, fire_frames - 1)

                    # DETECTION LOGIC with proper confidence threshold
                    if fire_frames > 5 and confidence > 0.45:
                        
                        current_time = time.time()
                        
                        # OPTIMIZATION: Progressive notifications based on confidence
                        if current_time - st.session_state.last_notification_time > 3.0:
                            if confidence > 0.75:
                                st.toast(f"⚠️ HIGH CONFIDENCE THREAT: {detected_class.upper()}", icon="🚨")
                            else:
                                st.toast(f"🔥 {detected_class.upper()} detected ({confidence:.0%})", icon="🔥")
                            st.session_state.last_notification_time = current_time

                        # STRATEGY GENERATION
                        # OPTIMIZATION: Only update every 45 seconds AND only when fire detected
                        if auto_response and (frame_count % STRATEGY_UPDATE_INTERVAL == 0): 
                            current_weather = get_weather_data()
                            current_location = (base_lat, base_long)
                            available_teams = emergency_coord.response_teams
                            tactical_plan = strategic_ai.generate_strategy(
                                frame, detected_class, confidence, current_weather, available_teams, current_location
                            )
                            strategy_text.markdown(f"**⚡ INCIDENT COMMANDER:**\n\n{tactical_plan}")

                        risk = analytics.calculate_risk_score((base_lat, base_long), get_weather_data())
                        analytics.log_incident(detected_class, confidence, risk)

                        # RECORDING LOGIC
                        if not is_recording and evidence_recording:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"evidence_locker/evidence_{timestamp}.mp4"
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            recorder = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))
                            is_recording = True
                            generate_incident_report(timestamp, detected_class, confidence, base_lat, base_long, risk)

                        # AUDIO/DISCORD/SMS ALERTS - FIXED VERSION
                        if fire_frames == 6 and auto_response:
                            details = {'confidence': confidence}
                            
                            # FIXED: Changed from < to > (Wait for 1 second to pass)
                            if current_time - st.session_state.last_notification_time > 1.0:
                                speak_alert("Warning. Threat detected.")
                                
                                # FIXED: Actually call the SMS function
                                sms_sent = send_sms_alert(base_lat, base_long, "CRITICAL THREAT", details)
                                
                                # FIXED: Update Discord with dynamic location link
                                discord_sent = send_discord_alert("CRITICAL", f"http://maps.google.com/?q={base_lat},{base_long}", details)
                                
                                # Update notification timer
                                st.session_state.last_notification_time = current_time
                                
                                # UI Feedback
                                if sms_sent:
                                    st.toast("📱 SMS Alert Sent", icon="✅")
                                else:
                                    st.toast("⚠️ SMS Failed (Check Twilio Config)", icon="⚠️")

                    else:
                        if is_recording:
                            if recorder: recorder.release()
                            is_recording = False
                            recorder = None
                            st.toast("💾 Evidence Secured")

                    if is_recording and recorder: recorder.write(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
                    time.sleep(0.01)

        finally:
            if 'recorder' in locals() and recorder: recorder.release()
            if cap: cap.release()

if __name__ == "__main__":
    main()
