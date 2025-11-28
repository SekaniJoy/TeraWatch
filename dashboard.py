import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import openai
import os
from dotenv import load_dotenv
from typing import Dict

# --- LLM SETUP ---
# Use Streamlit's native secrets management for deployment
try:
    # Check if the key exists in the Streamlit Secrets (for cloud deployment)
    # This is the recommended way for Streamlit Cloud
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except (AttributeError, KeyError):
    # Fallback for local testing or if secrets are missing
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


# Configuration
LOG_DIR = Path("logs")
DETECTION_LOG = LOG_DIR / "detections.csv"
ALERT_LOG = LOG_DIR / "alerts.csv"
REFRESH_RATE = 2 # seconds

# Define class lists, matching those in main.py, needed for risk calculation
PERSON_CLASSES = {"person", "man", "woman", "human"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane"}
WILDLIFE_CLASSES = {
    "elephant", "zebra", "giraffe", "lion", "leopard", "cheetah", 
    "rhinoceros", "rhino", "buffalo", "hippopotamus", "hippo", "hyena", 
    "ostrich", "camel", "crocodile", "gazelle", "impala", "antelope", 
    "baboon", "wildebeest", "warthog", 
}

# --- Utility Functions ---

@st.cache_data(ttl=REFRESH_RATE)
def load_data():
    """Reads the latest data from the CSV logs."""
    data = {}
    
    # 1. Load Detections
    if DETECTION_LOG.exists():
        try:
            df_det = pd.read_csv(DETECTION_LOG)
            df_det['timestamp'] = pd.to_datetime(df_det['timestamp'])
            data['detections'] = df_det
        except pd.errors.EmptyDataError:
            data['detections'] = pd.DataFrame(columns=['timestamp', 'camera', 'cls_name', 'confidence', 'zones', 'source'])
        except Exception:
            # Handle potential partial writes or other file errors gracefully
            data['detections'] = pd.DataFrame(columns=['timestamp', 'camera', 'cls_name', 'confidence', 'zones', 'source'])
    else:
        data['detections'] = pd.DataFrame(columns=['timestamp', 'camera', 'cls_name', 'confidence', 'zones', 'source'])

    # 2. Load Alerts
    if ALERT_LOG.exists():
        try:
            df_alert = pd.read_csv(ALERT_LOG)
            df_alert['timestamp'] = pd.to_datetime(df_alert['timestamp'])
            data['alerts'] = df_alert
        except pd.errors.EmptyDataError:
            data['alerts'] = pd.DataFrame(columns=['timestamp', 'severity', 'message', 'data'])
        except Exception:
             data['alerts'] = pd.DataFrame(columns=['timestamp', 'severity', 'message', 'data'])
    else:
        data['alerts'] = pd.DataFrame(columns=['timestamp', 'severity', 'message', 'data'])
        
    return data

def calculate_risk_score(df_det: pd.DataFrame) -> tuple[int, str]:
    """
    Replicates the risk score calculation from main.py based on the latest logged detections.
    """
    if df_det.empty:
        return 0, "LOW"

    # Identify the detections from the single latest recorded timestamp
    latest_timestamp = df_det['timestamp'].max()
    current_dets = df_det[df_det['timestamp'] == latest_timestamp]
    
    num_people = current_dets['cls_name'].isin(PERSON_CLASSES).sum()
    num_vehicles = current_dets['cls_name'].isin(VEHICLE_CLASSES).sum()
    num_wildlife = current_dets['cls_name'].isin(WILDLIFE_CLASSES).sum()

    # Base risk calculation: People=4, Vehicle=3, Wildlife=2
    base = num_people * 4 + num_vehicles * 3 + num_wildlife * 2

    # Interaction and Intrusion modifiers (only based on presence, not counting duplicates)
    has_person = num_people > 0
    has_wildlife = num_wildlife > 0
    has_vehicle = num_vehicles > 0
    
    # Check if ANY detection in the latest frame falls into a 'border' zone
    has_border = current_dets['zones'].astype(str).str.contains('border', case=False).any()

    if has_person and has_wildlife:
        base += 10 # Human-Wildlife Conflict
    if has_person and has_vehicle:
        base += 5  # Unauthorized Human/Vehicle Proximity
    if has_border:
        base += 5  # Perimeter Breach/Intrusion

    risk_score = int(max(0, min(100, base)))

    if risk_score < 30:
        risk_label = "LOW"
    elif risk_score < 70:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    return risk_score, risk_label

def get_risk_color(label):
    if label == "HIGH":
        return "#FF4B4B" # Red
    elif label == "MEDIUM":
        return "#FFD700" # Gold/Yellow
    else:
        return "#00CC96" # Green

def generate_patrol_brief_embedded(df_det: pd.DataFrame, df_alerts: pd.DataFrame, hours_back: int = 24) -> str:
    """Uses LLM to generate a patrol brief based on supplied dataframes."""

    # 1. Filter and Serialize Data
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    # Safely handle data filtering and conversion
    
    # Detections
    if 'timestamp' in df_det.columns and not df_det.empty:
        df_det_copy = df_det.copy()
        df_det_copy['timestamp'] = pd.to_datetime(df_det_copy['timestamp'], errors='coerce', utc=True)
        recent_det = df_det_copy[df_det_copy['timestamp'] >= cutoff_time].dropna(subset=['timestamp']).to_string()
    else:
        recent_det = "No detection data recorded."
        
    # Alerts
    if 'timestamp' in df_alerts.columns and not df_alerts.empty:
        df_alerts_copy = df_alerts.copy()
        df_alerts_copy['timestamp'] = pd.to_datetime(df_alerts_copy['timestamp'], errors='coerce', utc=True)
        recent_alerts = df_alerts_copy[df_alerts_copy['timestamp'] >= cutoff_time].dropna(subset=['timestamp']).to_string()
    else:
        recent_alerts = "No alerts recorded."

    # Data check
    if recent_det.strip().startswith("No detection data recorded."):
        return f"### ⚠️ No Data Available\nReport generation skipped: No detection data recorded in the last {hours_back} hours."

    # 2. Create the LLM Prompt (Removed ellipses)
    prompt = f"""
    Analyze the following surveillance log data from the TeraWatch AI system over the last {hours_back} hours.
    --- DETECTION LOG DATA (Last {hours_back} hours) ---
    {recent_det}
    --- ALERT LOG DATA (Last {hours_back} hours) ---
    {recent_alerts}

    Generate a professional 'Wildlife Reserve Patrol Brief' in Markdown format, including these sections: 
    1. **Summary of High-Risk Intrusions**: Detail any human or vehicle activity in 'border' or 'reserve' zones.
    2. **Key Wildlife Observations**: Note the most frequently detected wildlife and their locations.
    3. **Actionable Recommendations**: Suggest a high-priority patrol route or time window based on alert frequency and human/vehicle presence.
    4. **Overall Risk Profile**: State the activity level (Low/Medium/High) for the period.
    """
    
    # 3. Call the LLM API (Removed ellipses)
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[
                {"role": "system", "content": "You are an expert wildlife and border security analyst. Format your response strictly in Markdown with headings."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except openai.AuthenticationError:
        return "### 🔴 API ERROR: Authentication Failed\nYour OPENAI_API_KEY is invalid, expired, or the account is locked. Check your key and billing status."
    except openai.RateLimitError:
        return "### 🔴 API ERROR: Rate Limit Exceeded\nYou may need to upgrade your tier or wait for your usage window to reset."
    except openai.APIError as e:
        return f"### 🔴 API ERROR\nOpenAI API failed. Check model access (gpt-4-turbo) or key. Error: {e}"
    except Exception as e:
        return f"### 💔 UNEXPECTED ERROR\nAn unknown error occurred during API communication: {e}"
    
# --- Streamlit App ---

st.set_page_config(
    page_title="TeraWatch AI Surveillance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦁 TeraWatch AI Reserve Surveillance Dashboard")
st.markdown("---")

# Initialize session state for the report only once
if 'report_text' not in st.session_state:
    st.session_state['report_text'] = "Press the button below to generate the latest 24-hour patrol report."
if 'report_hours' not in st.session_state:
    st.session_state['report_hours'] = 24

# Use a container to auto-refresh the data
placeholder = st.empty()

# Main loop to continuously refresh the dashboard
while True:
    data = load_data()
    df_det = data['detections']
    df_alert = data['alerts']
    
    risk_score, risk_label = calculate_risk_score(df_det)
    risk_color = get_risk_color(risk_label)
    
    with placeholder.container():
        # 1. KEY PERFORMANCE INDICATORS (KPIs)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background-color:{risk_color}; padding: 10px; border-radius: 5px; text-align: center;'>
                <h3 style='color: black; margin: 0;'>RISK LEVEL</h3>
                <h1 style='color: black; margin: 0;'>{risk_label} ({risk_score}%)</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric(
                label="Total Objects Logged", 
                value=len(df_det)
            )

        with col3:
            # Check alerts within the last hour
            recent_alerts = df_alert[df_alert['timestamp'] >= datetime.now() - pd.Timedelta(hours=1)]
            alert_delta = None
            if not recent_alerts.empty:
                if 'high' in recent_alerts['severity'].str.lower().values:
                    alert_delta = "HIGH"
                elif 'medium' in recent_alerts['severity'].str.lower().values: # <-- COLON ADDED HERE
                    alert_delta = "MEDIUM"
                else:
                    alert_delta = "LOW"

            st.metric(
                label="Alerts (Last 1 Hr)", 
                value=len(recent_alerts),
                delta=alert_delta # Show the highest severity
            )

        with col4:
            st.metric(
                label="Data Source Mode", 
                value="Hybrid (YOLO/RF)",
                delta=f"Last refresh: {datetime.now().strftime('%H:%M:%S')}"
            )
        
        st.markdown("---")

        # 2. ALERTS AND LOGS (Two Columns)
        col_alert, col_log = st.columns([1, 2])

        with col_alert:
            st.subheader("🚨 Real-time Alerts")
            if not df_alert.empty:
                # Show only the most recent alerts
                recent_alerts_display = df_alert.tail(10).sort_values(by='timestamp', ascending=False)
                st.dataframe(
                    recent_alerts_display[['timestamp', 'severity', 'message']],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No alerts recorded yet.")
                
        with col_log:
            st.subheader("📊 Detections Over Time")
            
            if not df_det.empty:
                # Group data by 1-minute intervals and count
                df_det_agg = (
                    df_det.set_index('timestamp')
                    .resample('1min')['cls_name']
                    .count()
                    .reset_index(name='count')
                )
                
                # Create a line chart
                fig = px.line(
                    df_det_agg, 
                    x='timestamp', 
                    y='count', 
                    title='Detection Volume per Minute'
                )
                fig.update_layout(xaxis_title="Time", yaxis_title="Number of Objects")
                
                # --- FIX APPLIED: Dynamic key
                st.plotly_chart(fig, use_container_width=True, key=f"line_{time.time()}")
            else:
                st.info("No detection data recorded yet.")
        
        st.markdown("---")

        # 3. CLASS BREAKDOWN
        st.subheader("🔍 Class and Source Analysis")
        
        col_pie, col_bar = st.columns(2)
        
        if not df_det.empty:
            with col_pie:
                # Pie chart for top classes
                class_counts = df_det['cls_name'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                fig_pie = px.pie(
                    class_counts.head(10), 
                    values='Count', 
                    names='Class', 
                    title='Top 10 Detected Object Classes'
                )
                # --- FIX APPLIED: Dynamic key
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{time.time()}")

            with col_bar:
                # Bar chart for source comparison
                source_counts = df_det['source'].value_counts().reset_index()
                source_counts.columns = ['Source', 'Count']
                fig_bar = px.bar(
                    source_counts, 
                    x='Source', 
                    y='Count', 
                    color='Source',
                    title='Detections by Source (YOLO vs Roboflow)'
                )
                # --- FIX APPLIED: Dynamic key
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{time.time()}")
        
        # Display raw data tables for debugging/detailed inspection
        with st.expander("Show Raw Data Logs"):
            st.subheader("Recent Detections Log")
            st.dataframe(df_det.tail(20).sort_values(by='timestamp', ascending=False), use_container_width=True)
            
            st.subheader("Alerts Log")
            st.dataframe(df_alert.sort_values(by='timestamp', ascending=False), use_container_width=True)

        # 4. REPORT GENERATION SECTION
        st.subheader("📄 AI-Powered Patrol Brief")
        
        report_col, hours_col = st.columns([2, 1])

        # Selector for report hours
        selected_hours = hours_col.selectbox(
            "Report History (Hours)", 
            options=[6, 12, 24, 48, 72], 
            index=2, # Default to 24 hours
            key='report_hours_selector'
        )

        # Button to trigger the report generation
        if report_col.button(f"Generate AI Patrol Brief for Last {selected_hours} Hours", type="primary"):
            # Update the hours used for the next report
            st.session_state['report_hours'] = selected_hours
            
            with st.spinner(f"Contacting OpenAI (gpt-4-turbo) and analyzing {selected_hours} hours of logs..."):
                # Call the generator function with the data loaded at the start of the loop
                report = generate_patrol_brief_embedded(df_det, df_alert, hours_back=selected_hours)
                st.session_state['report_text'] = report
                
        # Display the report from session state
        st.markdown(st.session_state['report_text'])
        
        st.markdown("---")
        
        # Display raw data tables for debugging/detailed inspection
        with st.expander("Show Raw Data Logs (For Debugging)"):
            st.subheader("Recent Detections Log")
            st.dataframe(df_det.tail(20).sort_values(by='timestamp', ascending=False), use_container_width=True)
            
            st.subheader("Alerts Log")
            st.dataframe(df_alert.sort_values(by='timestamp', ascending=False), use_container_width=True)


    # Wait for the specified refresh rate
    time.sleep(REFRESH_RATE)