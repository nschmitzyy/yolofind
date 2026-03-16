import os
import sys

# Dieser Block muss GANZ OBEN stehen, vor jedem anderen Import
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

import streamlit as st
# Versuche OpenCV explizit zuerst zu laden, bevor Ultralytics es tut
try:
    import cv2
except ImportError:
    st.error("OpenCV System-Bibliotheken fehlen noch. Bitte 'packages.txt' prüfen.")

from ultralytics import YOLOimport os
# Verhindert, dass OpenCV nach bestimmten Grafik-Treibern sucht
os.environ["QT_QPA_PLATFORM"] = "offscreen"import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import datetime


# 1. Grundkonfiguration
st.set_page_config(page_title="KI Fundbüro", page_icon="🕵️", layout="wide")

# Ordner für Bilder erstellen, falls er nicht existiert
IMG_FOLDER = "found_images"
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# CSV Datenbank initialisieren
DB_FILE = "data.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["ID", "Datum", "Gegenstand", "Ort", "Bildpfad"])
    df.to_csv(DB_FILE, index=False)

# YOLO Modell laden
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 2. UI - Sidebar
st.sidebar.title("🕵️ Menü")
choice = st.sidebar.radio("Navigation", ["Gegenstand melden", "Galerie durchsuchen"])

# 3. Funktion: Gegenstand melden
if choice == "Gegenstand melden":
    st.header("📸 Fundstück hochladen")
    
    with st.expander("Anleitung", expanded=True):
        st.write("Lade ein Bild hoch. Unsere KI versucht automatisch zu erkennen, was es ist!")

    uploaded_file = st.file_uploader("Bild auswählen...", type=['jpg', 'jpeg', 'png'])
    fundort = st.text_input("Fundort (z.B. Café, Flur 2, Buslinie 10)")

    if uploaded_file and fundort:
        img = Image.open(uploaded_file)
        
        # KI Analyse
        results = model(img)
        
        # Erkennung verarbeiten
        detected_items = [model.names[int(c)] for r in results for c in r.boxes.cls]
        primary_item = detected_items[0] if detected_items else "Unbekannt"
        
        st.info(f"KI-Vorschlag: **{primary_item}**")
        
        if st.button("Fund offiziell registrieren"):
            # Bild speichern
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"{timestamp}_{primary_item}.jpg"
            img_path = os.path.join(IMG_FOLDER, img_filename)
            img.save(img_path)
            
            # In CSV speichern
            new_entry = {
                "ID": timestamp,
                "Datum": datetime.datetime.now().strftime("%d.%m.%Y"),
                "Gegenstand": primary_item,
                "Ort": fundort,
                "Bildpfad": img_path
            }
            pd.DataFrame([new_entry]).to_csv(DB_FILE, mode='a', header=False, index=False)
            
            st.success("Erfolgreich gespeichert!")
            st.balloons()

# 4. Funktion: Galerie durchsuchen
else:
    st.header("📦 Aktuelle Fundstücke")
    
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        
        if df.empty:
            st.warning("Noch keine Fundstücke registriert.")
        else:
            # Suche / Filter
            search = st.text_input("Nach Gegenstand oder Ort suchen...")
            if search:
                df = df[df['Gegenstand'].str.contains(search, case=False) | df['Ort'].str.contains(search, case=False)]

            # Darstellung in Spalten (Grid)
            cols = st.columns(3)
            for index, row in df.iterrows():
                with cols[index % 3]:
                    if os.path.exists(row['Bildpfad']):
                        st.image(row['Bildpfad'], use_container_width=True)
                    st.write(f"**{row['Gegenstand']}**")
                    st.caption(f"📍 {row['Ort']} | 📅 {row['Datum']}")
                    st.divider()
