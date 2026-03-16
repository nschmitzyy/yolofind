import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import datetime
import os

# Seite konfigurieren
st.set_page_config(page_title="KI Fundbüro", page_icon="🔍")

# Modell laden (YOLOv8 Nano - leicht & schnell für Streamlit)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Datenbank laden (CSV)
DB_FILE = "data.csv"
if not os.path.exists(DB_FILE):
    df = pd.DataFrame(columns=["Datum", "Gegenstand", "Ort", "Status"])
    df.to_csv(DB_FILE, index=False)

st.title("🔍 KI-gestütztes Fundbüro")
menu = ["Gegenstand melden", "Fundbüro durchsuchen"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Gegenstand melden":
    st.header("Neuen Fund registrieren")
    
    # Kamera oder Upload
    img_file = st.camera_input("Foto vom Fundstück machen") # Oder st.file_uploader
    fundort = st.text_input("Wo wurde es gefunden?", placeholder="z.B. Mensa, Parkplatz A")

    if img_file and st.button("KI-Analyse & Speichern"):
        img = Image.open(img_file)
        results = model(img)
        
        # Erkennung auslesen
        detected_names = []
        for r in results:
            for c in r.boxes.cls:
                detected_names.append(model.names[int(c)])
        
        label = detected_names[0] if detected_names else "Unbekannt"
        label_de = label.replace("cell phone", "Handy").replace("backpack", "Rucksack") # Quick-Fix Übersetzung
        
        st.success(f"Erkannt: **{label_de}**")
        st.image(results[0].plot(), caption="KI-Vorschau")

        # In CSV speichern
        new_data = pd.DataFrame([[datetime.datetime.now().strftime("%d.%m.%Y %H:%M"), label_de, fundort, "Gefunden"]], 
                                columns=["Datum", "Gegenstand", "Ort", "Status"])
        new_data.to_csv(DB_FILE, mode='a', header=False, index=False)
        st.balloons()
        st.info("Eintrag wurde gespeichert!")

elif choice == "Fundbüro durchsuchen":
    st.header("Aktuelle Fundstücke")
    df_display = pd.read_csv(DB_FILE)
    st.dataframe(df_display, use_container_width=True)
