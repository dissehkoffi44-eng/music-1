import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests  
import gc                                     
from scipy.signal import butter, lfilter

# --- CONFIGURATION & CONSTANTES ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils de Krumhansl-Kessler (Standard de l'industrie pour la détection de tonalité)
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS LOGIQUES MUSICALES ---

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode == 'minor' else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def check_leading_tone(chroma_avg, key_index):
    """RÈGLE DE LA SENSIBLE : Vérifie si la 7ème note augmentée est présente (caractéristique du mineur)."""
    leading_tone_idx = (key_index - 1) % 12
    return chroma_avg[leading_tone_idx] > np.mean(chroma_avg) * 1.2

def detect_perfect_cadence(n1, n2):
    """RÈGLE V-I : Détecte si un mouvement de quinte (Dominante -> Tonique) existe entre les deux favoris."""
    try:
        r1, r2 = n1.split()[0], n2.split()[0]
        i1, i2 = NOTES_LIST.index(r1), NOTES_LIST.index(r2)
        if (i1 + 7) % 12 == i2: return True, n2 # n1 est la dominante de n2
        if (i2 + 7) % 12 == i1: return True, n1 # n2 est la dominante de n1
        return False, n1
    except: return False, n1

# --- CORE ENGINE ---

def analyze_segment(y, sr, tuning=0.0):
    if len(y) < 512: return None, 0.0
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Harmonique Profonde...", max_entries=20)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.hpss(y)[0]
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. ANALYSE TEMPORELLE (VOTES)
    timeline_data, votes = [], []
    step = 8
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg, _ = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})

    # 2. RÈGLE DE LA FIN (RÉSOLUTION)
    y_final = y_harm[-int(5*sr):] # Analyse des 5 dernières secondes
    key_final, score_final, chroma_final = analyze_segment(y_final, sr, tuning=tuning_offset)

    # 3. ARBITRAGE ET LOGIQUE DE DÉCISION
    counts = Counter(votes)
    n1, n2 = counts.most_common(2)[0][0], (counts.most_common(2)[1][0] if len(counts)>1 else counts.most_common(1)[0][0])
    
    final_decision = n1
    musical_bonus = 0

    # Priorité à la résolution finale si elle est stable
    if score_final > 0.75:
        final_decision = key_final
        musical_bonus += 25

    # Test de la Sensible pour confirmer le mode mineur
    root_idx = NOTES_LIST.index(final_decision.split()[0])
    if "minor" in final_decision:
        if check_leading_tone(chroma_final, root_idx):
            musical_bonus += 15 # Confirmation Sensible
    
    # Détection Cadence
    is_cadence, confirmed_root = detect_perfect_cadence(n1, n2)
    if is_cadence:
        final_decision = confirmed_root
        musical_bonus += 20

    # 4. CALCUL DU SCORE FINAL
    base_conf = (counts[final_decision] / len(votes)) * 100 if votes else 0
    total_conf = min(int(base_conf + musical_bonus), 100)

    # UI Labeling
    if total_conf > 85: label, bg = "NOTE INDISCUTABLE", "linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"
    elif total_conf > 65: label, bg = "NOTE TRÈS FIABLE", "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)"
    else: label, bg = "ANALYSE COMPLEXE", "linear-gradient(135deg, #f83600 0%, #f9d423 100%)"

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "file_name": file_name,
        "recommended": {"note": final_decision, "conf": total_conf, "label": label, "bg": bg},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "is_cadence": is_cadence,
        "details": {"n1": n1, "n2": n2, "fin": key_final}
    }

# --- INTERFACE STREAMLIT (VERSION SIMPLIFIÉE POUR RÉSUMÉ) ---
st.set_page_config(page_title="RCDJ228 ULTIME KEY", layout="wide")
st.title("🎧 RCDJ228 ULTIME KEY PRO")

uploaded_files = st.file_uploader("Importer des pistes", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        res = get_full_analysis(f.read(), f.name)
        
        st.markdown(f"""
            <div style="background:{res['recommended']['bg']}; padding:20px; border-radius:15px; color:white; text-align:center;">
                <h2>{res['file_name']}</h2>
                <h1 style="font-size:4em;">{res['recommended']['note']} ({get_camelot_pro(res['recommended']['note'])})</h1>
                <p>{res['recommended']['label']} | Confiance : {res['recommended']['conf']}% | BPM : {res['tempo']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Graphique de stabilité
        df_tl = pd.DataFrame(res['timeline'])
        fig = px.line(df_tl, x="Temps", y="Note", title="Évolution de la tonalité (Chercher la ligne la plus plate)")
        st.plotly_chart(fig, use_container_width=True)
