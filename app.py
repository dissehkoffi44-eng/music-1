import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import requests  
import gc                                     
from scipy.signal import butter, lfilter

# --- CONFIGURATION & CONSTANTES ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# Ton rappel : F# MINOR = 11A
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS TECHNIQUES & FILTRAGE ---

def butter_lowpass_filter(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode == 'minor' else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def identify_chord_triad(chroma_vector):
    """
    LOGIQUE DE TRIADES : Vérifie la présence de l'accord complet 
    (Fondamentale, Tierce, Quinte) pour confirmer le mode.
    """
    best_score = -1
    detected_mode = "unknown"
    for i in range(12):
        # Masque Majeur : Fondamentale, +4 demi-tons, +7 demi-tons
        maj_mask = np.zeros(12); maj_mask[[i, (i+4)%12, (i+7)%12]] = 1
        # Masque Mineur : Fondamentale, +3 demi-tons, +7 demi-tons
        min_mask = np.zeros(12); min_mask[[i, (i+3)%12, (i+7)%12]] = 1
        
        s_maj = np.dot(chroma_vector, maj_mask)
        s_min = np.dot(chroma_vector, min_mask)
        
        if s_maj > best_score: best_score, detected_mode = s_maj, "major"
        if s_min > best_score: best_score, detected_mode = s_min, "minor"
    return detected_mode

def check_leading_tone(chroma_avg, key_index):
    leading_tone_idx = (key_index - 1) % 12
    return chroma_avg[leading_tone_idx] > np.mean(chroma_avg) * 1.2

def detect_perfect_cadence(n1, n2):
    try:
        r1, r2 = n1.split()[0], n2.split()[0]
        i1, i2 = NOTES_LIST.index(r1), NOTES_LIST.index(r2)
        if (i1 + 7) % 12 == i2: return True, n2
        if (i2 + 7) % 12 == i1: return True, n1
        return False, n1
    except: return False, n1

# --- MOTEUR D'ANALYSE ---

def analyze_segment(y, sr, tuning=0.0):
    if len(y) < 512: return None, 0.0, None
    # CHROMA CQT : Résolution constante adaptée au clavier musical
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Harmonique Profonde V4...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    # 1. Chargement, Nettoyage & Tuning
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    y, _ = librosa.effects.trim(y) 
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, _ = librosa.effects.hpss(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. FILTRAGE DES BASSES PRÉCIS (Confirmation Tonique)
    y_low = butter_lowpass_filter(y, cutoff=150, sr=sr)
    chroma_low = librosa.feature.chroma_cqt(y=y_low, sr=sr, tuning=tuning_offset)
    bass_tonique_idx = np.argmax(np.mean(chroma_low, axis=1))
    bass_note = NOTES_LIST[bass_tonique_idx]

    # 3. Analyse par segments (Timeline & Triades)
    timeline_data, votes, chord_modes = [], [], []
    step = 8
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            chord_modes.append(identify_chord_triad(chroma_vec))
            timeline_data.append({
                "Temps": start_t, 
                "Note": key_seg, 
                "Confiance": round(float(score_seg) * 100, 1)
            })

    # 4. Analyse de la zone stable (70-80% - dernier refrain)
    start_stable = int(duration * 0.7 * sr)
    end_stable = int(duration * 0.8 * sr)
    y_stable = y_harm[start_stable:end_stable]
    key_stable, score_stable, chroma_stable = analyze_segment(y_stable, sr, tuning=tuning_offset)

    # 5. Arbitrage Musical & BONUS MUSICAL
    counts = Counter(votes)
    votes_sorted = counts.most_common(2)
    n1 = votes_sorted[0][0] if len(votes_sorted) > 0 else "C major"
    n2 = votes_sorted[1][0] if len(votes_sorted) > 1 else n1
    
    # Détection du mode dominant par triades
    triad_mode_dominant = Counter(chord_modes).most_common(1)[0][0]
    
    warnings = []
    final_decision = n1
    musical_bonus = 0

    # Validation par la BASSE (Indicateur clé)
    if bass_note == final_decision.split()[0]:
        musical_bonus += 20
    elif bass_note == n2.split()[0]:
        final_decision = n2
        musical_bonus += 15
        warnings.append(f"🎸 PRIORITÉ BASSE : Basculement vers {n2} (Tonique basse dominante).")

    # Validation par Triades (Majeur vs Mineur)
    if triad_mode_dominant in final_decision:
        musical_bonus += 15
    else:
        warnings.append(f"⚠️ AMBIGUÏTÉ : Profil suggère {final_decision.split()[1]} mais texture triade {triad_mode_dominant}.")

    # Validation par la Zone Stable
    if key_stable == final_decision:
        musical_bonus += 10

    # Vérification Mode Mineur (Sensible)
    root_idx = NOTES_LIST.index(final_decision.split()[0])
    if "minor" in final_decision:
        if check_leading_tone(chroma_stable, root_idx):
            musical_bonus += 10
        else:
            warnings.append("❓ SANS SENSIBLE : Mode mineur mélodiquement ambigu.")

    # Cadence V-I
    is_cadence, confirmed_root = detect_perfect_cadence(n1, n2)
    if is_cadence and confirmed_root == final_decision:
        musical_bonus += 15

    total_conf = min(int((counts[final_decision]/len(votes)*100) + musical_bonus), 100)

    # UI Design & Tempo
    if total_conf > 85: label, bg = "NOTE INDISCUTABLE", "linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"
    elif total_conf > 65: label, bg = "NOTE TRÈS FIABLE", "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)"
    else: label, bg = "ANALYSE COMPLEXE", "linear-gradient(135deg, #f83600 0%, #f9d423 100%)"

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "file_name": file_name,
        "recommended": {"note": final_decision, "conf": total_conf, "label": label, "bg": bg},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "warnings": warnings,
        "is_cadence": is_cadence,
        "details": {"bass": bass_note, "triad": triad_mode_dominant}
    }

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="RCDJ228 ULTIME KEY PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 RCDJ228 ULTIME KEY PRO - V4")
st.subheader("Basses CQT, Triades & Bonus Musical")

files = st.file_uploader("📂 DEPOSEZ VOS FICHIERS AUDIO", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    for f in reversed(files):
        file_bytes = f.read()
        res = get_full_analysis(file_bytes, f.name)
        
        st.markdown(f"""
            <div style="background:{res['recommended']['bg']}; padding:35px; border-radius:20px; color:white; text-align:center; margin:20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                <h2 style="margin:0; opacity:0.9;">{res['file_name']}</h2>
                <h1 style="font-size:5.5em; margin:15px 0; font-weight:900;">{res['recommended']['note']}</h1>
                <h2 style="margin:0; font-weight:700;">{get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% PRÉCISION</h2>
                <p style="margin-top:10px; text-transform:uppercase; letter-spacing:3px; font-size:0.9em;">{res['recommended']['label']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tempo Estimé", f"{res['tempo']} BPM")
        with col2:
            st.metric("Tonique Basse", res['details']['bass'])
        with col3:
            st.metric("Mode Triade", res['details']['triad'].capitalize())
            
        if res['is_cadence']:
            st.success("🎹 Cadence V-I Détectée")
            
        if res['warnings']:
            st.warning("🔍 Diagnostic Technique")
            for w in res['warnings']:
                st.write(f"- {w}")

        with st.expander("Analyse de Stabilité Graphique"):
            df_tl = pd.DataFrame(res['timeline'])
            fig = px.scatter(df_tl, x="Temps", y="Note", color="Confiance", size="Confiance", 
                             color_continuous_scale="Viridis", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

gc.collect()
