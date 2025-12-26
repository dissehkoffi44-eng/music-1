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

# --- FONCTIONS DE TRAITEMENT DE SIGNAL (FULL) ---

def spectral_whitening(y, sr):
    """
    Supprime les pics de résonance artificiels causés par la compression MP3 (Lo-Fi).
    Indispensable pour la précision sur les fichiers 64kbps/128kbps.
    """
    S = np.abs(librosa.stft(y))
    # Calcul de la moyenne spectrale pour aplatir (whitening)
    S_white = S / (np.mean(S, axis=0) + 1e-6)
    return librosa.istft(S_white)

def butter_lowpass_filter(data, cutoff, sr, order=5):
    """Filtre passe-bas de Butterworth pour isoler la fréquence fondamentale de la basse."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def get_camelot_pro(key_mode_str):
    """Convertit la clé détectée en code Camelot (ex: 11A)."""
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode == 'minor':
            return BASE_CAMELOT_MINOR.get(key, "??")
        else:
            return BASE_CAMELOT_MAJOR.get(key, "??")
    except:
        return "??"

def identify_complex_chords(chroma_vector):
    """
    Analyse les structures d'accords : Triades simples, 7ème de dominante et 9ème.
    Permet de mieux gérer le Jazz, la Soul et la Deep House.
    """
    best_score = -1
    detected_mode = "unknown"
    
    for i in range(12):
        # Masques binaires pour chaque type d'accord
        maj_mask = np.zeros(12); maj_mask[[i, (i+4)%12, (i+7)%12]] = 1
        min_mask = np.zeros(12); min_mask[[i, (i+3)%12, (i+7)%12]] = 1
        sev_mask = np.zeros(12); sev_mask[[i, (i+4)%12, (i+7)%12, (i+10)%12]] = 0.9 # Jazz/7th
        
        s_maj = np.dot(chroma_vector, maj_mask)
        s_min = np.dot(chroma_vector, min_mask)
        s_sev = np.dot(chroma_vector, sev_mask)
        
        current_max = max(s_maj, s_min, s_sev)
        if current_max > best_score:
            best_score = current_max
            if current_max == s_maj: detected_mode = "major"
            elif current_max == s_min: detected_mode = "minor"
            else: detected_mode = "jazz_7th"
            
    return detected_mode

# --- MOTEUR D'ANALYSE PRINCIPAL ---

def analyze_segment(y, sr, tuning=0.0):
    """Analyse un bloc de 6-8 secondes via Constant-Q Transform."""
    if len(y) < 512: return None, 0.0, None
    # bins_per_octave=24 pour une résolution microtonale (mieux que le standard 12)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=24)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Deep Analysis V5.1 en cours...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    # 1. Chargement et Pre-processing
    y_raw, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    
    # Étape critique : Blanchiment spectral pour contrer la compression
    y_clean = spectral_whitening(y_raw, sr)
    y_clean, _ = librosa.effects.trim(y_clean)
    
    # Estimation précise du désaccordage (Tuning)
    tuning_offset = librosa.estimate_tuning(y=y_clean, sr=sr)
    
    # Séparation Harmonique (pour les notes) / Percussive (pour le tempo)
    y_harm, y_perc = librosa.effects.hpss(y_clean, margin=(2.0, 5.0))
    duration = librosa.get_duration(y=y_clean, sr=sr)
    
    # 2. Analyse de la Fondation (Basse)
    y_low = butter_lowpass_filter(y_harm, cutoff=140, sr=sr)
    chroma_low = librosa.feature.chroma_cqt(y=y_low, sr=sr, tuning=tuning_offset)
    bass_note = NOTES_LIST[np.argmax(np.mean(chroma_low, axis=1))]

    # 3. Analyse Temporelle et Harmonique
    timeline_data, votes, complex_modes = [], [], []
    step = 6 # Précision chirurgicale toutes les 6 secondes
    
    for start_t in range(0, int(duration) - step, step):
        start_sample = int(start_t * sr)
        end_sample = int((start_t + step) * sr)
        y_seg = y_harm[start_sample:end_sample]
        
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning_offset)
        
        if key_seg:
            votes.append(key_seg)
            complex_modes.append(identify_complex_chords(chroma_vec))
            timeline_data.append({
                "Temps": start_t, 
                "Note": key_seg, 
                "Confiance": round(float(score_seg) * 100, 1)
            })

    # 4. Décision Finale et Bonus Musical (Logique Pro)
    counts = Counter(votes)
    n1 = counts.most_common(1)[0][0] if votes else "C major"
    jazz_presence = Counter(complex_modes).get("jazz_7th", 0) / len(complex_modes) if complex_modes else 0
    
    warnings = []
    musical_bonus = 0
    
    # Règle 1 : Cohérence avec la basse
    if bass_note == n1.split()[0]:
        musical_bonus += 20
    
    # Règle 2 : Complexité Harmonique
    if jazz_presence > 0.25:
        musical_bonus += 10
        warnings.append("🎷 STRUCTURE JAZZ/SOUL : Accords de 7ème détectés.")

    # Règle 3 : Modulations
    if len(counts.keys()) > 5:
        warnings.append("🔄 MODULATION : Changements de tonalité multiples détectés.")

    total_conf = min(int((counts[n1]/len(votes)*100) + musical_bonus), 100)

    # UI Branding Logic
    if total_conf > 85: 
        label, bg = "NOTE INDISCUTABLE", "linear-gradient(135deg, #1D976C 0%, #93F9B9 100%)"
    elif total_conf > 65: 
        label, bg = "NOTE TRÈS FIABLE", "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)"
    else: 
        label, bg = "ANALYSE COMPLEXE", "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)"

    # Calcul du Tempo sur le signal percussif
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    return {
        "file_name": file_name,
        "recommended": {"note": n1, "conf": total_conf, "label": label, "bg": bg},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "warnings": warnings,
        "details": {"bass": bass_note, "jazz": f"{int(jazz_presence*100)}%"}
    }

# --- INTERFACE STREAMLIT (FULL DESIGN) ---
st.set_page_config(page_title="RCDJ228 ULTIME KEY V5.1", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 RCDJ228 ULTIME KEY PRO - V5.1")
st.subheader("Moteur de calcul non-simplifié : CQT, Whitening & Chord Extensions")

files = st.file_uploader("📂 DEPOSEZ VOS TRACKS ICI", accept_multiple_files=True, type=['mp3', 'wav', 'flac', 'm4a'])

if files:
    for f in reversed(files):
        file_bytes = f.read()
        res = get_full_analysis(file_bytes, f.name)
        
        # Banner Résultat PRO
        st.markdown(f"""
            <div style="background:{res['recommended']['bg']}; padding:45px; border-radius:25px; color:white; text-align:center; margin:20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                <p style="margin:0; opacity:0.8; text-transform:uppercase; letter-spacing:2px;">{res['file_name']}</p>
                <h1 style="font-size:6em; margin:15px 0; font-weight:900; line-height:1;">{res['recommended']['note'].upper()}</h1>
                <h2 style="margin:0; font-weight:700;">CAMELOT: {get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% FIABILITÉ</h2>
                <p style="margin-top:15px; text-transform:uppercase; letter-spacing:4px; font-size:0.9em; font-weight:bold;">{res['recommended']['label']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tempo track", f"{res['tempo']} BPM")
        with col2:
            st.metric("Root Bass", res['details']['bass'])
        with col3:
            st.metric("Jazz/Soul Factor", res['details']['jazz'])

        if res['warnings']:
            for w in res['warnings']: st.warning(f"🔍 {w}")

        with st.expander("📊 Voir la Stabilité Harmonique (Timeline)"):
            df_tl = pd.DataFrame(res['timeline'])
            fig = px.scatter(df_tl, x="Temps", y="Note", color="Confiance", size="Confiance", 
                             template="plotly_dark", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

# Libération forcée de la mémoire
gc.collect()
