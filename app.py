import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import gc
from scipy.signal import butter, lfilter

# --- CONFIGURATION & CONSTANTES ---
# Rappel utilisateur : F# MINOR = 11A
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils Krumhansl-Kessler (standard de l'industrie pour la détection de tonalité)
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS DE TRAITEMENT ---

def spectral_whitening(y, sr):
    """Version allégée pour éviter de créer du bruit artificiel."""
    S = np.abs(librosa.stft(y))
    S_white = S / (np.mean(S, axis=0) + 1e-4)
    return librosa.istft(S_white)

def butter_lowpass_filter(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode == 'minor':
            return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except:
        return "??"

def identify_complex_chords(chroma_vector):
    best_score = -1
    detected_mode = "unknown"
    for i in range(12):
        maj_mask = np.zeros(12); maj_mask[[i, (i+4)%12, (i+7)%12]] = 1
        min_mask = np.zeros(12); min_mask[[i, (i+3)%12, (i+7)%12]] = 1
        sev_mask = np.zeros(12); sev_mask[[i, (i+4)%12, (i+7)%12, (i+10)%12]] = 0.8
        
        scores = [np.dot(chroma_vector, maj_mask), np.dot(chroma_vector, min_mask), np.dot(chroma_vector, sev_mask)]
        current_max = max(scores)
        if current_max > best_score:
            best_score = current_max
            modes = ["major", "minor", "jazz_7th"]
            detected_mode = modes[np.argmax(scores)]
    return detected_mode

def analyze_segment(y, sr, tuning=0.0):
    """Analyse avec normalisation Z-Score pour éviter le biais vers F#."""
    if len(y) < 2048: return None, 0.0, None
    
    # Utilisation de CQT 12 bins pour une meilleure corrélation avec les profils
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=12)
    chroma_avg = np.mean(chroma, axis=1)
    
    # Normalisation du signal
    chroma_avg -= np.mean(chroma_avg)
    std = np.std(chroma_avg)
    if std > 0: chroma_avg /= std

    best_score, res_key = -1.0, ""
    for mode, profile in PROFILES.items():
        # Normalisation du profil
        prof = np.array(profile)
        prof -= np.mean(prof)
        prof /= np.std(prof)
        
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(prof, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
                
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Deep Engine V5.1...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    y_raw, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    
    # Pre-processing
    y_clean = spectral_whitening(y_raw, sr)
    y_clean, _ = librosa.effects.trim(y_clean)
    
    tuning_offset = librosa.estimate_tuning(y=y_clean, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y_clean, margin=(2.0, 5.0))
    duration = librosa.get_duration(y=y_clean, sr=sr)
    
    # Analyse de la basse
    y_low = butter_lowpass_filter(y_harm, cutoff=140, sr=sr)
    chroma_low = librosa.feature.chroma_cqt(y=y_low, sr=sr, tuning=tuning_offset)
    bass_note = NOTES_LIST[np.argmax(np.mean(chroma_low, axis=1))]

    timeline_data, votes, complex_modes = [], [], []
    step = 6 
    
    for start_t in range(0, int(duration) - step, step):
        start_sample = int(start_t * sr)
        end_sample = int((start_t + step) * sr)
        y_seg = y_harm[start_sample:end_sample]
        
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning_offset)
        
        if key_seg and score_seg > 0.3: # Filtrer les segments trop faibles
            votes.append(key_seg)
            complex_modes.append(identify_complex_chords(chroma_vec))
            timeline_data.append({
                "Temps": start_t, 
                "Note": key_seg, 
                "Confiance": round(float(score_seg) * 100, 1)
            })

    if not votes: return None

    # Décision finale pondérée par la confiance
    counts = Counter(votes)
    n1 = counts.most_common(1)[0][0]
    
    jazz_presence = Counter(complex_modes).get("jazz_7th", 0) / len(complex_modes) if complex_modes else 0
    warnings = []
    
    # Calcul de confiance amélioré
    base_conf = (counts[n1] / len(votes)) * 100
    if bass_note == n1.split()[0]: base_conf += 15
    if jazz_presence > 0.25: warnings.append("🎷 STRUCTURE JAZZ/SOUL détectée.")
    if len(counts.keys()) > 4: warnings.append("🔄 MODULATIONS fréquentes.")

    total_conf = min(int(base_conf), 100)

    # Design UI
    if total_conf > 80: label, bg = "NOTE INDISCUTABLE", "linear-gradient(135deg, #1D976C 0%, #93F9B9 100%)"
    elif total_conf > 60: label, bg = "NOTE FIABLE", "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)"
    else: label, bg = "ANALYSE COMPLEXE", "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)"

    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    return {
        "file_name": file_name,
        "recommended": {"note": n1, "conf": total_conf, "label": label, "bg": bg},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "warnings": warnings,
        "details": {"bass": bass_note, "jazz": f"{int(jazz_presence*100)}%"}
    }

# --- INTERFACE ---
st.set_page_config(page_title="RCDJ228 ULTIME V5.1", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 RCDJ228 ULTIME KEY PRO - V5.1")
files = st.file_uploader("📂 DEPOSEZ VOS TRACKS", accept_multiple_files=True, type=['mp3', 'wav', 'flac', 'm4a'])

if files:
    for f in reversed(files):
        file_bytes = f.read()
        res = get_full_analysis(file_bytes, f.name)
        
        if res:
            st.markdown(f"""
                <div style="background:{res['recommended']['bg']}; padding:40px; border-radius:20px; color:white; text-align:center; margin:20px 0;">
                    <p style="margin:0; opacity:0.8;">{res['file_name']}</p>
                    <h1 style="font-size:5em; margin:10px 0;">{res['recommended']['note'].upper()}</h1>
                    <h2 style="margin:0;">CAMELOT: {get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}%</h2>
                    <p style="font-weight:bold; letter-spacing:2px;">{res['recommended']['label']}</p>
                </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Tempo", f"{res['tempo']} BPM")
            c2.metric("Basse (Root)", res['details']['bass'])
            c3.metric("Complexité Jazz", res['details']['jazz'])

            for w in res['warnings']: st.warning(w)

            with st.expander("📊 Timeline Harmonique"):
                df_tl = pd.DataFrame(res['timeline'])
                fig = px.scatter(df_tl, x="Temps", y="Note", color="Confiance", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Erreur d'analyse pour {f.name}")

    gc.collect()
