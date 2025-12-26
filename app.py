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

# Profils académiques de Krumhansl-Kessler
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
    """Vérifie la présence de la sensible pour le mode mineur."""
    leading_tone_idx = (key_index - 1) % 12
    return chroma_avg[leading_tone_idx] > np.mean(chroma_avg) * 1.2

def detect_perfect_cadence(n1, n2):
    """Détecte une relation Dominante -> Tonique (V-I)."""
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
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Harmonique Profonde...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    # 1. Chargement et Nettoyage (Trim des silences)
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    y, _ = librosa.effects.trim(y) 
    
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, _ = librosa.effects.hpss(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. ANALYSE PAR SEGMENTS (VOTES) SUR TOUT LE MORCEAU
    timeline_data, votes = [], []
    step = 8
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg, _ = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})

    # 3. ANALYSE DE LA ZONE STABLE (70% - 80% du morceau)
    # On évite le fade out final et on cible le dernier refrain
    start_stable = int(duration * 0.7 * sr)
    end_stable = int(duration * 0.8 * sr)
    y_stable = y_harm[start_stable:end_stable]
    key_stable, score_stable, chroma_stable = analyze_segment(y_stable, sr, tuning=tuning_offset)

    # 4. ARBITRAGE ET DIAGNOSTIC
    counts = Counter(votes)
    votes_sorted = counts.most_common(2)
    n1 = votes_sorted[0][0] if len(votes_sorted) > 0 else "C major"
    n2 = votes_sorted[1][0] if len(votes_sorted) > 1 else n1
    
    # Logique de décision robuste
    warnings = []
    final_decision = n1 # Par défaut, la majorité globale
    musical_bonus = 0

    # Validation par la zone stable
    if key_stable == n1:
        musical_bonus += 15 # La zone stable confirme la majorité
    elif key_stable == n2:
        final_decision = n2 # On bascule sur la deuxième note si la zone stable la préfère
        musical_bonus += 10
        warnings.append(f"📍 Zone stable (70%) déviante : Bascule sur {n2}")
    elif score_stable > 0.85:
        # Si la fin est ultra claire mais différente de n1/n2 (Modulation)
        final_decision = key_stable
        musical_bonus += 20
        warnings.append(f"⚠️ MODULATION : Transition vers {key_stable} détectée en fin de morceau.")

    # Vérification du Mode Mineur (Sensible)
    root_idx = NOTES_LIST.index(final_decision.split()[0])
    if "minor" in final_decision:
        if check_leading_tone(chroma_stable, root_idx):
            musical_bonus += 15
        else:
            warnings.append("❓ AMBIGUÏTÉ : Mode mineur sans sensible (possible relatif majeur).")

    # Diagnostic de qualité
    avg_conf = np.mean([d['Confiance'] for d in timeline_data])
    if avg_conf < 50:
        warnings.append("🎸 DISTORSION : Signal bruyant ou complexe.")

    # Cadence V-I (Sur les deux notes dominantes globales)
    is_cadence, confirmed_root = detect_perfect_cadence(n1, n2)
    if is_cadence:
        final_decision = confirmed_root
        musical_bonus += 15

    # Calcul confiance finale
    total_conf = min(int((counts[final_decision]/len(votes)*100) + musical_bonus), 100)

    # Couleurs et Labels
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
        "is_cadence": is_cadence
    }

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="RCDJ228 ULTIME KEY PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 RCDJ228 ULTIME KEY PRO")
st.subheader("Analyseur Harmonique (Trim, Zone Stable & Arbitrage V-I)")

files = st.file_uploader("📂 DEPOSEZ VOS FICHIERS AUDIO", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    for f in reversed(files):
        # On lit les bytes une seule fois
        file_bytes = f.read()
        res = get_full_analysis(file_bytes, f.name)
        
        # Boîte de résultat principale
        st.markdown(f"""
            <div style="background:{res['recommended']['bg']}; padding:35px; border-radius:20px; color:white; text-align:center; margin:20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                <h2 style="margin:0; opacity:0.9;">{res['file_name']}</h2>
                <h1 style="font-size:5.5em; margin:15px 0; font-weight:900;">{res['recommended']['note']}</h1>
                <h2 style="margin:0; font-weight:700;">{get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% PRÉCISION</h2>
                <p style="margin-top:10px; text-transform:uppercase; letter-spacing:3px; font-size:0.9em;">{res['recommended']['label']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Tempo Estimé", f"{res['tempo']} BPM")
            if res['is_cadence']:
                st.success("🎹 Cadence V-I Détectée")
            
            if res['warnings']:
                st.warning("🔍 Diagnostic Technique")
                for w in res['warnings']:
                    st.write(f"- {w}")
            else:
                st.info("✅ Structure harmonique stable.")

        with col2:
            df_tl = pd.DataFrame(res['timeline'])
            fig = px.scatter(df_tl, x="Temps", y="Note", color="Confiance", size="Confiance", 
                             title=f"Stabilité Harmonique : {res['file_name']}", 
                             color_continuous_scale="Viridis", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

gc.collect()
