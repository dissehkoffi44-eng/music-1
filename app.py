import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import gc

# --- CONFIGURATION & CONSTANTES ---
# Respect strict de l'instruction : F# MINOR = 11A
BASE_CAMELOT_MINOR = {
    'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A',
    'C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A',
    'Db':'12A','C#':'12A'
}
BASE_CAMELOT_MAJOR = {
    'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B',
    'Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B',
    'D':'10B','A':'11B','E':'12B'
}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils de Krumhansl-Kessler pour la corrélation
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS LOGIQUES ---

def get_camelot_pro(key_mode_str):
    """Convertit la note textuelle en code Camelot Wheel."""
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode == 'minor':
            return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except:
        return "??"

def get_bass_priority(y, sr):
    """Analyse l'énergie des fréquences basses pour confirmer la tonique (root note)."""
    # Utilisation d'une fenêtre n_fft large pour une meilleure résolution dans les graves
    chroma_bass = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=1024)
    return np.mean(chroma_bass, axis=1)

def analyze_segment(y, sr, tuning=0.0):
    """Analyse un segment audio et retourne la clé la plus probable avec un score."""
    if len(y) < 512:
        return None, 0.0, None
    
    # Extraction Chroma CENS (robuste aux variations d'amplitude)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    
    # Analyse de la structure des basses
    bass_boost = get_bass_priority(y, sr)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            # Corrélation statistique
            corr = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            # Pondération par la présence en basse (favorise la vraie tonique)
            score = corr + (0.2 * bass_boost[i])
            
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
                
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse de la tonique en cours...", max_entries=20)
def get_full_analysis(file_bytes, file_name):
    """Moteur d'analyse complet sur l'ensemble du fichier."""
    # Chargement
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    
    # Extraction de la composante harmonique uniquement (ignore percussions/bruit)
    y_harm = librosa.effects.harmonic(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    timeline_data, votes = [], []
    step = 6 # Fenêtres de 6 secondes pour capter les changements
    
    for start_t in range(0, int(duration) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg, _ = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            timeline_data.append({
                "Temps": start_t, 
                "Note": key_seg, 
                "Confiance": round(float(score_seg) * 100, 1)
            })

    # Décision finale basée sur le vote majoritaire (stabilité)
    counts = Counter(votes)
    if not counts:
        return None
        
    final_decision = counts.most_common(1)[0][0]
    
    # Validation par l'Outro (souvent la tonique finale)
    y_outro = y_harm[-int(min(12, duration)*sr):]
    key_outro, _, _ = analyze_segment(y_outro, sr, tuning=tuning_offset)
    
    # Calcul de confiance final
    stability_ratio = (counts[final_decision] / len(votes))
    total_conf = int(stability_ratio * 100)
    
    # Bonus de confiance si l'outro confirme la tendance générale
    if key_outro == final_decision:
        total_conf = min(total_conf + 15, 100)

    # Définition du style visuel
    if total_conf > 80: bg = "linear-gradient(135deg, #1D976C 0%, #93F9B9 100%)"
    elif total_conf > 60: bg = "linear-gradient(135deg, #2193B0 0%, #6DD5ED 100%)"
    else: bg = "linear-gradient(135deg, #FF512F 0%, #DD2476 100%)"

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "file_name": file_name,
        "recommended": {"note": final_decision, "conf": total_conf, "bg": bg},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "outro_match": (key_outro == final_decision)
    }

# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="RCDJ228 KEY PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 RCDJ228 ULTIME KEY PRO")
st.write("Analyseur de précision focalisé sur la **stabilité de la tonique**.")

files = st.file_uploader("📂 DEPOSEZ VOS FICHIERS AUDIO", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    for f in reversed(files):
        res = get_full_analysis(f.read(), f.name)
        
        if res:
            # Grand bandeau de résultat
            st.markdown(f"""
                <div style="background:{res['recommended']['bg']}; padding:45px; border-radius:20px; color:white; text-align:center; margin:20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                    <h2 style="margin:0; opacity:0.8; font-weight:300;">{res['file_name']}</h2>
                    <h1 style="font-size:6em; margin:10px 0; font-weight:900; letter-spacing:-2px;">{res['recommended']['note']}</h1>
                    <h2 style="margin:0; font-weight:700;">CAMELOT : {get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% FIABILITÉ</h2>
                </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Tempo", f"{res['tempo']} BPM")
                if res['outro_match']:
                    st.success("✅ Tonique confirmée par la fin du morceau.")
                else:
                    st.warning("⚠️ La fin du morceau varie (Modulation possible).")
                
                if res['recommended']['conf'] < 50:
                    st.info("💡 Conseil : Le morceau semble complexe ou riche en harmoniques.")

            with col2:
                # Graphique de stabilité
                df_tl = pd.DataFrame(res['timeline'])
                fig = px.scatter(
                    df_tl, x="Temps", y="Note", color="Confiance", 
                    title="Analyse de la stabilité temporelle",
                    color_continuous_scale="Viridis", height=300
                )
                fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
        
        gc.collect()

# Nettoyage final
gc.collect()
