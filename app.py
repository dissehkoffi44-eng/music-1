import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import gc

# --- CONFIGURATION & CONSTANTES ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# Mapping Camelot (Note : F# Minor est bien identifié comme 11A)
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
    leading_tone_idx = (key_index - 1) % 12
    return chroma_avg[leading_tone_idx] > np.mean(chroma_avg) * 1.2

def detect_perfect_cadence(n1, n2):
    """Détecte les relations de quinte dans les deux sens (V->I et I->V)"""
    try:
        r1, r2 = n1.split()[0], n2.split()[0]
        i1, i2 = NOTES_LIST.index(r1), NOTES_LIST.index(r2)
        
        # Cas 1 : n1 est la dominante (V), n2 est la tonique (I) -> Cadence Parfaite
        if (i1 + 7) % 12 == i2: 
            return True, n2, "V-I (Parfaite)"
            
        # Cas 2 : n1 est la tonique (I), n2 est la dominante (V) -> Demi-cadence
        if (i2 + 7) % 12 == i1: 
            return True, n1, "I-V (Demi-cadence)"
            
        return False, n1, None
    except: return False, n1, None

# --- MOTEUR d'ANALYSE ---

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

@st.cache_data(show_spinner="Analyse Harmonique Profonde (Saut 20s + 70% track)...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    # 1. Obtenir la durée totale d'abord
    full_duration = librosa.get_duration(path=io.BytesIO(file_bytes))
    
    # 2. Calculer la portion à charger (70% du morceau après un saut de 20s)
    # On s'assure que le morceau est assez long pour le saut de 20s
    start_offset = 20 if full_duration > 25 else 0
    duration_to_load = (full_duration - start_offset) * 0.70
    
    # 3. Charger uniquement la partie définie
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, offset=start_offset, duration=duration_to_load)
    
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, _ = librosa.effects.hpss(y)
    duration_loaded = librosa.get_duration(y=y, sr=sr)
    
    timeline_data, votes = [], []
    step = 8
    for start_t in range(0, int(duration_loaded) - step, step):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg, _ = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            # On ajoute le start_offset au temps pour l'affichage réel
            timeline_data.append({"Temps": start_t + start_offset, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})

    # Analyse de la fin de la portion chargée
    y_final = y_harm[-int(min(5, duration_loaded)*sr):] 
    key_final, score_final, chroma_final = analyze_segment(y_final, sr, tuning=tuning_offset)

    if not votes:
        return None

    counts = Counter(votes)
    n1 = counts.most_common(1)[0][0]
    n2 = counts.most_common(2)[1][0] if len(counts) > 1 else n1
    
    final_decision = n1
    musical_bonus = 0
    warnings = []
    cadence_info = None

    # Logique de modulation
    if n1 != key_final and score_final > 0.75:
        warnings.append(f"⚠️ MODULATION : Transition vers {key_final} détectée sur cette portion.")
        final_decision = key_final
        musical_bonus += 20

    # Vérification de la sensible
    root_idx = NOTES_LIST.index(final_decision.split()[0])
    if "minor" in final_decision:
        if check_leading_tone(chroma_final, root_idx):
            musical_bonus += 15
        else:
            warnings.append("❓ AMBIGUÏTÉ : Mode mineur sans sensible.")

    # Détection de Cadence
    is_cadence, confirmed_root, c_type = detect_perfect_cadence(n1, n2)
    if is_cadence:
        final_decision = confirmed_root
        musical_bonus += 20
        cadence_info = c_type
        warnings.append(f"🎼 HARMONIE : Structure {c_type} identifiée.")

    avg_conf = np.mean([d['Confiance'] for d in timeline_data])
    if avg_conf < 50:
        warnings.append("🎸 DISTORSION : Signal complexe ou bruyant.")

    total_conf = min(int((counts[final_decision]/len(votes)*100) + musical_bonus), 100)

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
        "cadence_type": cadence_info,
        "analyzed_range": f"De {start_offset}s à {int(start_offset + duration_loaded)}s"
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
st.subheader("Analyseur Harmonique (Saut 20s + Analyse 70% du fichier)")

files = st.file_uploader("📂 DEPOSEZ VOS FICHIERS AUDIO", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

if files:
    for f in files:
        file_bytes = f.read()
        res = get_full_analysis(file_bytes, f.name)
        
        if res:
            st.markdown(f"""
                <div style="background:{res['recommended']['bg']}; padding:35px; border-radius:20px; color:white; text-align:center; margin:20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
                    <h2 style="margin:0; opacity:0.9;">{res['file_name']}</h2>
                    <p style="font-size:0.8em; opacity:0.8;">Portion analysée : {res['analyzed_range']}</p>
                    <h1 style="font-size:5.5em; margin:15px 0; font-weight:900;">{res['recommended']['note']}</h1>
                    <h2 style="margin:0; font-weight:700;">{get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% PRÉCISION</h2>
                    <p style="margin-top:10px; text-transform:uppercase; letter-spacing:3px; font-size:0.9em;">{res['recommended']['label']}</p>
                </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Tempo Estimé", f"{res['tempo']} BPM")
                if res['is_cadence']: 
                    st.success(f"🎹 {res['cadence_type']} Détectée")
                if res['warnings']:
                    st.warning("🔍 Diagnostic Technique")
                    for w in res['warnings']: st.write(f"- {w}")
                else: st.info("✅ Aucune anomalie détectée.")

            with col2:
                df_tl = pd.DataFrame(res['timeline'])
                fig = px.scatter(df_tl, x="Temps", y="Note", color="Confiance", size="Confiance", 
                                 title="Stabilité Harmonique sur la portion chargée", color_continuous_scale="Viridis", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            st.divider()

gc.collect()
