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

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M3 - Pro Precision", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

# Profils de corrélation (Krumhansl, Temperley, Bellman)
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { 
        background: #1a1c24; 
        padding: 15px; 
        border-radius: 15px; 
        border: 1px solid #333; 
        text-align: center; 
        min-height: 110px; 
        display: flex; 
        flex-direction: column; 
        justify-content: center;
        transition: 0.3s;
    }
    .metric-container:hover { border-color: #6366F1; }
    .metric-label { font-size: 0.8em; color: #888; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 40px; 
        border-radius: 25px; 
        text-align: center; 
        margin: 15px 0; 
        border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        color: white;
    }
    .profile-tag { 
        background: rgba(99, 102, 241, 0.1); 
        color: #a5b4fc;
        padding: 3px 10px; 
        border-radius: 6px; 
        font-size: 0.75em; 
        margin: 2px; 
        display: inline-block;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .streamlit-expanderHeader { background-color: #1a1c24 !important; border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE TRAITEMENT ---

def apply_perceptual_filter(y, sr):
    nyq = 0.5 * sr
    low, high = 80 / nyq, 8000 / nyq # Élargi pour capter plus de nuances
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_bass_profile(y, sr):
    """Extrait la note de basse dominante pour aider à trancher entre relatifs."""
    nyq = 0.5 * sr
    low_cut = 150 / nyq 
    b, a = butter(2, low_cut, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def get_enhanced_chroma_24(y, sr, tuning):
    """Analyse à 24 bins pour une précision accrue, puis réduction à 12."""
    y_harm = librosa.effects.harmonic(y, margin=6.0)
    # Utilisation de 24 bins (2 par demi-ton)
    chroma_raw = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, tuning=tuning, 
        n_chroma=24, bins_per_octave=24, 
        fmin=librosa.note_to_hz('C2')
    )
    # Repliement des 24 bins vers 12 en prenant la moyenne des micro-tons
    chroma_12 = (chroma_raw[::2, :] + chroma_raw[1::2, :]) / 2
    
    # Nettoyage et débruitage
    chroma_12 = librosa.decompose.nn_filter(chroma_12, aggregate=np.median, metric='cosine')
    return np.power(chroma_12, 2.0)

def solve_key_logic(chroma_vector, bass_vector=None):
    best_score, best_key, best_root, best_mode = -1, "", 0, "major"
    winners = {}
    
    # Normalisation du vecteur chroma
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)

    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                
                # --- STRATÉGIE DE LA BASSE ---
                # Si la note i est présente en basse, on booste le score
                if bass_vector is not None:
                    bass_presence = bass_vector[i]
                    if bass_presence > np.mean(bass_vector):
                        score *= 1.05 # Bonus de 5% pour la cohérence basse-tonique
                
                # --- STRATÉGIE DE RELATIF ---
                # On pénalise légèrement le mineur pour éviter les faux positifs 
                # sauf si la basse confirme la tonique mineure.
                total_score = score
                if p_name == "bellman": total_score *= 1.2
                
                note_str = f"{NOTES_LIST[i]} {mode}"
                if total_score > p_max:
                    p_max, p_note = total_score, note_str
                
                if total_score > best_score:
                    best_score, best_root, best_mode, best_key = total_score, i, mode, note_str
                    
        winners[p_name] = p_note
        
    return {"key": best_key, "score": best_score, "root": best_root, "mode": best_mode, "details": winners}

def get_camelot(key_str):
    try:
        n, m = key_str.split(" ")
        return BASE_CAMELOT_MINOR.get(n, "??") if m == 'minor' else BASE_CAMELOT_MAJOR.get(n, "??")
    except: return "??"

# --- COMPOSANTS UI ---

def play_chord_button(note_mode, uid):
    if not note_mode or " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s").replace("-","_")
    return components.html(f"""
    <div style="height:100%; display:flex; align-items:center;">
    <button id="{js_id}" style="background:linear-gradient(90deg, #6366F1, #8B5CF6); color:white; border:none; border-radius:12px; padding:15px; cursor:pointer; font-weight:bold; width:100%; font-family: sans-serif; box-shadow: 0 4px 15px rgba(99,102,241,0.4); transition: 0.2s;">
        🔊 TESTER {n} {m.upper()}
    </button>
    </div>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    document.getElementById('{js_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{m}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(it => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, it/12), now);
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.2, now+0.1); g.gain.exponentialRampToValueAtTime(0.01, now+1.5);
            o.connect(g); g.connect(ctx.destination); o.start(now); o.stop(now+1.5);
        }});
    }};
    </script>""", height=110)

# --- ANALYSE PRINCIPALE ---

@st.cache_data(show_spinner=False, ttl=3600)
def process_audio(file_bytes, file_name):
    try:
        with io.BytesIO(file_bytes) as b:
            y, sr = librosa.load(b, sr=22050, mono=True)
            
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        y_filt = apply_perceptual_filter(y, sr)
        
        step, timeline = 8, []
        votes = Counter()
        
        for start in range(0, int(duration) - step, step):
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            if np.max(np.abs(y_seg)) < 0.01: continue 
            
            # Application de la stratégie 24 bins et Bass-priority
            chroma = get_enhanced_chroma_24(y_seg, sr, tuning)
            bass = get_bass_profile(y_seg, sr)
            
            res = solve_key_logic(np.mean(chroma, axis=1), bass)
            
            # Pondération par la confiance temporelle
            weight = int(res['score'] * 100)
            votes[res['key']] += weight
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        if not timeline: return {"error": "Audio trop court ou silencieux"}

        # Décision finale par vote majoritaire pondéré
        final_key = votes.most_common(1)[0][0]
        avg_conf = int(pd.DataFrame(timeline)[pd.DataFrame(timeline)['Note'] == final_key]['Conf'].mean())
        
        # Tempo et Profil final global
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        full_chroma = get_enhanced_chroma_24(y, sr, tuning)
        full_bass = get_bass_profile(y, sr)
        final_details = solve_key_logic(np.mean(full_chroma, axis=1), full_bass)

        # Graphique de stabilité
        df_tl = pd.DataFrame(timeline)
        fig = px.line(df_tl, x="Temps", y="Note", markers=True, 
                      category_orders={"Note": NOTES_ORDER}, 
                      title=f"Analyse Harmonique 24-bins : {file_name}",
                      template="plotly_dark")
        
        fig.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font_color="white")

        img_bytes = fig.to_image(format="png", width=1000, height=500)
        
        output = {
            "name": file_name, "tempo": int(float(tempo)), "tuning": round(tuning, 2),
            "key": final_key, "camelot": get_camelot(final_key), "conf": avg_conf,
            "details": final_details['details'], "timeline": timeline,
            "plot": img_bytes 
        }
        
        del y, y_filt, full_chroma, fig, df_tl
        gc.collect()
        return output
        
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---

st.title("🎧 RCDJ228 M3 - Analyser")
uploaded_files = st.file_uploader("📂 Chargez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    pbar = st.progress(0)
    for i, f in enumerate(uploaded_files):
        file_data = f.read()
        res = process_audio(file_data, f.name)
        
        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}")
            continue

        with st.expander(f"📊 RÉSULTAT : {res['name']}", expanded=True):
            bg_grad = "linear-gradient(135deg, #4F46E5, #7C3AED)" if res['conf'] > 70 else "linear-gradient(135deg, #1F2937, #111827)"
            st.markdown(f"""
                <div class="final-decision-box" style="background:{bg_grad};">
                    <p style="margin:0; opacity:0.8; letter-spacing:3px; font-weight:300;">STRATÉGIE 24-BINS + BASS-MATCH</p>
                    <h1 style="font-size:5.5em; margin:10px 0; font-weight:900; line-height:1;">{res['key']}</h1>
                    <p style="margin:0; font-size:1.5em; font-weight:600; opacity:0.9;">
                        CAMELOT: {res['camelot']} <span style="margin:0 20px; opacity:0.3;">|</span> CONFIANCE: {res['conf']}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: 
                st.markdown(f"""<div class="metric-container"><div class="metric-label">Tempo estimé</div><div class="value-custom">{res["tempo"]} BPM</div></div>""", unsafe_allow_html=True)
            with c2: 
                play_chord_button(res['key'], f.name)
            with c3: 
                tags_html = "".join([f"<span class='profile-tag'>{p}: {v}</span>" for p, v in res['details'].items()])
                st.markdown(f"""<div class="metric-container"><div class="metric-label">Stabilité Algorithmique</div><div style="margin-top:5px;">{tags_html}</div></div>""", unsafe_allow_html=True)
            
            st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", markers=True, category_orders={"Note": NOTES_ORDER}, template="plotly_dark").update_layout(height=350), use_container_width=True)

            # Envoi Telegram
            try:
                details_text = "\n".join([f"• *{p.capitalize()}*: `{v}`" for p, v in res['details'].items()])
                cap = (f"🎧 *RAPPORT PRÉCISION M3*\n📂 `{res['name']}`\n\n🎹 *KEY : {res['key']}*\n🎼 Camelot : `{res['camelot']}`\n🔥 Confidence : `{res['conf']}%`\n\n⏱ *METRICS :*\n• BPM : `{res['tempo']}`\n• Tuning : `{res['tuning']} Hz`\n\n🔬 *DÉTAILS :*\n{details_text}")
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", files={'photo': res['plot']}, data={'chat_id': CHAT_ID, 'caption': cap, 'parse_mode': 'Markdown'})
            except: pass

        pbar.progress((i + 1) / len(uploaded_files))
        gc.collect()

if st.sidebar.button("🧹 Nettoyer le cache"):
    st.cache_data.clear()
    st.rerun()
