import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="LE SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

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
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        transition: 0.3s; height: 100%;
    }
    .metric-box:hover { border-color: #4F46E5; box-shadow: 0 0 15px rgba(79, 70, 229, 0.3); }
    .sniper-badge {
        background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em; vertical-align: middle;
    }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4F46E5 , #10b981); }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector, tuning):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if bv[i] > 0.6: score += (bv[i] * 0.2)
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.5: score += 0.1
                third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                if cv[third_idx] > 0.5: score += 0.1
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_key = f"{NOTES_LIST[i]} {mode}"
                    
    return {"key": best_key, "score": best_overall_score}

def process_audio(audio_file, file_name):
    # Barre de progression personnalisée
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 1. Chargement (Gestion RAM : librosa lit directement l'objet UploadedFile)
    status_text.text(f"🚀 Initialisation du scan : {file_name}")
    y, sr = librosa.load(audio_file, sr=22050, mono=True)
    progress_bar.progress(20)

    # 2. Filtrage
    status_text.text("⚙️ Application des filtres Sniper M3...")
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)
    progress_bar.progress(40)

    # 3. Analyse Temporelle
    status_text.text("🔍 Scan des fréquences harmoniques...")
    step = 6
    timeline = []
    votes = Counter()
    
    # Calcul des segments pour la barre de progression
    segments = range(0, int(duration) - step, 2)
    total_segments = len(segments)
    
    for i, start in enumerate(segments):
        idx_start = int(start * sr)
        idx_end = int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        
        if np.max(np.abs(seg)) < 0.01: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_12 = (c_raw[::2, :] + c_raw[1::2, :]) / 2
        c_avg = np.mean(c_12, axis=1)
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        
        res = solve_key_sniper(c_avg, b_seg, tuning)
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})
        
        # Mise à jour progression (de 40% à 90%)
        p = 40 + int((i / total_segments) * 50)
        progress_bar.progress(p)

    # 4. Finalisation
    status_text.text("📊 Synthèse des données...")
    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    mod_detected = False
    target_key = None
    if len(most_common) > 1:
        if (votes[most_common[1][0]] / sum(votes.values())) > 0.25:
            mod_detected = True
            target_key = most_common[1][0]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_full = librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma_full, axis=1)

    output = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99),
        "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline,
        "chroma": chroma_avg,
        "modulation": mod_detected,
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    del y, y_filt; gc.collect()
    return output

# --- FONCTIONS UTILITAIRES ---

def get_chord_js(btn_id, key_str):
    if not key_str or " " not in key_str: return ""
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }};
    """

def send_telegram_report(data, fig_tl):
    if not (TELEGRAM_TOKEN and CHAT_ID): return
    try:
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        mod_text = f"⚠️ *MODULATION:* `{data['target_key'].upper()}` ({data['target_camelot']})" if data['modulation'] else "✅ *STABILITÉ:* Constante"
        
        caption = (f"🎯 *SNIPER M3 REPORT* - {now}\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"📂 *FICHIER:* `{data['name']}`\n"
                   f"🎹 *TONALITÉ:* `{data['key'].upper()}`\n"
                   f"🧭 *CAMELOT:* `{data['camelot']}`\n"
                   f"🔥 *CONFIANCE:* `{data['conf']}%`\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"⏱ *TEMPO:* `{data['tempo']} BPM`\n"
                   f"🎸 *ACCORDAGE:* `{data['tuning']} Hz`\n"
                   f"{mod_text}\n"
                   f"━━━━━━━━━━━━━━━━━━━━")
        
        img_bytes = fig_tl.to_image(format="png")
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                      files={'photo': ('graph.png', img_bytes)}, 
                      data={'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'})
    except Exception as e:
        st.sidebar.error(f"Erreur Telegram: {e}")

# --- INTERFACE UTILISATEUR ---

st.title("🎯 LE SNIPER")
st.markdown("#### Système d'Analyse Harmonique Militaire | Précision 99%")

uploaded_files = st.file_uploader("📥 Déposez vos fichiers (Audio / Wav / MP3)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    # On inverse la liste pour que les dernières analyses soient en haut
    for f in reversed(uploaded_files):
        # OPTIMISATION RAM : On passe l'objet 'f' directement au lieu de 'f.read()'
        analysis_data = process_audio(f, f.name)
        
        with st.container():
            # Affichage du rapport
            color = "linear-gradient(135deg, #065f46, #064e3b)" if analysis_data['conf'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"
            
            st.markdown(f"""
                <div class="report-card" style="background:{color};">
                    <p style="letter-spacing:5px; opacity:0.8; font-size:0.8em;">SNIPER ENGINE v4.0 <span class="sniper-badge">ACTIVE</span></p>
                    <h1 style="font-size:6em; margin:10px 0; font-weight:900;">{analysis_data['key'].upper()}</h1>
                    <p style="font-size:1.5em; opacity:0.9;">CAMELOT: <b>{analysis_data['camelot']}</b> &nbsp; | &nbsp; CONFIANCE: <b>{analysis_data['conf']}%</b></p>
                    {f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : {analysis_data['target_key'].upper()} ({analysis_data['target_camelot']})</div>" if analysis_data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)
            
            # Métriques Grid
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-box'><b>TEMPO ANALYSÉ</b><br><span style='font-size:2em; color:#10b981;'>{analysis_data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-box'><b>ACCORDAGE RÉEL</b><br><span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
            with m3:
                btn_id = f"play_{hash(analysis_data['name'])}"
                components.html(f"""
                    <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">🎹 ÉCOUTER L'ACCORD</button>
                    <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                """, height=110)

            # Visualisations
            c_left, c_right = st.columns([2, 1])
            with c_left:
                df_tl = pd.DataFrame(analysis_data['timeline'])
                fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", 
                                 category_orders={"Note": NOTES_ORDER}, title="Stabilité Harmonique (Timeline)")
                fig_tl.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tl, use_container_width=True)
            
            with c_right:
                fig_radar = go.Figure(data=go.Scatterpolar(r=analysis_data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                fig_radar.update_layout(template="plotly_dark", title="Empreinte Chromatique", polar=dict(radialaxis=dict(visible=False)))
                fig_radar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_radar, use_container_width=True)

            # Envoi Telegram Auto
            send_telegram_report(analysis_data, fig_tl)
            
            st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=100)
    st.header("Paramètres Sniper")
    st.info("Le Sniper offre une précision chirurgicale.")
    if st.button("🧹 Vider la mémoire"):
        st.cache_data.clear()
        st.rerun()
