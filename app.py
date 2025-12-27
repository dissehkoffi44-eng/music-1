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

# --- CONFIGURATION SÉCURISÉE & SECRETS ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 MUSIC KEY 1", page_icon="🎧", layout="wide")

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .metric-container { background: #1a1c24; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #888; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 45px; border-radius: 20px; text-align: center; margin: 10px 0; 
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
    }
    .solid-note-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed #6366F1;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = ['C major', 'C minor', 'C# major', 'C# minor', 'D major', 'D minor', 
               'D# major', 'D# minor', 'E major', 'E minor', 'F major', 'F minor', 
               'F# major', 'F# minor', 'G major', 'G minor', 'G# major', 'G# minor', 
               'A major', 'A minor', 'A# major', 'A# minor', 'B major', 'B minor']

PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- FONCTIONS LOGIQUES ---

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def detect_perfect_cadence(n1, n2):
    try:
        root1, root2 = n1.split()[0], n2.split()[0]
        idx1, idx2 = NOTES_LIST.index(root1), NOTES_LIST.index(root2)
        if (idx1 + 7) % 12 == idx2: return True, n1
        if (idx2 + 7) % 12 == idx1: return True, n2
        return False, n1
    except: return False, n1

def detect_relative_key(n1, n2):
    try:
        c1, c2 = get_camelot_pro(n1), get_camelot_pro(n2)
        if c1 == "??" or c2 == "??": return False, n1
        val1, mod1 = int(c1[:-1]), c1[-1]
        val2, mod2 = int(c2[:-1]), c2[-1]
        if val1 == val2 and mod1 != mod2:
            return True, (n1 if mod1 == 'A' else n2)
        return False, n1
    except: return False, n1

def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        response = requests.post(url, files=files, data=data, timeout=45).json()
        return response.get("ok", False)
    except: return False

def get_sine_witness(note_mode_str, key_suffix=""):
    if note_mode_str == "N/A": return ""
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #888;">{note} {mode[:3].upper()} PIANO</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null;
    let activeNodes = [];
    function playPianoNote(freq, startTime) {{
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(0.4, startTime + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, startTime + 2.5);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(startTime);
        osc.stop(startTime + 2.6);
        return {{osc, gain}};
    }}
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            const isMinor = '{mode}' === 'minor' || '{mode}' === 'dorian';
            const intervals = isMinor ? [0, 3, 7, 12] : [0, 4, 7, 12];
            const now = audioCtx.currentTime;
            intervals.forEach((interval, index) => {{
                const freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                activeNodes.push(playPianoNote(freq, now + (index * 0.02)));
            }});
            setTimeout(() => {{ this.innerText = '▶'; this.style.background = '#6366F1'; activeNodes = []; }}, 2500);
        }} else {{
            activeNodes.forEach(node => {{
                node.gain.gain.cancelScheduledValues(audioCtx.currentTime);
                node.gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
                setTimeout(() => node.osc.stop(), 100);
            }});
            this.innerText = '▶'; this.style.background = '#6366F1';
            activeNodes = [];
        }}
    }};
    </script>
    """, height=40)

def analyze_segment(y, sr, tuning=0.0):
    nyq = 0.5 * sr
    low, high = 60 / nyq, 1000 / nyq
    b, a = butter(4, [low, high], btype='band')
    y_filtered = lfilter(b, a, y)
    
    chroma = librosa.feature.chroma_cens(y=y_filtered, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES_LIST[i]} {mode}"
    return res_key, best_score

@st.cache_data(show_spinner="Analyse Harmonique Profonde...", max_entries=20)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.harmonic(y)
    duration = librosa.get_duration(y=y, sr=sr)
    
    timeline_data, votes = [], []
    step = 8 
    
    progress_bar = st.progress(0)
    for i, start_t in enumerate(range(0, int(duration) - step, step)):
        y_seg = y_harm[int(start_t*sr):int((start_t+step)*sr)]
        key_seg, score_seg = analyze_segment(y_seg, sr, tuning=tuning_offset)
        if key_seg:
            votes.append(key_seg)
            timeline_data.append({
                "Temps": start_t, 
                "Note": key_seg, 
                "Camelot": get_camelot_pro(key_seg),
                "Confiance": round(float(score_seg) * 100, 1)
            })
    progress_bar.progress(min(start_t / duration, 1.0))
    progress_bar.empty()

    if not votes: return None

    # Calcul de la note solide
    df_tl = pd.DataFrame(timeline_data)
    df_tl['is_stable'] = df_tl['Note'] == df_tl['Note'].shift(1)
    stability_scores = {}
    for note in df_tl['Note'].unique():
        note_mask = df_tl['Note'] == note
        count = note_mask.sum()
        avg_conf = df_tl[note_mask]['Confiance'].mean()
        repos_bonus = df_tl[note_mask & df_tl['is_stable']].shape[0] * 1.5
        stability_scores[note] = (count * 0.4) + (avg_conf * 0.3) + (repos_bonus * 0.3)

    note_solide = max(stability_scores, key=stability_scores.get)
    solid_conf = int(df_tl[df_tl['Note'] == note_solide]['Confiance'].mean())

    # Logique de décision finale
    counts = Counter(votes)
    top_votes = counts.most_common(2)
    n1 = top_votes[0][0]
    n2 = top_votes[1][0] if len(top_votes) > 1 else n1
    
    is_relative, relative_preferred = detect_relative_key(n1, n2)
    musical_bonus = 0
    if is_relative:
        musical_bonus += 20
        n1 = relative_preferred

    is_cadence, confirmed_root = detect_perfect_cadence(n1, n2)
    if is_cadence:
        musical_bonus += 30
        n1 = confirmed_root

    purity = (counts[n1] / len(votes)) * 100
    avg_conf_n1 = df_tl[df_tl['Note'] == n1]['Confiance'].mean()
    musical_score = min(int((purity * 0.5) + (avg_conf_n1 * 0.5) + musical_bonus), 100)

    if musical_score > 80: bg = "linear-gradient(135deg, #1D976C 0%, #93F9B9 100%)"; label = "NOTE INDISCUTABLE"
    elif musical_score > 60: bg = "linear-gradient(135deg, #2193B0 0%, #6DD5ED 100%)"; label = "NOTE TRÈS FIABLE"
    else: bg = "linear-gradient(135deg, #FF512F 0%, #DD2476 100%)"; label = "ANALYSE COMPLEXE"

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res = {
        "file_name": file_name,
        "recommended": {"note": n1, "conf": musical_score, "bg": bg, "label": label},
        "tempo": int(float(tempo)),
        "timeline": timeline_data,
        "note_solide": note_solide, "solid_conf": solid_conf,
        "n1": n1, "c1": int(purity), "n2": n2, "c2": int((counts[n2]/len(votes))*100),
        "is_cadence": is_cadence, "is_relative": is_relative,
        "energy": int(np.clip(musical_score/10, 1, 10))
    }
    
    del y, y_harm; gc.collect()
    return res

# --- INTERFACE UTILISATEUR ---

st.title("🎧 RCDJ228 MUSIC KEY 1")

with st.sidebar:
    st.header("⚙️ SYSTÈME")
    if st.button("🧹 VIDER LE CACHE & RAM"):
        st.session_state.processed_files = {}
        st.session_state.order_list = []
        st.cache_data.clear()
        gc.collect()
        st.rerun()

if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []

files = st.file_uploader("📂 DEPOSEZ VOS FICHIERS AUDIO", accept_multiple_files=True, type=['mp3', 'wav', 'flac'])

tabs = st.tabs(["🚀 ANALYSEUR", "📜 HISTORIQUE"])

with tabs[0]:
    if files:
        for f in reversed(files):
            fid = f"{f.name}_{f.size}"
            if fid not in st.session_state.processed_files:
                f_bytes = f.read()
                res = get_full_analysis(f_bytes, f.name)
                if res:
                    # --- REPORTING TELEGRAM DÉTAILLÉ AVEC MISE EN AVANT NOTE SOLIDE ---
                    status_icon = "🟢" if res['recommended']['conf'] > 80 else "🟡" if res['recommended']['conf'] > 60 else "🔴"
                    tg_cap = (
                        f"🎵 *RAPPORT HARMONIQUE PRO*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📄 *FICHIER* : `{res['file_name']}`\n"
                        f"🎹 *CLÉ RECOMMANDÉE* : `{res['recommended']['note'].upper()}`\n"
                        f"🎡 *CAMELOT* : `{get_camelot_pro(res['recommended']['note'])}`\n"
                        f"🎯 *FIABILITÉ* : `{res['recommended']['conf']}%` {status_icon}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"💎 *NOTE LA PLUS SOLIDE* : `{res['note_solide'].upper()}`\n"
                        f"📈 *STABILITÉ* : `{res['solid_conf']}%` 🔥\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"🎼 *AUTRES DÉTAILS* :\n"
                        f"• Tempo détecté : `{res['tempo']} BPM`\n"
                        f"• Énergie globale : `{res['energy']}/10`\n"
                        f"• Cadence Parfaite : `{'✅ Oui' if res['is_cadence'] else '❌ Non'}`\n"
                        f"• Tonalité Relative : `{'✅ Détectée' if res['is_relative'] else '❌ Non'}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📢 *Analyse par RCDJ228 AI*"
                    )
                    upload_to_telegram(io.BytesIO(f_bytes), f.name, tg_cap)
                    st.session_state.processed_files[fid] = res
                    st.session_state.order_list.insert(0, fid)

        for fid in st.session_state.order_list:
            res = st.session_state.processed_files[fid]
            with st.expander(f"📊 {res['file_name']}", expanded=True):
                st.markdown(f"""
                    <div class="final-decision-box" style="background:{res['recommended']['bg']};">
                        <h2 style="margin:0; opacity:0.8; font-weight:300;">{res['recommended']['label']}</h2>
                        <h1 style="font-size:5em; margin:10px 0; font-weight:900;">{res['recommended']['note']}</h1>
                        <h2 style="margin:0; font-weight:700;">CAMELOT : {get_camelot_pro(res['recommended']['note'])} • {res['recommended']['conf']}% FIABILITÉ</h2>
                    </div>
                """, unsafe_allow_html=True)

                # --- MISE EN AVANT DE LA NOTE SOLIDE ---
                st.markdown(f"""
                    <div class="solid-note-box">
                        <div style="color: #888; font-size: 0.8em; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Analyse de Stabilité Temporelle</div>
                        <div style="font-size: 2.2em; font-weight: 800; color: #6366F1; margin: 5px 0;">💎 NOTE SOLIDE : {res['note_solide']}</div>
                        <div style="display: inline-block; background: #6366F1; color: white; padding: 2px 12px; border-radius: 20px; font-size: 0.9em; font-weight: bold;">
                            Score de Stabilité : {res['solid_conf']}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">AUDITION</div><div class="value-custom">TEST</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["note_solide"], f"sol_{fid}")
                with c3:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">ÉNERGIE</div><div class="value-custom">{res["energy"]}/10</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">CADENCE</div><div class="value-custom">{"OUI" if res["is_cadence"] else "NON"}</div></div>', unsafe_allow_html=True)

                df_tl = pd.DataFrame(res['timeline'])
                fig = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", title="Stabilité Harmonique")
                fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray':NOTES_ORDER}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    if st.session_state.processed_files:
        hist_data = [{"Fichier": r["file_name"], "Note": r['recommended']['note'], "Camelot": get_camelot_pro(r['recommended']['note']), "BPM": r["tempo"], "Confiance": f"{r['recommended']['conf']}%"} for r in st.session_state.processed_files.values()]
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True)

gc.collect()
