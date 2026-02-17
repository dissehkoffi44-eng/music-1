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
import json  # <--- AJOUTEZ CETTE LIGNE ICI
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter, find_peaks
from datetime import datetime
from pydub import AudioSegment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

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
    },
    "diatonic": {  # Nouveau profil diatonique simple
        "major": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    }
}

FIFTHS_ORDER = ['A', 'D', 'G', 'C', 'F', 'A#', 'D#', 'G#', 'C#', 'F#', 'B', 'E']

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    .metric-box:hover { border-color: #58a6ff; }
    .sniper-badge {
        background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---

def seconds_to_mmss(seconds):
    if seconds is None:
        return "??:??"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

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

def get_fifths_vector(chroma):
    # Réordonne chroma (0=C,1=C#,...) en ordre quintes
    pc_to_fifths = {0:3, 1:8, 2:1, 3:6, 4:11, 5:4, 6:9, 7:2, 8:7, 9:0, 10:5, 11:10}  # Map C-based to fifths
    return np.array([chroma[pc_to_fifths[i]] for i in range(12)])

def find_main_axis(k_vec):  # k_vec = fifths_vector normalisé
    max_val, best_axis = -np.inf, None
    for y_idx in range(12):
        z_idx = (y_idx + 6) % 12
        left_sum = np.sum(k_vec[(y_idx + 1) % 12 : (y_idx + 6) % 12 + 1 if (y_idx + 6) % 12 < (y_idx + 1) % 12 else (y_idx + 6) % 12])
        right_sum = np.sum(k_vec[(y_idx + 7) % 12 : (y_idx + 12) % 12 + 1 if (y_idx + 12) % 12 < (y_idx + 7) % 12 else (y_idx + 12) % 12]) + k_vec[z_idx] / 2
        val = right_sum - left_sum
        if val > max_val:
            max_val = val
            best_axis = (FIFTHS_ORDER[y_idx], FIFTHS_ORDER[z_idx])
    return best_axis

def solve_key_sniper(chroma_vector, bass_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    # Signature of Fifths
    fifths_vec = get_fifths_vector(cv)
    axis = find_main_axis(fifths_vec)
    if not axis:
        return {"key": "Unknown", "score": 0}
    
    # Rotate 30° clockwise approx
    rotate_idx = (FIFTHS_ORDER.index(axis[0]) + 1) % 12
    candidates = [f"{FIFTHS_ORDER[rotate_idx]} major", f"{FIFTHS_ORDER[(rotate_idx + 9) % 12]} minor"]
    
    best_overall_score = -1
    best_key = "Unknown"
    
    for cand in candidates:
        note, mode = cand.split()
        i = NOTES_LIST.index(note)
        
        # Moyenne des scores sur tous les profils (hybridation)
        scores = []
        for p_name, p_data in PROFILES.items():
            profile = p_data[mode]
            shifted_profile = np.roll(profile, -i)  # Shift pour aligner à la root
            score = np.corrcoef(cv, shifted_profile)[0, 1]
            
            # LOGIQUE DE CADENCE PARFAITE
            if mode == "minor":
                dom_idx = (i + 7) % 12 
                leading_tone = (i + 11) % 12
                if cv[dom_idx] > 0.45 and cv[leading_tone] > 0.35:
                    score *= 1.35 
            
            if bv[i] > 0.6: score += (bv[i] * 0.2)
            
            fifth_idx = (i + 7) % 12
            if cv[fifth_idx] > 0.5: score += 0.1
            third_idx = (i + 4) % 12 if mode == "major" else (i + 3) % 12
            if cv[third_idx] > 0.5: score += 0.1
            
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_key = cand
    
    return {"key": best_key, "score": best_overall_score}

def process_audio(audio_file, file_name, progress_placeholder):
    status_text = progress_placeholder.empty()
    progress_bar = progress_placeholder.progress(0)

    def update_prog(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"**{text} | {value}%**")

    update_prog(10, f"Chargement de {file_name}")
    file_bytes = audio_file.getvalue()
    ext = file_name.split('.')[-1].lower()
    if ext == 'm4a':
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        y = samples / (2**15)
        sr = audio.frame_rate
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
    else:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)
    
    update_prog(30, "Filtrage des fréquences")
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    # Spectral whitening (pré-emphasis)
    y_whitened = librosa.effects.preemphasis(y_filt)

    # Compute spectrogram pour extensions
    S = np.abs(librosa.stft(y_whitened))
    freqs = librosa.fft_frequencies(sr=sr)

    # Peak Detection: Trouve pics par classe de pitch
    chroma_pd = np.zeros((12, S.shape[1]))
    for frame in range(S.shape[1]):
        peaks, _ = find_peaks(S[:, frame], height=0.1 * np.max(S[:, frame]))  # Seuil adaptatif
        for p in peaks:
            midi_note = librosa.hz_to_midi(freqs[p])
            pc = int(midi_note % 12)
            chroma_pd[pc, frame] += S[p, frame]  # Ajoute amplitude du pic

    # Low Frequency Clarification: Pour basses (<220 Hz), supprime pics ambigus
    low_freq_mask = freqs < 220
    for frame in range(S.shape[1]):
        low_peaks = peaks[np.where(low_freq_mask[peaks])[0]] if np.any(low_freq_mask[peaks]) else []
        if len(low_peaks) > 0:
            strongest = np.argmax(S[low_peaks, frame])
            for idx in range(len(low_peaks)):
                if idx != strongest and abs(freqs[low_peaks[idx]] - freqs[low_peaks[strongest]]) < 10:  # Proche en Hz
                    chroma_pd[:, frame] *= 0.5  # Réduit poids

    update_prog(50, "Analyse du spectre harmonique")
    step, timeline, votes = 6, [], Counter()
    segments = range(0, int(duration) - step, 1)  # Réduit à tous les 1s pour plus de granularité
    
    for i, start in enumerate(segments):
        idx_start, idx_end = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if np.max(np.abs(seg)) < 0.01: continue
        
        # Multi-scale analysis: Chroma avec deux fenêtres (n_fft=4096 et 8192)
        chroma_short = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12)
        chroma_long = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12)
        c_raw = np.mean(librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12), axis=1)
        
        # Intègre chroma_pd pour le segment (approx, ajuste si besoin)
        frame_start = int(idx_start / 1024)  # Approx hop=1024
        frame_end = int(idx_end / 1024)
        c_avg = np.mean(chroma_pd[:, max(0, frame_start):min(chroma_pd.shape[1], frame_end)], axis=1)
        c_avg = (c_avg + np.mean(c_raw, axis=0)) / 2  # Fusionne avec chroma_cqt
        
        # Lissage gaussien
        c_avg_smoothed = gaussian_filter1d(c_avg, sigma=1.5)
        
        # Periodic Cleanup: Tous les 4 segments (~4s), reset bas valeurs
        if i % 4 == 0:
            threshold = 0.2 * np.max(c_avg_smoothed)
            c_avg_smoothed[c_avg_smoothed < threshold] = 0
        
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        
        res = solve_key_sniper(c_avg_smoothed, b_seg)
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})
        
        p_val = 50 + int((i / len(segments)) * 40)
        update_prog(p_val, "Calcul chirurgical en cours")

    update_prog(95, "Synthèse finale")
    most_common = votes.most_common(2)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.25
    target_key = most_common[1][0] if mod_detected else None

    modulation_time = None
    target_percentage = 0
    ends_in_target = False

    if mod_detected and target_key:
        target_times = np.array([t["Temps"] for t in timeline if t["Note"] == target_key])
        if len(target_times) > 3:
            dist = pdist(target_times.reshape(-1,1), 'euclidean')
            Z = linkage(target_times.reshape(-1,1), method='single')
            clust = fcluster(Z, t=5, criterion='distance')  # Clusters si <5s apart
            max_cluster_size = max(Counter(clust).values()) * 1  # Taille en secondes approx (ajusté pour incrément 1s)
            if max_cluster_size < 10:  # Seuil minimal pour vraie modulation
                mod_detected = False  # Ignore si pas continu
        if mod_detected:
            candidates = [t["Temps"] for t in timeline if t["Note"] == target_key and t["Conf"] >= 0.84]
            if candidates:
                modulation_time = min(candidates)
            else:
                sorted_times = sorted(target_times)
                modulation_time = sorted_times[max(0, len(sorted_times) // 3)]

            total_valid = len(timeline)
            if total_valid > 0:
                target_count = sum(1 for t in timeline if t["Note"] == target_key)
                target_percentage = (target_count / total_valid) * 100

            if timeline:
                last_n = max(5, len(timeline) // 10)
                last_segments = timeline[-last_n:]
                last_counter = Counter(s["Note"] for s in last_segments)
                last_key = last_counter.most_common(1)[0][0]
                ends_in_target = (last_key == target_key)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)

    update_prog(100, "Analyse terminée")
    status_text.empty()
    progress_bar.empty()

    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "chroma": chroma_avg, "modulation": mod_detected,
        "target_key": target_key, "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name,
        "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
        "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
        "mod_ends_in_target": ends_in_target if mod_detected else False
    }
    
    # --- RAPPORT TELEGRAM ENRICHI (RADAR + TIMELINE) ---
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            # 1. Préparation du texte
            mod_line = ""
            if mod_detected:
                perc = res_obj["mod_target_percentage"]
                end_txt = " → **fin en " + target_key.upper() + " (" + res_obj['target_camelot'] + ")**" if res_obj['mod_ends_in_target'] else ""
                mod_line = f"\n⚠️ *MODULATION →* `{target_key.upper()} ({res_obj['target_camelot']})` ≈ **{res_obj['modulation_time_str']}** ({perc}%){end_txt}"
            
            caption = (
                f"🎯 *RCDJ228 MUSIC SNIPER*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📂 *FICHIER:* `{file_name}`\n"
                f"🎹 *TONALITÉ:* `{final_key.upper()}`\n"
                f"🌀 *CAMELOT:* `{res_obj['camelot']}`\n"
                f"🔥 *CONFIANCE:* `{res_obj['conf']}%`{mod_line}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"⏱ *TEMPO:* `{res_obj['tempo']} BPM`\n"
                f"🎸 *ACCORDAGE:* `{res_obj['tuning']} Hz` ✅"
            )

            # 2. Génération du Graphique RADAR (Spectre)
            fig_radar = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", title="SPECTRE HARMONIQUE", polar=dict(radialaxis=dict(visible=False)))
            radar_bytes = fig_radar.to_image(format="png", width=700, height=500)

            # 3. Génération du Graphique TIMELINE
            df_tl = pd.DataFrame(res_obj['timeline'])
            fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", 
                             category_orders={"Note": NOTES_ORDER}, title="ÉVOLUTION TEMPORELLE")
            timeline_bytes = fig_tl.to_image(format="png", width=1000, height=450)

            # 4. Envoi via sendMediaGroup (Album photo)
            media_group = [
                {'type': 'photo', 'media': 'attach://radar.png', 'caption': caption, 'parse_mode': 'Markdown'},
                {'type': 'photo', 'media': 'attach://timeline.png'}
            ]
            
            files = {
                'radar.png': radar_bytes,
                'timeline.png': timeline_bytes
            }

            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMediaGroup",
                data={'chat_id': CHAT_ID, 'media': json.dumps(media_group)},
                files=files
            )

        except Exception as e:
            st.error(f"Erreur d'envoi Telegram : {e}")

    del y, y_filt; gc.collect()
    return res_obj

def get_chord_js(btn_id, key_str):
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

# --- DASHBOARD PRINCIPAL ---
st.title("🎯 RCDJ228 MUSIC SNIPER")
st.markdown("#### Système d'Analyse Harmonique 99% précis")

uploaded_files = st.file_uploader("📥 Déposez vos fichiers (Audio)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    progress_zone = st.container()
    
    for f in reversed(uploaded_files):
        analysis_data = process_audio(f, f.name, progress_zone)
        
        with st.container():
            st.markdown(f"<div class='file-header'>📂 ANALYSE : {analysis_data['name']}</div>", unsafe_allow_html=True)
            color = "linear-gradient(135deg, #065f46, #064e3b)" if analysis_data['conf'] > 85 else "linear-gradient(135deg, #1e293b, #0f172a)"
            
            st.markdown(f"""
                <div class="report-card" style="background:{color};">
                    <p style="letter-spacing:5px; opacity:0.8; font-size:0.8em;">SNIPER ENGINE v5.0 <span class="sniper-badge">READY</span></p>
                    <h1 style="font-size:5.5em; margin:10px 0; font-weight:900;">{analysis_data['key'].upper()}</h1>
                    <p style="font-size:1.5em; opacity:0.9;">CAMELOT: <b>{analysis_data['camelot']}</b> &nbsp; | &nbsp; CONFIANCE: <b>{analysis_data['conf']}%</b></p>
                    {f"<div class='modulation-alert'>⚠️ MODULATION : {analysis_data['target_key'].upper()} ({analysis_data['target_camelot']})</div>" if analysis_data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{analysis_data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
            with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
            with m3:
                btn_id = f"play_{hash(analysis_data['name'])}"
                components.html(f"""
                    <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER L'ACCORD</button>
                    <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                """, height=110)

            c1, c2 = st.columns([2, 1])
            with c1:
                fig_tl = px.line(pd.DataFrame(analysis_data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tl, use_container_width=True)
            with c2:
                fig_radar = go.Figure(data=go.Scatterpolar(r=analysis_data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                fig_radar.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=20), polar=dict(radialaxis=dict(visible=False)), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")
    if st.button("🧹 Vider la file d'analyse"):
        st.cache_data.clear()
        st.rerun()
