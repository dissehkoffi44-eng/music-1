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
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import pickle
import os
import tempfile
import shutil

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
    }
}

# --- FIX PERFORMANCE : Précomputation des profils roulés ---
# Évite des milliers de np.roll() redondants dans la boucle d'analyse.
# Ces tableaux sont calculés une seule fois au démarrage de l'application.
PROFILES_ROLLED = {
    p_name: {
        mode: [np.roll(PROFILES[p_name][mode], i) for i in range(12)]
        for mode in ["major", "minor"]
    }
    for p_name in PROFILES
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

def arbitrage_expert_universel(chroma, key_cons, key_dom, cam_map):
    """
    Arbitrage Expert Universel v12 — Juge de Paix Spectral (Quinte + Tierce).

    Étend la couverture de l'arbitrage à toutes les paires de clés voisines
    sur le Camelot Wheel, incluant :
      - Voisins directs  (dist_num=1, même mode)
      - Relatifs         (dist_num=0, modes croisés A↔B)
      - Diagonaux        (dist_num=1, modes croisés)

    Algorithme :
      1. Distance numérique circulaire (0-6) + distance de mode (0 ou 1).
      2. Si dist_num <= 1 ET dist_mode <= 1 : duel spectral Quinte/Tierce pondéré.
         - Force harmonique = (quinte × 0.6) + (tierce × 0.4)
         - La dominante gagne si sa force dépasse la consonante de 10 %.
         - Sinon, la consonance est conservée (biais de stabilité).
      3. Si hors zone de duel : consonance par défaut.

    Paramètres
    ----------
    chroma   : np.ndarray — vecteur chroma global (12 valeurs normalisées).
    key_cons : str — clé Consonance (ex. "A minor").
    key_dom  : str — clé Dominante (ex. "E minor").
    cam_map  : dict — mapping clé → Camelot (ex. CAMELOT_MAP).

    Retourne
    --------
    str — la clé gagnante, ou key_cons si aucun arbitrage n'est possible.
    """
    cam_c = cam_map.get(key_cons)
    cam_d = cam_map.get(key_dom)

    # Garde-fou : clés inconnues du référentiel Camelot → pas d'arbitrage
    if not cam_c or not cam_d:
        return key_cons

    # Extraction des coordonnées Camelot
    val_c, mode_c = int(cam_c[:-1]), cam_c[-1]
    val_d, mode_d = int(cam_d[:-1]), cam_d[-1]

    # Distance numérique circulaire (roue de 12 positions, max 6)
    dist_num = abs(val_c - val_d)
    if dist_num > 6:
        dist_num = 12 - dist_num

    # Distance de mode (0 = même mode A/A ou B/B, 1 = modes croisés A/B)
    dist_mode = 0 if mode_c == mode_d else 1

    # --- ZONE DE DUEL UNIVERSELLE ---
    # Couvre : voisins directs, relatifs (enharmoniques de mode) et diagonaux
    if dist_num <= 1 and dist_mode <= 1:
        idx_c = NOTES_LIST.index(key_cons.split()[0])
        idx_d = NOTES_LIST.index(key_dom.split()[0])

        def get_harmonic_strength(idx, mode, chroma_vec):
            """Force harmonique pondérée : Quinte (60%) + Tierce (40%)."""
            quinte = chroma_vec[(idx + 7) % 12]
            tierce = chroma_vec[(idx + 3) % 12] if mode == 'minor' else chroma_vec[(idx + 4) % 12]
            return (quinte * 0.6) + (tierce * 0.4)

        force_c = get_harmonic_strength(idx_c, key_cons.split()[1], chroma)
        force_d = get_harmonic_strength(idx_d, key_dom.split()[1], chroma)

        # La dominante gagne si elle est spectralement plus claire d'au moins 10 %
        if force_d > force_c * 1.10:
            return key_dom

    # Par défaut : la Consonance reste la clé retenue
    return key_cons

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

def detect_harmonic_sections(y, sr, duration, step=6, min_harm_duration=20, harm_threshold=0.3, perc_threshold=0.5):
    """
    Détecte les sections harmoniques en ignorant les intros/outros avec seulement kicks ou voix parlée.

    FIX BUG : hop_length fixé à 512 (valeur par défaut de librosa) pour garantir
    une indexation cohérente des tableaux chroma et flatness.
    L'ancienne formule sr//2048 produisait des indices incorrects selon le sample rate.
    """
    hop_length = 512  # FIX : valeur fixe et cohérente avec librosa par défaut

    chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)

    harmonic_starts = []
    segments = range(0, int(duration) - step, step // 2)

    for start in segments:
        idx_start = int(start * sr) // hop_length
        idx_end   = int((start + step) * sr) // hop_length

        seg_chroma = chroma_full[:, idx_start:idx_end]
        chroma_var = np.var(seg_chroma)

        seg_flat = np.mean(flatness[0, idx_start:idx_end])

        if chroma_var > harm_threshold and seg_flat < perc_threshold:
            harmonic_starts.append(start)

    if not harmonic_starts:
        return 0, duration

    harmonic_starts = np.array(harmonic_starts)
    diffs = np.diff(harmonic_starts)
    breaks = np.where(diffs > step)[0]

    sections = np.split(harmonic_starts, breaks + 1)
    longest_section = max(sections, key=len)

    if len(longest_section) * step < min_harm_duration:
        return 0, duration

    harm_start = longest_section[0]
    harm_end = longest_section[-1] + step

    harm_start = max(harm_start, 5)
    harm_end = min(harm_end, duration - 5)

    return harm_start, harm_end

def detect_cadence_resolution(timeline, final_key):
    """
    Détection des cadences de résolution (ex. : V-I) pour valider la tonique.
    """
    note, mode = final_key.split()
    root_idx = NOTES_LIST.index(note)
    dom_idx = (root_idx + 7) % 12
    subdom_idx = (root_idx + 5) % 12

    resolution_count = 0
    for i in range(1, len(timeline)):
        prev_note = timeline[i-1]["Note"]
        curr_note = timeline[i]["Note"]

        dom_key = f"{NOTES_LIST[dom_idx]} {mode}"
        if mode == 'minor':
            if (prev_note == f"{NOTES_LIST[dom_idx]} major" or prev_note == dom_key) and curr_note == final_key:
                resolution_count += 1 if 'major' in prev_note else 0.5
        else:
            if prev_note == dom_key and curr_note == final_key:
                resolution_count += 1

        subdom_key = f"{NOTES_LIST[subdom_idx]} {mode}"
        if prev_note == subdom_key and curr_note == final_key:
            resolution_count += 0.5

    last_third = len(timeline) // 3
    last_resolutions = sum(
        1 for j in range(len(timeline) - last_third, len(timeline) - 1)
        if timeline[j]["Note"].startswith(NOTES_LIST[dom_idx]) and timeline[j+1]["Note"] == final_key
    )

    cadence_score = resolution_count + (last_resolutions * 2)
    return cadence_score

def solve_key_sniper(chroma_vector, bass_vector):
    """
    FIX PERFORMANCE : utilise PROFILES_ROLLED précomputé au lieu de recalculer
    np.roll() à chaque appel (économise ~36 appels roll par segment analysé).
    """
    best_overall_score = -1
    best_key = "Unknown"

    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)

    for mode in ["major", "minor"]:
        for i in range(12):
            profile_scores = []
            for p_name in PROFILES_ROLLED:
                rolled = PROFILES_ROLLED[p_name][mode][i]  # Précalculé — pas de roll ici
                score = np.corrcoef(cv, rolled)[0, 1]

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

                profile_scores.append(score)

            avg_score = np.mean(profile_scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = f"{NOTES_LIST[i]} {mode}"

    return {"key": best_key, "score": best_overall_score}

def get_key_score(key, chroma_vector, bass_vector):
    """
    FIX PERFORMANCE : utilise PROFILES_ROLLED précomputé.
    """
    note, mode = key.split()
    root_idx = NOTES_LIST.index(note)

    cv_norm = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv_norm = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)

    profile_scores = []
    for p_name in PROFILES_ROLLED:
        rolled = PROFILES_ROLLED[p_name][mode][root_idx]  # Précalculé — pas de roll ici
        corr = np.corrcoef(cv_norm, rolled)[0, 1]
        score = corr

        if mode == "minor":
            dom_idx = (root_idx + 7) % 12
            leading_tone = (root_idx + 11) % 12
            if cv_norm[dom_idx] > 0.45 and cv_norm[leading_tone] > 0.35:
                score *= 1.35

        if bv_norm[root_idx] > 0.6: score += (bv_norm[root_idx] * 0.2)

        fifth_idx = (root_idx + 7) % 12
        if cv_norm[fifth_idx] > 0.5: score += 0.1
        third_idx = (root_idx + 4) % 12 if mode == "major" else (root_idx + 3) % 12
        if cv_norm[third_idx] > 0.5: score += 0.1

        profile_scores.append(score)

    return np.mean(profile_scores)


def process_audio(audio_file, file_name, progress_placeholder):
    status_text = progress_placeholder.empty()
    progress_bar = progress_placeholder.progress(0)

    def update_prog(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"**{text} | {value}%**")

    # FIX ROBUSTESSE : temp_dir créé avant le try pour garantir le nettoyage en cas d'erreur
    temp_dir = tempfile.mkdtemp()

    try:
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

        update_prog(20, "Détection des sections harmoniques")
        duration = librosa.get_duration(y=y, sr=sr)
        harm_start, harm_end = detect_harmonic_sections(y, sr, duration)
        update_prog(30, f"Section harmonique détectée : {seconds_to_mmss(harm_start)} à {seconds_to_mmss(harm_end)}")

        idx_harm_start = int(harm_start * sr)
        idx_harm_end = int(harm_end * sr)
        y_harm = y[idx_harm_start:idx_harm_end]
        duration_harm = harm_end - harm_start

        update_prog(40, "Filtrage des fréquences")
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        y_filt = apply_sniper_filters(y_harm, sr)

        chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
        bass_global = get_bass_priority(y_harm, sr)

        update_prog(50, "Analyse du spectre harmonique")
        step, timeline, votes = 6, [], Counter()
        segments = range(0, int(duration_harm) - step, 2)

        for i, start in enumerate(segments):
            idx_start, idx_end = int(start * sr), int((start + step) * sr)
            seg = y_filt[idx_start:idx_end]
            if np.max(np.abs(seg)) < 0.01: continue

            c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
            c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
            b_seg = get_bass_priority(y_harm[idx_start:idx_end], sr)

            res = solve_key_sniper(c_avg, b_seg)

            weight = 3.0 if start > (duration_harm - 15) else 2.0 if start < 10 else 1.0
            votes[res['key']] += int(res['score'] * 100 * weight)
            timeline.append({"Temps": harm_start + start, "Note": res['key'], "Conf": res['score']})

            p_val = 50 + int((i / len(segments)) * 40)
            update_prog(p_val, "Calcul chirurgical en cours")

        update_prog(90, "Synthèse finale et validation de la tonique")
        most_common = votes.most_common(10)
        total_votes = sum(votes.values())

        res_global = solve_key_sniper(chroma_avg, bass_global)
        final_key = res_global['key']
        final_conf = int(res_global['score'] * 100)

        cadence_score = detect_cadence_resolution(timeline, final_key)
        if cadence_score < 2 and len(most_common) > 1:
            alt_keys = [k for k, _ in most_common if k != final_key]
            alt_cadences = {ak: detect_cadence_resolution(timeline, ak) for ak in alt_keys}
            best_alt = max(alt_cadences, key=alt_cadences.get)
            if alt_cadences[best_alt] > cadence_score + 1:
                final_key = best_alt
                final_conf = int(get_key_score(final_key, chroma_avg, bass_global) * 100)

        if timeline and timeline[-1]["Note"] == final_key:
            final_conf = min(final_conf + 5, 99)

        dominant_key = most_common[0][0] if most_common else "Unknown"
        dominant_votes = most_common[0][1] if most_common else 0
        dominant_percentage = (dominant_votes / total_votes * 100) if total_votes > 0 else 0
        dominant_conf = int(get_key_score(dominant_key, chroma_avg, bass_global) * 100) if dominant_key != "Unknown" else 0

        # Étape A : Calcul de la présence de la clé retenue par l'algorithme (Consonance)
        final_vote_count = votes.get(final_key, 0)
        final_percentage = (final_vote_count / total_votes * 100) if total_votes > 0 else 0
        dominant_camelot = CAMELOT_MAP.get(dominant_key, "??")

        mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.25
        target_key = most_common[1][0] if mod_detected else None

        target_conf = min(int(get_key_score(target_key, chroma_avg, bass_global) * 100), 99) if mod_detected else None

        modulation_time = None
        target_percentage = 0
        ends_in_target = False

        if mod_detected and target_key:
            target_times = np.array([t["Temps"] for t in timeline if t["Note"] == target_key])
            if len(target_times) > 3:
                Z = linkage(target_times.reshape(-1, 1), method='single')
                clust = fcluster(Z, t=5, criterion='distance')
                max_cluster_size = max(Counter(clust).values()) * 2
                if max_cluster_size < 10:
                    mod_detected = False
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

        update_prog(100, "Analyse terminée")
        status_text.empty()
        progress_bar.empty()

        # ══════════════════════════════════════════════════════════════════════════
        # --- MOTEUR DE DÉCISION SNIPER V12.0 (MODIFIÉ) ---
        # ══════════════════════════════════════════════════════════════════════════

        # --- CALCULS DE FORCE ET TEMPS DYNAMIQUE ---
        total_duration    = duration  # durée totale du fichier audio
        dynamic_threshold = min(total_duration * 0.20, 60)  # 20% du morceau, max 60s

        # --- AVANT (Calcul bridé) ---
        # final_power = min(final_conf, 99) * np.sqrt(max(final_percentage, 0))
        # --- APRÈS (Calcul libre pour le Power Score) ---
        # On garde la valeur brute pour le calcul de puissance (sans plafond)
        raw_final_conf    = final_conf
        raw_dominant_conf = dominant_conf
        # Power Score = Confiance brute × √Présence (sans plafond → arbitrage équitable)
        final_power = raw_final_conf    * np.sqrt(max(final_percentage, 0))
        dom_power   = raw_dominant_conf * np.sqrt(max(dominant_percentage, 0))
        power_ratio = dom_power / final_power if final_power > 0 else 0

        # --- SÉCURITÉ ANTI-ERREUR STATISTIQUE (VERSION BLINDÉE) ---
        # RÈGLE 1 : Si la clé retenue est un "fantôme" (présence < 5%) et qu'une dominante existe.
        est_fantome = (final_percentage < 5.0 and dominant_percentage > 20.0)

        # RÈGLE 2 : Si la dominante est statistiquement écrasante (Ratio de puissance > 1.5).
        domination_statistique = (power_ratio > 1.5)

        # Application de la bascule de sécurité
        if (final_conf < 0) or est_fantome or domination_statistique:
            final_key        = dominant_key
            final_conf       = dominant_conf
            raw_final_conf   = dominant_conf
            final_percentage = dominant_percentage
            # Recalcul des scores de puissance pour l'affichage
            final_power = raw_final_conf * np.sqrt(max(final_percentage, 0))
            power_ratio = dom_power / final_power if final_power > 0 else 0

        # Pré-calcul de l'arbitrage universel (Test Spectral sur quintes — v11)
        decision_pivot = None
        if final_conf >= 75 and dominant_conf >= 75:
            decision_pivot = arbitrage_expert_universel(
                chroma_avg, final_key, dominant_key, CAMELOT_MAP
            )
            # Si l'arbitrage renvoie la consonance elle-même, on neutralise
            # (pas de déclenchement de la branche ARBITRAGE HARMONIQUE)
            if decision_pivot == final_key:
                decision_pivot = None

        # 🎯 PRIORITÉ @ : VERROUILLAGE STATISTIQUE (Consonance == Dominante)
        # Si les deux moteurs sont d'accord avec une confiance et présence décente
        if final_key == dominant_key and final_conf >= 70 and final_percentage >= 30:
            confiance_pure_key = final_key
            avis_expert = "💎 VERROUILLAGE STATISTIQUE"
            color_bandeau = "linear-gradient(135deg, #10b981, #059669)"  # Vert Émeraude

        # ⚡ PRIORITÉ 0 : LA FORCE SUPRÊME (Power Score juge suprême)
        # Déclenché si la dominante écrase la consonance (ratio > 1.25)
        # OU si ce sont des voisins et que la dominante est plus solide (ratio > 1.10)
        elif power_ratio > 1.25 or (decision_pivot and power_ratio > 1.10):
            confiance_pure_key = dominant_key
            avis_expert = f"⚡ FORCE SUPRÊME ({round(dominant_percentage, 1)}%)"
            color_bandeau = "linear-gradient(135deg, #7c3aed, #4c1d95)"  # Violet Puissance

        # ⚖️ PRIORITÉ 1 : ARBITRAGE HARMONIQUE (Spectral — Power Score non décisif)
        elif decision_pivot:
            confiance_pure_key = decision_pivot
            avis_expert = "⚖️ ARBITRAGE HARMONIQUE"
            color_bandeau = "linear-gradient(135deg, #0369a1, #0c4a6e)"  # Bleu Océan

        # 🏁 PRIORITÉ 2 : MODULATION DYNAMIQUE (Proportionnelle)
        elif (mod_detected and ends_in_target and target_percentage >= 25.0
              and modulation_time is not None and modulation_time <= dynamic_threshold):
            confiance_pure_key = target_key
            avis_expert = f"🏁 MODULATION VALIDÉE ({round(modulation_time)}s / {round(total_duration)}s)"
            color_bandeau = "linear-gradient(135deg, #4338ca, #1e1b4b)"  # Violet

        # 💎 PRIORITÉ 3 : ACCORD PARFAIT (Consonance = Dominante, confiance ≥ 85%)
        elif final_key == dominant_key and final_conf >= 85:
            confiance_pure_key = final_key
            avis_expert = "💎 ACCORD PARFAIT"
            color_bandeau = "linear-gradient(135deg, #059669, #064e3b)"  # Vert Émeraude

        # ✅ FALLBACK : ANALYSE STABLE
        else:
            confiance_pure_key = final_key
            avis_expert = "✅ ANALYSE STABLE"
            color_bandeau = "linear-gradient(135deg, #065f46, #064e3b)"  # Vert Classique

        # ══════════════════════════════════════════════════════════════════════════

        res_obj = {
            "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
            "conf": min(int(raw_final_conf), 100),  # Plafond uniquement pour l'esthétique
            "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
            "chroma": chroma_avg, "modulation": mod_detected,
            "target_key": target_key, "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
            "name": file_name,
            "modulation_time_str": seconds_to_mmss(modulation_time) if mod_detected else None,
            "mod_target_percentage": round(target_percentage, 1) if mod_detected else 0,
            "mod_ends_in_target": ends_in_target if mod_detected else False,
            "harm_start": seconds_to_mmss(harm_start), "harm_end": seconds_to_mmss(harm_end),
            "target_conf": target_conf,
            "dominant_key": dominant_key, "dominant_camelot": dominant_camelot,
            "dominant_conf": dominant_conf, "dominant_percentage": round(dominant_percentage, 1),
            "key_presence": round(final_percentage, 1),
            "duration_detected": round(total_duration, 1),
            "final_power": round(final_power, 1),
            "dom_power": round(dom_power, 1),
            "power_ratio": round(power_ratio, 2),
            "confiance_pure": confiance_pure_key,
            "pure_camelot": CAMELOT_MAP.get(confiance_pure_key, "??"),
            "avis_expert": avis_expert,
            "color_bandeau": color_bandeau,
        }

        # --- RAPPORT TELEGRAM ENRICHI (RADAR + TIMELINE) ---
        if TELEGRAM_TOKEN and CHAT_ID:
            try:
                mod_line = ""
                if mod_detected:
                    perc = res_obj["mod_target_percentage"]
                    end_txt = " 🏁 *FIN SUR MODULATION*" if res_obj['mod_ends_in_target'] else ""
                    mod_line = (
                        f"\n⚠️ *MODULATION →* `{target_key.upper()} ({res_obj['target_camelot']})`"
                        f" | ≈ *{res_obj['modulation_time_str']}*"
                        f" | *PRÉSENCE:* `{perc}%`"
                        f" | *CONFIANCE:* `{res_obj['target_conf']}%`"
                        f"{end_txt}"
                    )

                dom_line = (
                    f"\n🏆 *DOMINANTE:* `{dominant_key.upper()} ({res_obj['dominant_camelot']})`"
                    f" | *PRÉSENCE:* `{res_obj['dominant_percentage']}%`"
                    f" | *CONFIANCE:* `{res_obj['dominant_conf']}%`"
                )
                pure_line = f"\n🔒 *TONALITÉ PURE:* `{res_obj['confiance_pure'].upper()} ({res_obj['pure_camelot']})` | *AVIS:* `{res_obj['avis_expert']}`"

                caption = (
                    f"🎯 *RCDJ228 MUSIC SNIPER*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📂 *FICHIER:* `{file_name}`\n"
                    f"🎹 *MEILLEURE CONSONANCE:* `{final_key.upper()} ({res_obj['camelot']})`"
                    f" | *PRÉSENCE:* `{res_obj['key_presence']}%`"
                    f" | *CONFIANCE:* `{res_obj['conf']}%`"
                    + dom_line
                    + pure_line
                    + f"{mod_line}\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🎸 *ACCORDAGE:* `{res_obj['tuning']} Hz` ✅\n"
                    f"🛡️ *SECTION HARMONIQUE:* {res_obj['harm_start']} → {res_obj['harm_end']}"
                )

                fig_radar = go.Figure(data=go.Scatterpolar(r=res_obj['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                fig_radar.update_layout(template="plotly_dark", title="SPECTRE HARMONIQUE", polar=dict(radialaxis=dict(visible=False)))
                radar_bytes = fig_radar.to_image(format="png", width=700, height=500)

                df_tl = pd.DataFrame(res_obj['timeline'])
                fig_tl = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark",
                                 category_orders={"Note": NOTES_ORDER}, title="ÉVOLUTION TEMPORELLE")
                timeline_bytes = fig_tl.to_image(format="png", width=1000, height=450)

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

        # Sauvegarde disque pour données lourdes
        timeline_path = os.path.join(temp_dir, f"{file_name}_timeline.pkl")
        chroma_path = os.path.join(temp_dir, f"{file_name}_chroma.npy")
        with open(timeline_path, 'wb') as tf:
            pickle.dump(res_obj['timeline'], tf)
        np.save(chroma_path, res_obj['chroma'])

        res_obj['timeline_path'] = timeline_path
        res_obj['chroma_path'] = chroma_path
        res_obj['temp_dir'] = temp_dir
        del res_obj['timeline']
        del res_obj['chroma']

        del y, y_filt
        gc.collect()
        return res_obj

    except Exception as e:
        # FIX ROBUSTESSE : nettoyage garanti du dossier temp même en cas de crash
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


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

global_status = st.empty()

uploaded_files = st.file_uploader("📥 Déposez vos fichiers (Audio)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True, key="file_uploader")

if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

if uploaded_files:
    global_status.info("Analyse des fichiers en cours...")
    progress_zone = st.container()

    for i, f in enumerate(reversed(uploaded_files)):
        file_name = f.name

        if file_name not in st.session_state.analyses:
            st.session_state.analyzing = True
            analysis_data = process_audio(f, file_name, progress_zone)
            st.session_state.analyses[file_name] = analysis_data
            if len(st.session_state.analyses) > 5:
                oldest_file = next(iter(st.session_state.analyses))
                if 'temp_dir' in st.session_state.analyses[oldest_file] and os.path.exists(st.session_state.analyses[oldest_file]['temp_dir']):
                    shutil.rmtree(st.session_state.analyses[oldest_file]['temp_dir'])
                del st.session_state.analyses[oldest_file]

        if file_name in st.session_state.analyses:
            analysis_data = st.session_state.analyses[file_name]

            with open(analysis_data['timeline_path'], 'rb') as tf:
                timeline = pickle.load(tf)
            chroma = np.load(analysis_data['chroma_path'])

            with st.container():
                st.markdown(f"<div class='file-header'>📂 ANALYSE : {analysis_data['name']}</div>", unsafe_allow_html=True)

                mod_alert = ""
                if analysis_data['modulation']:
                    ends_badge = " &nbsp; 🏁 <b>FIN SUR MODULATION</b>" if analysis_data['mod_ends_in_target'] else ""
                    mod_alert = (
                        f"<div class='modulation-alert'>"
                        f"⚠️ MODULATION : <b>{analysis_data['target_key'].upper()}</b> ({analysis_data['target_camelot']})"
                        f" &nbsp;|&nbsp; PRÉSENCE : <b>{analysis_data['mod_target_percentage']}%</b>"
                        f" &nbsp;|&nbsp; CONFIANCE : <b>{analysis_data['target_conf']}%</b>"
                        f"{ends_badge}"
                        f"</div>"
                    )

                st.markdown(f"""
                    <div class="report-card" style="background:{analysis_data['color_bandeau']};">
                        <p style="letter-spacing:5px; opacity:0.8; font-size:0.7em; margin-bottom:0px;">
                            SNIPER ENGINE v5.0 | {analysis_data['avis_expert']}
                        </p>
                        <h1 style="font-size:5em; margin:0px 0; font-weight:900; line-height:1; text-align: center;">
                            {analysis_data['pure_camelot']}
                        </h1>
                        <p style="font-size:2em; font-weight:bold; margin-top:-10px; margin-bottom:20px; opacity:0.9; text-align: center;">
                            {analysis_data['confiance_pure'].upper()}
                        </p>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.2); width:50%; margin: 20px auto;">
                        <p style="font-size:0.9em; opacity:0.7; font-family: 'JetBrains Mono', monospace;">
                            DÉTAILS : Consonance {analysis_data['key'].upper()} | PRÉSENCE {analysis_data.get('key_presence', analysis_data.get('dominant_percentage', 0))}% | CONFIANCE {analysis_data['conf']}%
                            &nbsp;·&nbsp; Dominante {analysis_data['dominant_key'].upper()} ({analysis_data['dominant_camelot']}) | PRÉSENCE {analysis_data['dominant_percentage']}% | CONFIANCE {analysis_data['dominant_conf']}%
                        </p>
                        {mod_alert}
                    </div>
                """, unsafe_allow_html=True)

                m2, m3 = st.columns(2)
                with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{analysis_data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                with m3:
                    btn_id = f"play_{hash(analysis_data['name'])}"
                    components.html(f"""
                        <button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER L'ACCORD</button>
                        <script>{get_chord_js(btn_id, analysis_data['key'])}</script>
                    """, height=110)

                # --- POWER SCORES (debug & transparence) ---
                ps1, ps2, ps3 = st.columns(3)
                with ps1:
                    st.markdown(f"<div class='metric-box'><b>💪 FORCE CONSONANCE</b><br><span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('final_power', '—')}</span></div>", unsafe_allow_html=True)
                with ps2:
                    st.markdown(f"<div class='metric-box'><b>💪 FORCE DOMINANTE</b><br><span style='font-size:1.6em; color:#a78bfa;'>{analysis_data.get('dom_power', '—')}</span></div>", unsafe_allow_html=True)
                with ps3:
                    ratio_val = analysis_data.get('power_ratio', 0)
                    ratio_color = "#ef4444" if ratio_val > 1.25 else "#f59e0b" if ratio_val > 1.10 else "#10b981"
                    st.markdown(f"<div class='metric-box'><b>📊 RATIO DE PUISSANCE</b><br><span style='font-size:1.6em; color:{ratio_color};'>{ratio_val}</span></div>", unsafe_allow_html=True)

                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_tl = px.line(pd.DataFrame(timeline), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                    fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_tl, use_container_width=True, key=f"timeline_{analysis_data['name']}_{i}")
                with c2:
                    fig_radar = go.Figure(data=go.Scatterpolar(r=chroma, theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                    fig_radar.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=20), polar=dict(radialaxis=dict(visible=False)), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{analysis_data['name']}_{i}")

                st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

            del timeline, chroma
            gc.collect()

    st.session_state.analyzing = False
    global_status.success("Tous les fichiers ont été analysés avec succès !")
    gc.collect()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569107.png", width=80)
    st.header("Sniper Control")
    if st.button("🧹 Vider la file d'analyse"):
        for data in list(st.session_state.analyses.values()):
            if 'temp_dir' in data and os.path.exists(data['temp_dir']):
                shutil.rmtree(data['temp_dir'])
        st.session_state.analyses = {}
        st.session_state.analyzing = False
        gc.collect()
        st.rerun()
