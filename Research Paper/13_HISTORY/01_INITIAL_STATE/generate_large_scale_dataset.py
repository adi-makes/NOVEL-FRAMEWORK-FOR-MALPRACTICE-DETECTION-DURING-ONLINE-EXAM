import os
import json
import random
import numpy as np
import pandas as pd

# ---------------- FORMAL FEATURE & TARGET WHITELIST ----------------
FEATURE_COLUMNS = [
    # Stream 1: Gaze (7 features)
    "fixation_duration_mean", "fixation_count", "saccade_velocity_mean",
    "gaze_deviation", "gaze_confidence", "head_yaw", "head_pitch",
    # Stream 2: Mouse / Interaction (7 features)
    "cursor_velocity_mean", "cursor_velocity_std", "click_frequency",
    "keystroke_frequency", "idle_fraction", "tab_switch_count", "velocity_spike_ratio",
    # Stream 3: Environment Vision (5 features)
    "phone_detected", "phone_confidence", "notes_detected",
    "extra_person_count", "suspicious_objects_count"
]

TARGET_COLUMN = "label"

METADATA_COLUMNS = [
    "participant_id", "session_id", "window_id", "timestamp_start", "timestamp_end",
    "split", "room_id", "camera_config", "hardware_config", "screen_resolution",
    "scenario_type", "cheating_behavior", "gaze_active", "mouse_active", "env_active",
    "temporal_detection_phase", "is_detectability_benchmark_sample"
]

def create_session_pool(participant_ids, num_sessions):
    """Guarantees every participant in the split receives exactly 2 or 3 sessions."""
    pids = sorted(list(participant_ids))
    random.shuffle(pids)

    base_sessions = num_sessions // len(pids)
    extra_sessions = num_sessions % len(pids)

    pool = []
    for i, pid in enumerate(pids):
        n_sessions = base_sessions + (1 if i < extra_sessions else 0)
        pool.extend([pid] * n_sessions)

    random.shuffle(pool)
    assert len(pool) == num_sessions, f"Expected {num_sessions} pooled sessions, got {len(pool)}"
    assert pd.Series(pool).value_counts().between(2, 3).all(), "Participant session count violation"
    return pool

def generate_target_dataset(output_dir="P2/data/large_scale", seed=42):
    random.seed(seed)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_path = os.path.join(base_dir, output_dir)
    os.makedirs(target_path, exist_ok=True)

    num_participants = 125
    num_sessions = 350

    # 1. Physical Environment & Hardware Profiles
    rooms = [
        {"id": "room_00", "name": "bedroom_dim", "light_noise": -0.06, "bg_noise": 0.02, "fp_prob": 0.03},
        {"id": "room_01", "name": "dorm_cluttered", "light_noise": 0.01, "bg_noise": 0.06, "fp_prob": 0.07},
        {"id": "room_02", "name": "home_office_clean", "light_noise": 0.02, "bg_noise": 0.01, "fp_prob": 0.01},
        {"id": "room_03", "name": "living_room_busy", "light_noise": 0.03, "bg_noise": 0.09, "fp_prob": 0.08},
        {"id": "room_04", "name": "library_cubicle", "light_noise": 0.00, "bg_noise": 0.05, "fp_prob": 0.04},
        {"id": "room_05", "name": "kitchen_glare", "light_noise": 0.07, "bg_noise": 0.05, "fp_prob": 0.04},
        {"id": "room_06", "name": "shared_apartment", "light_noise": 0.02, "bg_noise": 0.08, "fp_prob": 0.06},
        {"id": "room_07", "name": "basement_low_light", "light_noise": -0.09, "bg_noise": 0.01, "fp_prob": 0.02},
        {"id": "room_08", "name": "sunroom_bright", "light_noise": 0.08, "bg_noise": 0.03, "fp_prob": 0.03},
        {"id": "room_09", "name": "studio_motion", "light_noise": 0.02, "bg_noise": 0.10, "fp_prob": 0.08}
    ]

    camera_configs = [
        {"id": "cam_webcam_centered", "yaw_bias": 0.0, "pitch_bias": 0.0, "conf_bias": 0.0},
        {"id": "cam_laptop_integrated", "yaw_bias": 0.0, "pitch_bias": -4.0, "conf_bias": -0.03},
        {"id": "cam_webcam_left_offset", "yaw_bias": 7.5, "pitch_bias": 1.5, "conf_bias": -0.02},
        {"id": "cam_external_usb_wide", "yaw_bias": -6.5, "pitch_bias": 2.5, "conf_bias": -0.02}
    ]

    hardware_configs = [
        {"id": "hw_laptop_trackpad", "vel_mult": 0.88, "click_mult": 0.92, "jitter": 3.0, "res": "1920x1080"},
        {"id": "hw_desktop_optical_mouse", "vel_mult": 1.0, "click_mult": 1.0, "jitter": 1.0, "res": "1920x1080"},
        {"id": "hw_gaming_high_dpi", "vel_mult": 1.25, "click_mult": 1.12, "jitter": 5.0, "res": "2560x1440"},
        {"id": "hw_budget_wireless_mouse", "vel_mult": 0.94, "click_mult": 0.96, "jitter": 7.0, "res": "1366x768"},
        {"id": "hw_macbook_trackpad", "vel_mult": 0.90, "click_mult": 1.02, "jitter": 2.0, "res": "2560x1600"},
        {"id": "hw_ultrawide_dual_display", "vel_mult": 1.15, "click_mult": 1.0, "jitter": 2.5, "res": "3440x1440"}
    ]

    participant_traits = {
        f"participant_{p:03d}": {
            "gaze_bias_yaw": float(np.random.normal(0.0, 2.0)),
            "gaze_bias_pitch": float(np.random.normal(0.0, 1.5)),
            "motor_speed_bias": float(np.random.normal(1.0, 0.08)),
            "baseline_typing_rate": float(np.random.normal(1.1, 0.18))
        } for p in range(num_participants)
    }

    # Participant split: 75 Train, 25 Val, 25 Test (Zero Leakage)
    all_pids = sorted(list(participant_traits.keys()))
    random.shuffle(all_pids)
    train_pids = sorted(all_pids[:75])
    val_pids = sorted(all_pids[75:100])
    test_pids = sorted(all_pids[100:125])

    # 2. Stratified Scenario Distribution Mapping (Exact Integer Balancing)
    # Total = 175 Honest + 150 Observable Cheating + 25 Silent Cheating = 350
    scenario_split_plan = {
        # Honest Scenarios (Total 175: 105 Train, 35 Val, 35 Test)
        "honest_baseline": {"train": 45, "val": 15, "test": 15, "beh": "none", "g": "normal", "m": "normal", "e": "clean", "lbl": 0},
        "benign_gaze_deviation": {"train": 21, "val": 7, "test": 7, "beh": "reading_scratchpad", "g": "benign_drift", "m": "normal", "e": "clean", "lbl": 0},
        "benign_mouse_anomaly": {"train": 21, "val": 7, "test": 7, "beh": "accessibility_rapid_scroll", "g": "normal", "m": "benign_scroll", "e": "clean", "lbl": 0},
        "benign_environment_object": {"train": 18, "val": 6, "test": 6, "beh": "harmless_calculator_or_passerby", "g": "normal", "m": "normal", "e": "benign_env_noise", "lbl": 0},

        # Observable Cheating Scenarios (Total 150: 105 Train, 35 Val, 10 Test)
        "single_modality_gaze": {"train": 10, "val": 4, "test": 1, "beh": "offscreen_glance", "g": "cheating_gaze", "m": "normal", "e": "clean", "lbl": 1},
        "single_modality_mouse": {"train": 10, "val": 4, "test": 1, "beh": "clipboard_copy_paste", "g": "normal", "m": "cheating_mouse", "e": "clean", "lbl": 1},
        "single_modality_env": {"train": 10, "val": 4, "test": 1, "beh": "phone_lookup", "g": "normal", "m": "normal", "e": "phone_detected", "lbl": 1},
        "two_modality_gaze_mouse": {"train": 18, "val": 5, "test": 2, "beh": "web_search_rapid_switching", "g": "cheating_gaze", "m": "cheating_mouse", "e": "clean", "lbl": 1},
        "two_modality_gaze_env": {"train": 18, "val": 5, "test": 2, "beh": "paper_notes", "g": "cheating_gaze", "m": "normal", "e": "notes_detected", "lbl": 1},
        "two_modality_mouse_env": {"train": 18, "val": 5, "test": 2, "beh": "external_reference_lookup", "g": "normal", "m": "cheating_mouse", "e": "notes_detected", "lbl": 1},
        "three_modality_fusion": {"train": 21, "val": 8, "test": 1, "beh": "phone_lookup_and_copy", "g": "cheating_gaze", "m": "cheating_mouse", "e": "phone_detected", "lbl": 1},

        # Silent Cheating (Total 25: 0 Train, 0 Val, 25 Test)
        "silent_cheating": {"train": 0, "val": 0, "test": 25, "beh": "unobservable_latent_assistance", "g": "normal", "m": "normal", "e": "clean", "lbl": 1}
    }

    train_scenarios, val_scenarios, test_scenarios = [], [], []
    for scen_name, spec in scenario_split_plan.items():
        base_item = {
            "scenario": scen_name,
            "beh": spec["beh"],
            "g": spec["g"],
            "m": spec["m"],
            "e": spec["e"],
            "lbl": spec["lbl"]
        }
        train_scenarios.extend([dict(base_item) for _ in range(spec["train"])])
        val_scenarios.extend([dict(base_item) for _ in range(spec["val"])])
        test_scenarios.extend([dict(base_item) for _ in range(spec["test"])])

    # Pre-generation split & scenario assertions
    assert len(train_scenarios) == 210, f"Expected 210 train scenarios, got {len(train_scenarios)}"
    assert len(val_scenarios) == 70, f"Expected 70 val scenarios, got {len(val_scenarios)}"
    assert len(test_scenarios) == 70, f"Expected 70 test scenarios, got {len(test_scenarios)}"

    train_pool = create_session_pool(train_pids, 210)
    val_pool = create_session_pool(val_pids, 70)
    test_pool = create_session_pool(test_pids, 70)

    random.shuffle(train_scenarios)
    random.shuffle(val_scenarios)
    random.shuffle(test_scenarios)

    session_plan = []
    for p, s in zip(train_pool, train_scenarios):
        session_plan.append((p, s, "train"))
    for p, s in zip(val_pool, val_scenarios):
        session_plan.append((p, s, "val"))
    for p, s in zip(test_pool, test_scenarios):
        session_plan.append((p, s, "test"))

    random.shuffle(session_plan)

    all_records = []
    session_metadata_records = []
    print(f"Synthesizing {num_sessions} sessions across {num_participants} participants...")

    for sess_idx, (p_id, s_meta, split) in enumerate(session_plan):
        s_id = f"sess_large_{sess_idx:03d}"
        traits = participant_traits[p_id]

        rm = random.choice(rooms)
        cam = random.choice(camera_configs)
        hw = random.choice(hardware_configs)

        duration_windows = int(random.randint(180, 360))  # 30-60 min
        session_label = s_meta["lbl"]

        if session_label == 1:
            cheat_start = int(random.randint(25, max(30, duration_windows // 3)))
            cheat_duration = int(random.randint(40, 100))
            cheat_end = min(duration_windows - 15, cheat_start + cheat_duration)
            
            # Cross-modal temporal onset offsets (asynchronous human actions)
            gaze_onset = cheat_start + random.randint(0, 2)
            mouse_onset = cheat_start + random.randint(1, 3)
            env_onset = cheat_start + random.randint(1, 4)
        else:
            cheat_start, cheat_end = -1, -1
            gaze_onset, mouse_onset, env_onset = -1, -1, -1

        session_metadata_records.append({
            "session_id": s_id,
            "participant_id": p_id,
            "split": split,
            "scenario": s_meta["scenario"],
            "behavior": s_meta["beh"],
            "session_label": session_label,
            "room_id": rm["id"],
            "camera_config": cam["id"],
            "hardware_config": hw["id"]
        })

        ar_gaze_dev = float(np.random.normal(0.12, 0.04))
        ar_cursor_vel = float(np.random.normal(125.0, 20.0))
        ar_yaw = traits["gaze_bias_yaw"]
        ar_pitch = traits["gaze_bias_pitch"]

        for w in range(duration_windows):
            is_active_cheat = session_label == 1 and (cheat_start <= w <= cheat_end)
            current_window_label = 1 if is_active_cheat else 0
            current_behavior = s_meta["beh"] if (is_active_cheat or session_label == 0) else "none"

            gaze_act = (session_label == 1 and gaze_onset <= w <= cheat_end) or session_label == 0
            mouse_act = (session_label == 1 and mouse_onset <= w <= cheat_end) or session_label == 0
            env_act = (session_label == 1 and env_onset <= w <= cheat_end) or session_label == 0

            gaze_state = s_meta["g"] if gaze_act else "normal"
            mouse_state = s_meta["m"] if mouse_act else "normal"
            env_state = s_meta["e"] if env_act else "clean"

            if not is_active_cheat:
                detection_phase = "baseline"
            else:
                active_cues = sum([gaze_state != "normal", mouse_state != "normal", env_state != "clean"])
                expected_cues = sum([s_meta["g"] != "normal", s_meta["m"] != "normal", s_meta["e"] != "clean"])
                if active_cues == 0:
                    detection_phase = "onset_latency"
                elif active_cues < expected_cues:
                    detection_phase = "partially_observable"
                else:
                    detection_phase = "fully_observable"

            # ---------------- 1. GAZE STREAM (CONTINUOUS STATISTICAL OVERLAP) ----------------
            alpha = 0.65
            if gaze_state == "cheating_gaze":
                target_dev = float(np.random.normal(0.40, 0.09))
                target_yaw = float(np.random.normal(19.0, 5.5))
                target_pitch = float(np.random.normal(-11.0, 4.5))
                fix_dur = float(np.clip(np.random.normal(2.3, 0.5), 0.5, 5.5))
                sacc_vel = float(np.clip(np.random.normal(240, 45), 80, 500))
            elif gaze_state == "benign_drift":
                target_dev = float(np.random.normal(0.34, 0.08))
                target_yaw = float(np.random.normal(15.0, 4.5))
                target_pitch = float(np.random.normal(10.5, 4.0))
                fix_dur = float(np.clip(np.random.normal(2.1, 0.4), 0.5, 5.0))
                sacc_vel = float(np.clip(np.random.normal(215, 40), 70, 450))
            else:
                target_dev = float(np.random.normal(0.14, 0.05))
                target_yaw = traits["gaze_bias_yaw"]
                target_pitch = traits["gaze_bias_pitch"]
                fix_dur = float(np.clip(np.random.normal(1.8, 0.35), 0.5, 4.0))
                sacc_vel = float(np.clip(np.random.normal(180, 30), 60, 350))

            ar_gaze_dev = float(np.clip(alpha * ar_gaze_dev + (1 - alpha) * target_dev + np.random.normal(0, 0.035), 0.0, 1.0))
            ar_yaw = float(np.clip(alpha * ar_yaw + (1 - alpha) * target_yaw + np.random.normal(0, 2.0), -60.0, 60.0))
            ar_pitch = float(np.clip(alpha * ar_pitch + (1 - alpha) * target_pitch + np.random.normal(0, 1.8), -50.0, 50.0))

            gaze_conf = float(np.clip(0.92 - (0.14 * ar_gaze_dev) + cam["conf_bias"] + rm["light_noise"], 0.25, 1.0))
            sacc_vel = float(sacc_vel * (1.0 + rm["bg_noise"]))

            gaze_feats = {
                "fixation_duration_mean": fix_dur,
                "fixation_count": int(np.clip(np.random.poisson(5 if gaze_state != "normal" else 6), 1, 12)),
                "saccade_velocity_mean": sacc_vel,
                "gaze_deviation": ar_gaze_dev,
                "gaze_confidence": gaze_conf,
                "head_yaw": float(np.clip(ar_yaw + cam["yaw_bias"], -60.0, 60.0)),
                "head_pitch": float(np.clip(ar_pitch + cam["pitch_bias"], -60.0, 60.0))
            }

            # ---------------- 2. INTERACTION STREAM (MOTOR OVERLAP) ----------------
            if mouse_state == "cheating_mouse":
                target_vel = float(np.random.normal(165.0, 40.0)) * traits["motor_speed_bias"]
                click_freq = float(np.clip(np.random.normal(0.90, 0.28) * hw["click_mult"], 0.0, 3.5))
                keys_freq = float(np.clip(np.random.normal(2.0, 0.6) * traits["baseline_typing_rate"], 0.0, 7.0))
                tab_switches = int(np.random.choice([0, 1, 2], p=[0.25, 0.55, 0.20]))
                v_spike = float(np.clip(np.random.normal(0.16, 0.05), 0.01, 0.6))
                idle_f = float(np.clip(np.random.normal(0.12, 0.05), 0.0, 0.50))
            elif mouse_state == "benign_scroll":
                target_vel = float(np.random.normal(155.0, 35.0)) * traits["motor_speed_bias"]
                click_freq = float(np.clip(np.random.normal(1.10, 0.28) * hw["click_mult"], 0.0, 3.5))
                keys_freq = float(np.clip(np.random.normal(1.3, 0.40) * traits["baseline_typing_rate"], 0.0, 5.0))
                tab_switches = int(np.random.choice([0, 1], p=[0.90, 0.10]))
                v_spike = float(np.clip(np.random.normal(0.14, 0.04), 0.01, 0.5))
                idle_f = float(np.clip(np.random.normal(0.10, 0.04), 0.0, 0.45))
            else:
                target_vel = float(np.random.normal(130.0, 30.0)) * traits["motor_speed_bias"]
                click_freq = float(np.clip(np.random.normal(0.50, 0.15) * hw["click_mult"], 0.0, 2.0))
                keys_freq = float(np.clip(np.random.normal(1.1, 0.30) * traits["baseline_typing_rate"], 0.0, 3.5))
                tab_switches = int(np.random.choice([0, 1], p=[0.98, 0.02]))
                v_spike = float(np.clip(np.random.normal(0.05, 0.02), 0.0, 0.25))
                idle_f = float(np.clip(np.random.normal(0.16, 0.06), 0.0, 0.65))

            ar_cursor_vel = float(np.clip(alpha * ar_cursor_vel + (1 - alpha) * target_vel + np.random.normal(0, 14.0), 0.0, 750.0))
            final_cursor_vel = float(np.clip((ar_cursor_vel * hw["vel_mult"]) + hw["jitter"], 0.0, 850.0))

            mouse_feats = {
                "cursor_velocity_mean": final_cursor_vel,
                "cursor_velocity_std": float(np.clip(final_cursor_vel * 0.30, 5.0, 220.0)),
                "click_frequency": click_freq,
                "keystroke_frequency": keys_freq,
                "idle_fraction": idle_f,
                "tab_switch_count": tab_switches,
                "velocity_spike_ratio": v_spike
            }

            # ---------------- 3. ENVIRONMENT STREAM (HARD NEGATIVES & CONFIDENCE OVERLAP) ----------------
            if env_state == "phone_detected":
                phone_det, notes_det = 1, 0
                phone_c = float(np.clip(np.random.normal(0.68, 0.12), 0.30, 0.95))
                extra_p = 0
                susp_objs = int(np.clip(np.random.poisson(2), 1, 3))
            elif env_state == "notes_detected":
                phone_det, notes_det = 0, 1
                phone_c = float(np.clip(np.random.normal(0.09, 0.05), 0.0, 0.25))
                extra_p = 0
                susp_objs = int(np.clip(np.random.poisson(1), 1, 3))
            elif env_state == "benign_env_noise":
                phone_det = 0
                notes_det = int(np.random.choice([0, 1], p=[0.65, 0.35]))
                phone_c = float(np.clip(np.random.normal(0.36, 0.09), 0.10, 0.58))
                extra_p = int(np.random.choice([0, 1], p=[0.85, 0.15]))
                susp_objs = int(np.random.choice([1, 2], p=[0.8, 0.2]))
            else:
                phone_det, notes_det, phone_c, extra_p, susp_objs = 0, 0, float(np.clip(np.random.normal(0.04, 0.02), 0.0, 0.14)), 0, 0

            if random.random() < rm["fp_prob"] and susp_objs == 0:
                susp_objs = 1
                phone_c = float(np.clip(phone_c + 0.14, 0.0, 0.50))

            env_feats = {
                "phone_detected": phone_det,
                "phone_confidence": phone_c,
                "notes_detected": notes_det,
                "extra_person_count": extra_p,
                "suspicious_objects_count": susp_objs
            }

            row = {
                "participant_id": p_id,
                "session_id": s_id,
                "window_id": f"{s_id}_w{w:04d}",
                "timestamp_start": w * 10,
                "timestamp_end": (w + 1) * 10,
                "split": split,
                # Context / Robustness Metadata (Whitelisted out of training features)
                "room_id": rm["id"],
                "camera_config": cam["id"],
                "hardware_config": hw["id"],
                "screen_resolution": hw["res"],
                "scenario_type": s_meta["scenario"] if is_active_cheat else "session_baseline",
                "cheating_behavior": current_behavior,
                "gaze_active": bool(gaze_act and gaze_state != "normal"),
                "mouse_active": bool(mouse_act and mouse_state != "normal"),
                "env_active": bool(env_act and env_state != "clean"),
                "temporal_detection_phase": detection_phase,
                "is_detectability_benchmark_sample": bool(s_meta["scenario"] == "silent_cheating" and is_active_cheat),
                TARGET_COLUMN: current_window_label
            }
            row.update(gaze_feats)
            row.update(mouse_feats)
            row.update(env_feats)
            all_records.append(row)

    df = pd.DataFrame(all_records)
    session_df = pd.DataFrame(session_metadata_records)

    # ---------------- 4. POST-GENERATION INTEGRITY AUDIT ----------------
    print("Executing post-generation integrity audit...")
    assert df["participant_id"].nunique() == num_participants, f"Expected {num_participants} participants"
    assert df["session_id"].nunique() == num_sessions, f"Expected {num_sessions} sessions"
    assert df.groupby("session_id").size().between(180, 360).all(), "Session window counts out of bounds"
    assert set(df["split"].unique()) == {"train", "val", "test"}, "Splits missing"
    assert df.isnull().sum().sum() == 0, "Found NaNs in dataset"

    # Exact session split assertions (210 / 70 / 70)
    split_counts = session_df["split"].value_counts()
    assert split_counts["train"] == 210, f"Expected 210 train sessions, got {split_counts.get('train', 0)}"
    assert split_counts["val"] == 70, f"Expected 70 val sessions, got {split_counts.get('val', 0)}"
    assert split_counts["test"] == 70, f"Expected 70 test sessions, got {split_counts.get('test', 0)}"

    # Verify participant session counts (strictly 2 to 3 sessions per participant across entire dataset)
    p_counts = session_df["participant_id"].value_counts()
    assert p_counts.between(2, 3).all(), "Participant session count violation (must be 2-3)"

    # Verify silent cheating is strictly isolated to the test split
    silent_splits = set(session_df.loc[session_df["scenario"] == "silent_cheating", "split"])
    assert silent_splits == {"test"}, f"Silent cheating leaked into non-test splits: {silent_splits}"

    # Verify zero participant leakage across splits
    train_p = set(df.loc[df["split"] == "train", "participant_id"])
    val_p = set(df.loc[df["split"] == "val", "participant_id"])
    test_p = set(df.loc[df["split"] == "test", "participant_id"])
    assert train_p.isdisjoint(val_p), "Participant leakage between train and val"
    assert train_p.isdisjoint(test_p), "Participant leakage between train and test"
    assert val_p.isdisjoint(test_p), "Participant leakage between val and test"
    print("[PASS] Stratification (210/70/70), zero leakage, and silent-cheating isolation verified.")

    # ---------------- 5. EXPORT ARTIFACTS & AUDITS ----------------
    csv_file = os.path.join(target_path, "large_scale_dataset.csv")
    df.to_csv(csv_file, index=False)

    distribution_audit = {
        "room_by_split": pd.crosstab(session_df["room_id"], session_df["split"]).to_dict(),
        "camera_by_split": pd.crosstab(session_df["camera_config"], session_df["split"]).to_dict(),
        "hardware_by_split": pd.crosstab(session_df["hardware_config"], session_df["split"]).to_dict(),
        "scenario_by_split": pd.crosstab(session_df["scenario"], session_df["split"]).to_dict()
    }
    with open(os.path.join(target_path, "distribution_audit.json"), "w") as f:
        json.dump(distribution_audit, f, indent=2)

    metadata = {
        "dataset_version": "3.1_stratified_temporal_benchmark",
        "description": "Synthetic multimodal temporal benchmark for online exam malpractice detection research.",
        "scientific_disclaimer": "This benchmark is generated synthetically to emulate realistic behavioral variance, benign anomalies, environmental noise, and cross-modal temporal asynchrony. It is designed for algorithmic benchmarking and not as empirical human data.",
        "feature_whitelist": {
            "model_training_features": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "metadata_columns_to_exclude_from_training": METADATA_COLUMNS
        },
        "total_participants": num_participants,
        "total_sessions": num_sessions,
        "session_level_balance": {
            "honest_sessions": 175,
            "cheating_sessions": 175,
            "honest_percentage": 50.0,
            "cheating_percentage": 50.0
        },
        "window_level_balance": {
            "total_windows": len(df),
            "honest_windows": int((df[TARGET_COLUMN] == 0).sum()),
            "cheating_windows": int((df[TARGET_COLUMN] == 1).sum()),
            "honest_percentage": float(((df[TARGET_COLUMN] == 0).sum() / len(df)) * 100),
            "cheating_percentage": float(((df[TARGET_COLUMN] == 1).sum() / len(df)) * 100),
            "note": "Window-level distribution is naturally imbalanced because active malpractice occupies a continuous temporal sub-interval (40-100 windows) of each 30-60 minute exam session."
        },
        "splits_by_participant": {
            "train_participants": len(train_p),
            "val_participants": len(val_p),
            "test_participants": len(test_p)
        },
        "splits_by_session": {
            "train_sessions": int(split_counts["train"]),
            "val_sessions": int(split_counts["val"]),
            "test_sessions": int(split_counts["test"])
        },
        "scenario_distribution": df["scenario_type"].value_counts().to_dict(),
        "temporal_detection_phases": df["temporal_detection_phase"].value_counts().to_dict(),
        "detectability_analysis": {
            "silent_cheating_windows": int((df["is_detectability_benchmark_sample"] == True).sum()),
            "split_allocation": "test_only",
            "guideline": "Evaluate 'silent_cheating' separately to characterize the fundamental observability limit of the proposed multimodal system."
        }
    }
    with open(os.path.join(target_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Stratified Dataset Generation Completed Successfully ===")
    print(f"File: {csv_file}")
    print(f"Total Windows: {len(df):,}")
    print(f"Session Balance: 175 Honest / 175 Cheating (50% / 50% across {num_participants} Participants)")
    print(f"Window Counts: Honest={metadata['window_level_balance']['honest_windows']:,} ({metadata['window_level_balance']['honest_percentage']:.1f}%), Cheating={metadata['window_level_balance']['cheating_windows']:,} ({metadata['window_level_balance']['cheating_percentage']:.1f}%)")
    print(f"Splits: Train={len(train_p)} PIDs (210 sessions), Val={len(val_p)} PIDs (70 sessions), Test={len(test_p)} PIDs (70 sessions)")
    print("Audits written to 'distribution_audit.json' and 'metadata.json'.")

if __name__ == "__main__":
    generate_target_dataset()