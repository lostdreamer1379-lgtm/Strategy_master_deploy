"""
STRATEGY MASTER - Data Collection
Pulls race data from FastF1 across configured seasons, builds per-lap records
with tyre degradation, pit, fuel, weather and circuit features.

Cross-platform: uses a thread-based timeout (works on Windows, macOS, Linux),
and falls back to sensible weather defaults if a session's weather data
can't be loaded, rather than crashing the whole collection run.
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
import os
import json
import threading

warnings.filterwarnings('ignore')

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '../data/cache')
RAW_DIR = os.path.join(HERE, '../data/raw')
PROCESSED_DIR = os.path.join(HERE, '../data/processed')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# --- Configuration ---
SEASONS = [2023, 2024, 2025, 2026]
MAX_RACES_PER_SEASON = 24
SESSION_TIMEOUT_SEC = 90
OUT_PATH = os.path.join(RAW_DIR, 'f1_strategy_data.csv')

with open(os.path.join(HERE, 'circuits_config.json'), 'r') as f:
    CIRCUITS_MAP = json.load(f)

with open(os.path.join(HERE, 'compounds_config.json'), 'r') as f:
    COMPOUND_MAP = json.load(f)

DEFAULT_CIRCUIT = {'corner_ratio': 0.55, 'abrasiveness': 7, 'speed_index': 6, 'lap_length_km': 5.0}
DEFAULT_COMPOUND = COMPOUND_MAP.get('UNKNOWN', {'durability': 5, 'grip': 5, 'optimal_temp_c': 80, 'fuel_mult': 1.0})


# --- Thread-based timeout (cross-platform; signal.SIGALRM is Unix-only) ---
class TimeoutException(Exception):
    pass


class TimedFunction:
    def __init__(self, timeout_sec=60):
        self.timeout_sec = timeout_sec
        self.result = None
        self.exception = None

    def run(self, func, *args, **kwargs):
        def target():
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive():
            raise TimeoutException(f"Operation timed out after {self.timeout_sec}s")
        if self.exception:
            raise self.exception
        return self.result


def collect_session(season, race_name):
    """Load a single race session and extract per-lap feature rows.
    Returns [] (never raises) so one bad race never kills the whole run."""
    try:
        print(f"    {race_name:<24}", end=" ", flush=True)

        try:
            timer = TimedFunction(timeout_sec=SESSION_TIMEOUT_SEC)

            def load_func():
                session = fastf1.get_session(season, race_name, 'Race')
                session.load(telemetry=False, weather=True, messages=False)
                return session

            session = timer.run(load_func)
        except TimeoutException:
            print("TIMEOUT (skipped)")
            return []
        except Exception as e:
            print(f"load error: {str(e)[:40]} (skipped)")
            return []

        laps = session.laps.copy()
        laps = laps[laps['LapTime'].notna()].reset_index(drop=True)
        if len(laps) == 0:
            print("no valid laps")
            return []

        # Weather, with safe fallback
        try:
            wx = session.weather_data
            avg_track_temp = float(wx['TrackTemp'].mean()) if 'TrackTemp' in wx.columns and len(wx) else 30.0
            avg_air_temp = float(wx['AirTemp'].mean()) if 'AirTemp' in wx.columns and len(wx) else 20.0
            avg_humidity = float(wx['Humidity'].mean()) if 'Humidity' in wx.columns and len(wx) else 50.0
            avg_wind_speed = float(wx['WindSpeed'].mean()) if 'WindSpeed' in wx.columns and len(wx) else 2.0
            rainfall = int(wx['Rainfall'].any()) if 'Rainfall' in wx.columns and len(wx) else 0
        except Exception:
            avg_track_temp, avg_air_temp, avg_humidity, avg_wind_speed, rainfall = 30.0, 20.0, 50.0, 2.0, 0

        location = session.event.get('Location', race_name)
        circuit_key = next(
            (k for k in CIRCUITS_MAP if k.lower() in str(location).lower() or str(location).lower() in k.lower()),
            None
        )
        circ = CIRCUITS_MAP.get(circuit_key, DEFAULT_CIRCUIT)

        rows = []
        for driver in laps['Driver'].unique():
            d_laps = laps[laps['Driver'] == driver].reset_index(drop=True)
            stint_start = 0
            prev_compound = None

            for idx, lap in d_laps.iterrows():
                compound = str(lap.get('Compound', 'UNKNOWN')).upper()
                comp = COMPOUND_MAP.get(compound, DEFAULT_COMPOUND)

                if compound != prev_compound:
                    stint_start = idx
                prev_compound = compound

                tyre_life = int(lap.get('TyreLife', idx - stint_start + 1))
                lap_time_s = lap['LapTime'].total_seconds()
                stint_lap = idx - stint_start + 1

                first_lap_row = d_laps.iloc[stint_start]
                first_lap_s = first_lap_row['LapTime'].total_seconds()
                time_loss_s = lap_time_s - first_lap_s
                deg_pct = (time_loss_s / first_lap_s * 100) if first_lap_s > 0 else 0.0

                pit_in = 1 if pd.notna(lap.get('PitInTime')) else 0
                pit_out = 1 if pd.notna(lap.get('PitOutTime')) else 0

                speed_fl = lap.get('SpeedFL', np.nan)
                speed_st = lap.get('SpeedST', np.nan)

                total_laps = len(d_laps)
                fuel_at_lap = max(0, 110 - (lap['LapNumber'] / total_laps * 105)) if total_laps > 0 else 55

                temp_mismatch = abs(avg_track_temp - comp['optimal_temp_c'])
                dur_x_abrasive = comp['durability'] * circ['abrasiveness']
                grip_x_corners = comp['grip'] * circ['corner_ratio']

                rows.append({
                    'season': season, 'race': race_name, 'circuit': circuit_key or race_name,
                    'driver': driver, 'lap_number': int(lap['LapNumber']), 'stint_lap': stint_lap,
                    'tyre_life': tyre_life, 'compound': compound,
                    'lap_time_s': round(lap_time_s, 4), 'first_lap_s': round(first_lap_s, 4),
                    'time_loss_s': round(time_loss_s, 4), 'deg_pct': round(deg_pct, 4),
                    'pit_in': pit_in, 'pit_out': pit_out,
                    'speed_fl': speed_fl, 'speed_st': speed_st,
                    'fuel_estimate_kg': round(fuel_at_lap, 2),
                    'track_temp_c': round(avg_track_temp, 1), 'air_temp_c': round(avg_air_temp, 1),
                    'humidity_pct': round(avg_humidity, 1), 'wind_speed_ms': round(avg_wind_speed, 2),
                    'rainfall': rainfall,
                    'corner_ratio': circ['corner_ratio'], 'abrasiveness': circ['abrasiveness'],
                    'speed_index': circ['speed_index'], 'lap_length_km': circ['lap_length_km'],
                    'comp_durability': comp['durability'], 'comp_grip': comp['grip'],
                    'comp_opt_temp': comp['optimal_temp_c'], 'temp_mismatch': round(temp_mismatch, 1),
                    'dur_x_abrasive': round(dur_x_abrasive, 2), 'grip_x_corners': round(grip_x_corners, 2),
                    'is_wet_compound': 1 if compound in ['INTERMEDIATE', 'WET'] else 0,
                })

        print(f"ok ({len(rows)} laps)")
        return rows

    except Exception as e:
        print(f"error: {str(e)[:50]}")
        return []


def load_existing_data():
    """Load previously collected rows so reruns only fetch new sessions."""
    if not os.path.exists(OUT_PATH) or os.path.getsize(OUT_PATH) == 0:
        return pd.DataFrame()

    try:
        existing = pd.read_csv(OUT_PATH)
    except Exception as e:
        raise RuntimeError(f"Could not read existing data at {OUT_PATH}: {e}") from e

    required_columns = {'season', 'race'}
    missing_columns = required_columns - set(existing.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise RuntimeError(f"Existing data is missing required columns: {missing}")

    return existing


def run_collection():
    existing_df = load_existing_data()
    collected_sessions = set(
        zip(existing_df['season'].astype(int), existing_df['race'].astype(str))
    ) if len(existing_df) else set()

    all_rows = []
    total_races, skipped_races, failed_races = 0, 0, 0

    print("\n" + "=" * 70)
    print("STRATEGY MASTER - FASTF1 DATA COLLECTION")
    print("=" * 70)
    if collected_sessions:
        print(f"   Existing sessions found: {len(collected_sessions)}")

    for season in SEASONS:
        print(f"\nSeason {season}")
        print("-" * 70)
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
            races = schedule['EventName'].tolist()[:MAX_RACES_PER_SEASON]
            print(f"   Found {len(races)} races\n")
        except Exception as e:
            print(f"   Schedule error: {e}")
            continue

        for race in races:
            if (season, race) in collected_sessions:
                print(f"    {race:<24} already collected (skipped)")
                skipped_races += 1
                continue

            result = collect_session(season, race)
            if result:
                all_rows.extend(result)
                total_races += 1
            else:
                failed_races += 1

    new_df = pd.DataFrame(all_rows)
    if len(new_df) == 0 and len(existing_df) == 0:
        print("\nNO DATA COLLECTED — check network access / FastF1 availability.")
        return

    if len(new_df):
        new_df = new_df[new_df['deg_pct'].between(-20, 20)].reset_index(drop=True)
        new_df['is_outlap'] = (new_df['stint_lap'] == 1).astype(int)

    df = pd.concat([existing_df, new_df], ignore_index=True)
    row_identity = ['season', 'race', 'driver', 'lap_number']
    df = df.drop_duplicates(subset=row_identity, keep='first').reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"   Races processed: {total_races}")
    print(f"   Races skipped:   {skipped_races}")
    print(f"   Races failed:    {failed_races}")
    print(f"   Total records:   {len(df):,}")
    print(f"   Circuits:        {df['circuit'].nunique()}")
    print(f"   Drivers:         {df['driver'].nunique()}")
    print(f"   Compounds:       {df['compound'].nunique()}")
    print(f"\nSaved to: {OUT_PATH}\n")


if __name__ == '__main__':
    run_collection()
