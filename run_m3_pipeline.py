import json
import os
from datetime import datetime, timedelta
import copernicusmarine
from shapely.geometry import shape

from modules.drift.pipeline import DriftPipeline

METADATA_FILEPATH = os.path.join("outputs", "geospatial", "spill_metadata.json")
OUTPUT_RESULT_FILEPATH = os.path.join("outputs", "drift", "drift_simulation_result.json")


MAX_SLICKS_TO_SIMULATE = 10 
MIN_AREA_THRESHOLD = 0.0001 


def filter_and_rank_slicks(slicks, max_count=10):
    valid_slicks = []
    
    for s in slicks:
        try:
            geom = shape(s.get("geometry", {}))
            area = geom.area
            if area >= MIN_AREA_THRESHOLD:
                valid_slicks.append((area, s))
        except Exception:
            continue

   
    valid_slicks.sort(key=lambda x: x[0], reverse=True)
    selected = [item[1] for item in valid_slicks[:max_count]]
    
    print(f"[FILTER] Total slicks: {len(slicks)} -> Filtered down to top {len(selected)} largest features.")
    return selected


def _compute_dynamic_bbox(slicks, buffer=0.5):
    all_lons, all_lats = [], []
    for slick in slicks:
        coords = slick.get("geometry", {}).get("coordinates", [[]])[0]
        all_lons.extend([c[0] for c in coords])
        all_lats.extend([c[1] for c in coords])

    return dict(
        minimum_longitude=min(all_lons) - buffer,
        maximum_longitude=max(all_lons) + buffer,
        minimum_latitude=min(all_lats) - buffer,
        maximum_latitude=max(all_lats) + buffer,
    )


def fetch_cmems_data(obs_time_str, bbox, hindcast_hours=12, forecast_hours=11):
    obs_time = datetime.strptime(obs_time_str, "%Y-%m-%dT%H:%M:%SZ")
    
    start_dt = obs_time - timedelta(hours=hindcast_hours, minutes=60)
    end_dt = obs_time + timedelta(hours=forecast_hours, minutes=60)

    start_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    currents_file = "cmems_currents.nc"
    winds_file = "cmems_winds.nc"

    print(f"[CMEMS] Downloading ocean/wind data: {start_time} to {end_time}...")

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
        variables=["uo", "vo"],
        start_datetime=start_time,
        end_datetime=end_time,
        output_filename=currents_file,
        overwrite=True,
        **bbox
    )

    copernicusmarine.subset(
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        variables=["eastward_wind", "northward_wind"],
        start_datetime=start_time,
        end_datetime=end_time,
        output_filename=winds_file,
        overwrite=True,
        **bbox
    )

    return currents_file, winds_file


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_RESULT_FILEPATH), exist_ok=True)

    if not os.path.exists(METADATA_FILEPATH):
        raise FileNotFoundError(f"Could not find spill metadata at {METADATA_FILEPATH}")

    with open(METADATA_FILEPATH, "r") as f:
        spill_data = json.load(f)

    all_slicks = spill_data if isinstance(spill_data, list) else [spill_data]
    
    # Select only the top priority slicks
    target_slicks = filter_and_rank_slicks(all_slicks, max_count=MAX_SLICKS_TO_SIMULATE)
    
    if not target_slicks:
        print("[WARNING] No valid slicks met the filter criteria. Exiting.")
        exit(0)

    dynamic_bbox = _compute_dynamic_bbox(target_slicks)
    
    observation_time = target_slicks[0].get("properties", {}).get("observation_time", "2026-09-04T11:00:00Z")
    
    HINDCAST_HOURS = 12
    FORECAST_HOURS = 11

    currents_nc, winds_nc = fetch_cmems_data(
        obs_time_str=observation_time,
        bbox=dynamic_bbox,
        hindcast_hours=HINDCAST_HOURS,
        forecast_hours=FORECAST_HOURS
    )

    pipeline = DriftPipeline(metocean_sources=[currents_nc, winds_nc])
    all_results = []

    print(f"\nProcessing {len(target_slicks)} selected slicks sequentially...")

    for idx, slick in enumerate(target_slicks, start=1):
        try:
            m3_result = pipeline.run_simulation(
                geojson=slick,
                observation_time_iso=observation_time,
                forecast_hours=FORECAST_HOURS,
                hindcast_hours=HINDCAST_HOURS
            )

            m4_params = {
                "slick_id": m3_result.get("slick_id"),
                "bounding_box": m3_result["origin_estimation"]["bounding_box"],
                "start_time": m3_result["origin_estimation"]["estimated_spill_time_utc"],
                "end_time": observation_time
            }

            all_results.append({
                "simulation": m3_result,
                "m4_ais_query": m4_params
            })
            
            print(f"[{idx}/{len(target_slicks)}] Done: {m3_result.get('slick_id')}")

        except KeyboardInterrupt:
            print("\n[CANCELLED] Stopped by user (Ctrl+C). Saving partial results...")
            break
        except Exception as e:
            print(f"[{idx}/{len(target_slicks)}] Skipped due to error: {str(e)}")

    with open(OUTPUT_RESULT_FILEPATH, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n[SUCCESS] Saved {len(all_results)} simulations to: {OUTPUT_RESULT_FILEPATH}")