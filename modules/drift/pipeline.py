import logging
from datetime import timedelta
import numpy as np
import pandas as pd
from opendrift.models.openoil import OpenOil
from .metocean import attach_metocean_readers
from .utils import extract_polygon_coordinates, compute_bounding_box

logger = logging.getLogger(__name__)


def _nan_to_none(nested_list):
    if isinstance(nested_list, list):
        return [_nan_to_none(item) for item in nested_list]
    elif isinstance(nested_list, (float, np.floating)):
        return None if np.isnan(nested_list) else float(nested_list)
    return nested_list


class DriftPipeline:
    def __init__(self, metocean_sources=None):
        self.metocean_sources = metocean_sources or []

    def _init_model_instance(self, disable_weathering=False):
        o = OpenOil(loglevel=20)

        if disable_weathering:
            o.set_config('processes:dispersion', False)
            o.set_config('processes:evaporation', False)
            o.set_config('processes:emulsification', False)
            o.set_config('processes:biodegradation', False)
            
        attach_metocean_readers(o, self.metocean_sources)
        return o

    def _extract_summary_trajectory(self, result_xr):
        times = [pd.to_datetime(t).strftime('%Y-%m-%dT%H:%M:%SZ') for t in result_xr.time.values]
        
        
        mean_lons = np.nanmean(result_xr.lon.values, axis=0).tolist()
        mean_lats = np.nanmean(result_xr.lat.values, axis=0).tolist()
 
        min_lons = np.nanmin(result_xr.lon.values, axis=0).tolist()
        max_lons = np.nanmax(result_xr.lon.values, axis=0).tolist()
        min_lats = np.nanmin(result_xr.lat.values, axis=0).tolist()
        max_lats = np.nanmax(result_xr.lat.values, axis=0).tolist()

        return {
            "times": times,
            "centroid_path": {
                "lons": _nan_to_none(mean_lons),
                "lats": _nan_to_none(mean_lats)
            },
            "bounding_envelope": {
                "min_lons": _nan_to_none(min_lons),
                "max_lons": _nan_to_none(max_lons),
                "min_lats": _nan_to_none(min_lats),
                "max_lats": _nan_to_none(max_lats)
            }
        }

    def run_simulation(self, geojson, observation_time_iso, forecast_hours=12, hindcast_hours=12):
        obs_time = pd.to_datetime(observation_time_iso, utc=True).tz_localize(None).to_pydatetime()
        lons, lats = extract_polygon_coordinates(geojson)

        # Backward Hindcast 
        o_hind = self._init_model_instance(disable_weathering=True)
        o_hind.seed_within_polygon(
            lons=lons,
            lats=lats,
            time=obs_time,
            number=100,
            oil_type='GENERIC MEDIUM CRUDE'
        )

        o_hind.run(duration=timedelta(hours=hindcast_hours), time_step=-900)
        
        # Sort hindcast results sequentially (earliest -> latest observation)
        hindcast_times = pd.to_datetime(o_hind.result.time.values)
        order = np.argsort(hindcast_times.values)
        ordered_result = o_hind.result.isel(time=order)
        
        # Origin state corresponds to the earliest time step (index 0 of sorted result)
        origin_lons_at_t0 = ordered_result.lon.isel(time=0).values
        origin_lats_at_t0 = ordered_result.lat.isel(time=0).values
        
        # Filter out NaN values resulting from deactivated or unseeded slots
        valid_mask = ~np.isnan(origin_lons_at_t0) & ~np.isnan(origin_lats_at_t0)
        valid_origin_lons = origin_lons_at_t0[valid_mask]
        valid_origin_lats = origin_lats_at_t0[valid_mask]

        origin_bbox = compute_bounding_box(valid_origin_lons, valid_origin_lats)
        estimated_spill_time = obs_time - timedelta(hours=hindcast_hours)

        # Extract compact centroid summary trajectory
        hindcast_history = self._extract_summary_trajectory(ordered_result)

        # Check for stranded elements warning
        n_stranded = len(o_hind.elements_deactivated.lon) if hasattr(o_hind, 'elements_deactivated') else 0
        n_total = n_stranded + len(o_hind.elements.lon)
        if n_total > 0 and (n_stranded / n_total) > 0.3:
            logger.warning(
                f"{n_stranded}/{n_total} elements deactivated during hindcast — "
                "origin estimate may be less reliable than usual."
            )

        # Forward Drift Forecast 
        o_fore = self._init_model_instance(disable_weathering=False)
        o_fore.seed_within_polygon(
            lons=lons,
            lats=lats,
            time=obs_time,
            number=100,
            oil_type='GENERIC MEDIUM CRUDE'
        )

        o_fore.run(duration=timedelta(hours=forecast_hours), time_step=900)

        forecast_history = self._extract_summary_trajectory(o_fore.result)

        slick_id = (
            geojson.get("image_id") or 
            geojson.get("properties", {}).get("slick_id", "SLICK_UNKNOWN")
        )

        return {
            "slick_id": slick_id,
            "observation_time": obs_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "metocean_forcing": {
                "sources": [str(s) for s in self.metocean_sources]
            },
            "origin_estimation": {
                "estimated_spill_time_utc": estimated_spill_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "spill_age_hours": hindcast_hours,
                "bounding_box": origin_bbox,
                "centroid": {
                    "lon": float(np.mean(valid_origin_lons)),
                    "lat": float(np.mean(valid_origin_lats))
                }
            },
            "hindcast_trajectories": hindcast_history,
            "forecast_trajectories": forecast_history
        }