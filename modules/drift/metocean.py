import logging
from opendrift.readers import reader_netCDF_CF_generic, reader_constant

logger = logging.getLogger(__name__)


def _build_constant_fallback_readers():
    r_curr = reader_constant.Reader({
        'x_sea_water_velocity': 0.3,
        'y_sea_water_velocity': 0.1
    })
    r_wind = reader_constant.Reader({
        'x_wind': 2.5,
        'y_wind': 4.0
    })
    return [r_curr, r_wind]


def attach_metocean_readers(openoil_instance, metocean_sources=None):
    readers_to_add = []

    if metocean_sources:
        for source in metocean_sources:
            if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
                # OPeNDAP / THREDDS stream
                logger.info(f"Connecting to remote dataset: {source}")
                try:
                    reader = reader_netCDF_CF_generic.Reader(source)
                    readers_to_add.append(reader)
                except Exception as exc:
                    logger.warning(
                        f"Failed to connect to remote dataset {source}: {exc}. Skipping this source."
                    )
            elif isinstance(source, str):
                # Local NetCDF file path
                logger.info(f"Loading local NetCDF dataset: {source}")
                try:
                    reader = reader_netCDF_CF_generic.Reader(source)
                    readers_to_add.append(reader)
                except Exception as exc:
                    logger.warning(
                        f"Failed to load local dataset {source}: {exc}. Skipping this source."
                    )
            else:
                readers_to_add.append(source)

        if not readers_to_add:
            logger.warning("All provided metocean sources failed to load. Using constant drift vectors only.")
    else:
        logger.warning("No metocean sources supplied. Using constant fallback drift vectors.")

    
    readers_to_add += _build_constant_fallback_readers()

    openoil_instance.add_reader(readers_to_add)

    return openoil_instance