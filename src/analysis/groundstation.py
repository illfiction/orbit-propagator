import numpy as np


class GroundStation:
    def __init__(self, name, lat_deg, lon_deg, alt_km, min_elevation_deg):
        self.name = name
        self.lat_rad = np.radians(lat_deg)
        self.lon_rad_initial = np.radians(lon_deg)
        self.alt_km = alt_km
        self.min_elevation_deg = min_elevation_deg

    @classmethod
    def from_config(cls, config_dict: dict):
        """Creates a GroundStation object from a configuration dictionary."""
        return cls(
            name=config_dict["name"],
            lat_deg=config_dict["latitude_deg"],
            lon_deg=config_dict["longitude_deg"],
            alt_km=config_dict["altitude_km"],
            min_elevation_deg=config_dict["min_elevation_deg"],
        )
