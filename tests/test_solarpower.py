import numpy as np

from constants import EARTH_RADIUS
from analysis import solarpower
from analysis.solarpower import compute_generated_power


class DummySatellite:
    def __init__(self, position):
        self._position = np.array(position, dtype=float)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.face_areas = np.array([0.5, 0.5, 0.5])

    @property
    def position(self):
        return self._position


SUN_DIR = np.array([1.0, 0.0, 0.0])


def _force_sun_direction(monkeypatch):
    monkeypatch.setattr(solarpower, "sun_direction_unit", lambda _t: SUN_DIR.copy())


def test_power_zero_during_eclipse(monkeypatch):
    _force_sun_direction(monkeypatch)
    position = -SUN_DIR * (EARTH_RADIUS + 400.0)
    sat = DummySatellite(position)

    power, in_eclipse = compute_generated_power(0.0, sat)

    assert in_eclipse is True
    assert power == 0.0


def test_power_positive_when_illuminated(monkeypatch):
    _force_sun_direction(monkeypatch)
    position = SUN_DIR * (EARTH_RADIUS + 400.0)
    sat = DummySatellite(position)

    custom = {
        "panel_area_m2": 2.0,
        "panel_efficiency": 0.3,
        "system_losses": 0.1,
        "panel_normal_body_frame": [1.0, 0.0, 0.0],
    }

    power, in_eclipse = compute_generated_power(0.0, sat, custom)

    expected = (
        1367.0
        * custom["panel_area_m2"]
        * custom["panel_efficiency"]
        * (1.0 - custom["system_losses"])
    )

    assert in_eclipse is False
    assert np.isclose(power, expected, rtol=1e-3)


def test_power_zero_when_panel_faces_away(monkeypatch):
    _force_sun_direction(monkeypatch)
    position = SUN_DIR * (EARTH_RADIUS + 400.0)
    sat = DummySatellite(position)

    config = {
        "panel_area_m2": 1.0,
        "panel_efficiency": 0.3,
        "panel_normal_body_frame": [-1.0, 0.0, 0.0],
    }

    power, in_eclipse = compute_generated_power(0.0, sat, config)

    assert in_eclipse is False
    assert power == 0.0
