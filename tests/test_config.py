from roofkit.config import LOCATIONS, Config


def test_locations_present():
    assert "karlsplatz" in LOCATIONS
    assert len(LOCATIONS["karlsplatz"]) == 4


def test_config_at_preset():
    cfg = Config.at("sonnwendviertel", seed=7)
    assert cfg.seed == 7
    assert (cfg.west, cfg.south, cfg.east, cfg.north) == LOCATIONS["sonnwendviertel"]


def test_defaults_are_karlsplatz():
    assert (Config().west, Config().south) == LOCATIONS["karlsplatz"][:2]
