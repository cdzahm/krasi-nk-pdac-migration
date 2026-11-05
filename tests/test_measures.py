import numpy as np
import pandas as pd

from nkmigration import features, percell, states, stats
from nkmigration.recordings import load_recordings


def _track(xs, ys, t=None, shape_ok=True):
    n = len(xs)
    return pd.DataFrame({"track_id": 1, "t": t if t is not None else np.arange(n),
                         "x_um": xs, "y_um": ys, "aspect_ratio": 1.5, "eccentricity": 0.7,
                         "area_um2": 50.0, "equivalent_diameter_um": 8.0, "solidity": 0.9,
                         "circularity": 0.8, "orientation": 0.0, "use_for_shape": shape_ok})


def test_speed_arrest_and_confinement():
    m = features.track_measures(_track([0, 0.5, 3.5, 6.5], [0, 0, 0, 0]), pixel_um=1.0,
                                frame_min=1.0)
    assert np.isclose(m["speed_um_per_min"], 3.0)
    assert np.isclose(m["arrest_coefficient"], 1 / 3)
    assert np.isclose(m["confinement_ratio"], 1.0)
    assert np.isclose(m["axis_motion_angle_deg"], 0.0)  # moving along the long axis


def test_speed_uses_frame_interval():
    m = features.track_measures(_track([0, 2, 4], [0, 0, 0]), pixel_um=1.0, frame_min=2.0)
    assert np.isclose(m["speed_um_per_min"], 1.0)


def test_states_split_at_experiment_median():
    df = pd.DataFrame({"recording_id": "r", "experiment": "e", "track_id": 1,
                       "t": np.arange(5), "x_um": [0, 1, 3, 4, 8], "y_um": 0.0,
                       "aspect_ratio": [1, 1, 3, 1, 3], "use_for_shape": True})
    frames = states.frame_speeds(df)
    cuts = states.state_cutoffs(frames)
    st = states.assign_states(frames, cuts)
    assert np.isnan(st["state"].iloc[0])           # no step into the first frame
    assert list(st["state"].iloc[1:]) == [0, 3, 0, 3]
    cell = states.per_cell_states(states.episodes(st)).iloc[0]
    assert cell["stated_frames"] == 4 and cell["state_transitions"] == 3
    assert np.isclose(cell["elongated_fast_occupancy"], 0.5)


def test_inclusion_rules():
    cells = pd.DataFrame({m.name: [1.0, 1.0] for m in percell.MEASURES})
    cells["n_frames"] = [2, 3]
    cells["n_shape_frames"] = [3, 3]
    cells["n_axis_motion_steps"] = [3, 3]
    cells["stated_frames"] = [9, 10]
    out = percell.apply_inclusion(cells)
    assert out["speed_um_per_min"].isna().tolist() == [True, False]
    assert out["area_um2_median"].notna().all()
    assert out["round_slow_occupancy"].isna().tolist() == [True, False]


def test_stats():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert stats.stars(5e-5) == "****" and stats.stars(0.2) == "ns"
    assert stats.cliffs_delta(np.arange(10), np.arange(10) + 100) == 1.0
    res = stats.compare([np.arange(30.0), np.arange(30.0) + 0.5, np.arange(30.0) + 60])
    assert res["test"].startswith("Kruskal") and len(res["pairs"]) == 3


def test_recording_table():
    recs = load_recordings()
    assert len(recs) == 27 and len({r.recording_id for r in recs}) == 27
    arms = {r.recording_id: r.arm for r in recs}
    assert arms["HPAF2_B_IL15_KRASi"] == "KRASi"
    assert arms["HPAF2_B_TriKE_KRASi"] == "TriKE+KRASi"
    assert arms["Titration_A_0ng"] == "0 ng"
