import pytest
from aimguard.pipeline import build_baseline, run, evaluate
from aimguard.risk import policy

@pytest.fixture(scope="module")
def base():
    return build_baseline()

@pytest.mark.parametrize("seed", [11, 12, 13])
def test_all_legit_population_not_flagged(base, seed):
    *_, risk = run(base, n_players=300, cheater_rate=0.0, seed=seed)
    m = evaluate(risk)
    assert m["legit_fp_rate"] <= 0.01 and m["pro_false_pos"] == 0

def test_strong_cheats_detected_with_high_precision(base):
    tp = cheaters = fp = 0
    for seed in (21, 22, 23, 24, 25):   # pooled: one seed has too few cheaters to judge recall
        *_, risk = run(base, n_players=300, cheater_rate=0.08, strength_range=(0.6, 1.0), seed=seed)
        m = evaluate(risk); tp += m["true_pos"]; cheaters += m["cheaters"]; fp += m["false_pos"]
    assert tp / cheaters >= 0.9 and fp <= 1

def test_weak_cheats_do_not_create_false_positives(base):
    *_, risk = run(base, n_players=300, cheater_rate=0.08, strength_range=(0.15, 0.3), seed=31)
    assert evaluate(risk)["false_pos"] <= 1   # recall is allowed to be lower here

def test_statistics_alone_never_ban_eligible(base):
    *_, risk = run(base, n_players=300, cheater_rate=0.08, seed=41)
    assert "ban_eligible_after_review" not in set(risk.action)

def test_policy_requires_hard_evidence():
    assert policy(0.999, hard_evidence=False) == "human_review"
    assert policy(0.999, hard_evidence=True) == "ban_eligible_after_review"
    assert policy(0.5) == "monitor"


def test_tiny_reference_gives_clear_error():
    with pytest.raises(ValueError, match="reference players"):
        build_baseline(n_ref=10)

def test_window_length_mismatch_rejected(base):
    with pytest.raises(ValueError, match="window mismatch"):
        run(base, n_players=100, n_matches=4, seed=5)

def test_unseen_bracket_rejected(base):
    from aimguard.simulator import simulate
    eng, _ = simulate(50, n_matches=base.n_matches, seed=6)
    eng["bracket"] = 9
    with pytest.raises(ValueError, match="not present"):
        base.validate(eng)


# ---- signatures ----
def test_signature_matching_and_policy(tmp_path):
    from aimguard.signatures import sha256_file, load_blocklist, blocklist_matches, unknown_module_players
    f = tmp_path / "sample.bin"; f.write_bytes(b"pretend this is a captured sample")
    h = sha256_file(str(f))
    bl = tmp_path / "bl.txt"; bl.write_text(f"# header\n{h.upper()}  # from report #1\n\n")
    blocklist = load_blocklist(str(bl))
    reports = {"a": {"hashes": [h]}, "b": {"hashes": ["00" * 32]}, "c": {"unknown_modules": ["x.dll"]}}
    assert blocklist_matches(reports, blocklist) == {"a"}
    assert unknown_module_players(reports) == {"c"}

def test_hash_match_goes_to_review_not_auto_ban(base):
    *_, r0 = run(base, n_players=60, cheater_rate=0.0, seed=51)
    pid = r0.player_id.iloc[-1]                 # a clean player
    *_, r = run(base, n_players=60, cheater_rate=0.0, seed=51, hard_evidence={pid})
    row = r[r.player_id == pid].iloc[0]
    assert row.action == "human_review" and row.fired.get("hash_match") == 1

def test_unknown_modules_alone_never_trigger_review(base):
    from aimguard.simulator import simulate
    from aimguard.features import match_features, player_features
    from aimguard.risk import score_players
    eng, _ = simulate(60, n_matches=base.n_matches, cheater_rate=0.0, seed=52)
    mf = match_features(eng); pf = player_features(mf)
    pid = eng.player_id.iloc[-1]
    res = score_players(base.add_z(mf), base.player_tests(eng), base.iforest_flags(pf), None, {pid})
    assert res[res.player_id == pid].action.iloc[0] in ("none", "monitor")

def test_policy_hard_evidence_always_reviewed():
    from aimguard.risk import policy
    assert policy(0.05, hard_evidence=True) == "human_review"


# ---- input validation ----
@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_bad_n_matches_rejected(base, bad):
    with pytest.raises(ValueError, match="n_matches"):
        run(base, n_players=50, n_matches=bad)

@pytest.mark.parametrize("kw", [dict(cal_fraction=0), dict(cal_fraction=1), dict(cal_fraction=1.5),
                                dict(iforest_quantile=0), dict(iforest_quantile=1.2)])
def test_bad_calibration_settings_rejected(kw):
    from aimguard.detectors import Baseline
    with pytest.raises(ValueError, match="cal_fraction|iforest_quantile"):
        Baseline(**kw)

@pytest.mark.parametrize("kw", [dict(n_players=0), dict(cheater_rate=1.5), dict(cheater_rate=-0.1),
                                dict(strength_range=(0.9, 0.2)), dict(strength_range=(0.1, 1.5))])
def test_bad_simulation_settings_rejected(base, kw):
    with pytest.raises(ValueError):
        run(base, **{"n_players": 50, **kw})


# ---- ID normalization ----
def test_ids_are_strings_everywhere(tmp_path):
    from aimguard.io_utils import read_engagement_csv, normalize_reports, read_id_list
    from aimguard.signatures import blocklist_matches
    csvp = tmp_path / "live.csv"; csvp.write_text("player_id,match_id,x\n007,100,1\n 42 ,100,2\n")
    df = read_engagement_csv(str(csvp))
    assert df.player_id.tolist() == ["007", "42"] and df.match_id.tolist() == ["100", "100"]
    reports = normalize_reports({"007": {"hashes": ["aa"]}, 42: {"hashes": ["aa"]}, " 9 ": {}})
    assert set(reports) == {"007", "42", "9"} and set(reports) & set(df.player_id) == {"007", "42"}
    assert blocklist_matches(reports, {"aa"}) == {"007", "42"}
    idf = tmp_path / "ids.txt"; idf.write_text("007  # verified\n\n 42\n")
    assert read_id_list(str(idf)) == {"007", "42"}
