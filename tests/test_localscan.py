from aimguard.localscan import classify_file, unknown_game_modules

def levels(path, sha=None, bl=frozenset()):
    return {lvl for lvl, _ in classify_file(path, sha, bl)}

def test_blocklist_match():
    assert "MATCH" in levels(r"C:\Users\u\app.exe", "ab" * 32, {"ab" * 32})

def test_cheat_word_tokens_not_substrings():
    assert "REVIEW" in levels(r"C:\Users\u\Downloads\free_aimbot_v2.exe")
    assert "REVIEW" not in levels(r"C:\Program Files\ESPN\espnapp.exe")   # 'espnapp' is not the token 'esp'
    assert "REVIEW" not in levels(r"C:\Windows\System32\svchost.exe")

def test_dual_use_is_info_only():
    assert levels(r"C:\Program Files\Cheat Engine 7.5\cheatengine-x86_64.exe") == {"INFO"}

def test_risky_dir_info_but_trusted_dir_not():
    assert levels(r"C:\Users\u\AppData\Local\Temp\x.exe") == {"INFO"}
    assert levels(r"C:\Program Files\App\x.exe") == set()

def test_unknown_modules_filter():
    mods = [r"C:\Games\PUBG\engine.dll", r"C:\Windows\System32\kernel32.dll",
            r"C:\Users\u\Desktop\inject.dll", r"C:\Games\PUBG\readme.txt"]
    assert unknown_game_modules(mods, r"C:\Games\PUBG") == [r"C:\Users\u\Desktop\inject.dll"]


def test_scanner_ignores_itself():
    import sys
    from aimguard.localscan import is_self
    assert is_self(sys.executable)
    assert not is_self(r"C:\\Users\\u\\Downloads\\other\\python.exe", own=[sys.executable])


# ---- scanner path-boundary and hash-order regressions ----
from aimguard.localscan import in_dir, is_trusted_location, module_findings

def test_game_folder_check_respects_path_boundaries():
    assert in_dir(r"C:\Games\PUBG\mods\x.dll", r"C:\Games\PUBG")
    assert not in_dir(r"C:\Games\PUBG_cheats\inject.dll", r"C:\Games\PUBG")      # sibling folder with the same prefix
    assert not in_dir(r"C:\Games\PUBGx.dll", r"C:\Games\PUBG")

def test_trusted_locations_are_anchored_to_the_drive_root():
    assert is_trusted_location(r"C:\Windows\System32\kernel32.dll")
    assert is_trusted_location(r"C:\Program Files (x86)\App\a.dll")
    assert not is_trusted_location(r"C:\Users\u\Downloads\windows\evil.dll")        # folder merely NAMED windows
    assert not is_trusted_location(r"D:\stuff\program files\evil.dll")
    assert not is_trusted_location("/libevil/x.so")                                      # '/lib' prefix bug
    assert is_trusted_location("/usr/lib/x.so")

def test_blocklisted_module_is_caught_even_inside_trusted_or_game_folders():
    bad = "ab" * 32; hasher = lambda p: bad if "evil" in p else "00" * 32
    for path in (r"C:\Games\PUBG\evil.dll", r"C:\Windows\System32\evil.dll"):
        f = module_findings([path], r"C:\Games\PUBG", {bad}, hasher)
        assert [x["level"] for x in f] == ["MATCH"], path                                # hashed BEFORE any location exclusion

def test_clean_modules_in_trusted_or_game_folders_are_not_reported():
    f = module_findings([r"C:\Games\PUBG\engine.dll", r"C:\Windows\System32\k.dll"], r"C:\Games\PUBG", {"ab" * 32}, lambda p: "00" * 32)
    assert f == []

def test_unknown_module_reported_and_unreadable_file_noted():
    def hasher(p): raise OSError("locked")
    f = module_findings([r"C:\Users\u\Desktop\inject.dll"], r"C:\Games\PUBG", set(), hasher)
    assert {x["level"] for x in f} == {"REVIEW", "INFO"} and any("could not be read" in x["why"] for x in f)

def test_downloads_folder_named_windows_is_still_a_risky_location():
    assert "INFO" in {l for l, _ in classify_file(r"C:\Users\u\Downloads\windows\tool.exe")}
