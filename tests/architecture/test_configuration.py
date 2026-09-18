def test_project_structure():
    from pathlib import Path

    assert Path("architecture").is_dir()
    assert Path("configs").is_dir()
    assert Path("model").is_dir()
