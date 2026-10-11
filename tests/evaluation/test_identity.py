from evaluation.suites.identity import run_identity_eval


def test_identity_eval() -> None:
    report = run_identity_eval()
    assert report["passed"] is True
