from evaluation.suites.smoke import run_smoke_eval


def test_smoke_eval_runs() -> None:
    report = run_smoke_eval(device="cpu")
    assert report.perplexity > 0
    assert isinstance(report.samples, list)
