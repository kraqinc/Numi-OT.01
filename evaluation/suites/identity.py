"""Smoke checks that the model/prompt path respects Numi identity."""

from __future__ import annotations

from tokenizer.templates.system import RUNTIME_SYSTEM_PROMPT, ensure_system_message


def check_runtime_prompt_identity() -> dict[str, bool]:
    text = RUNTIME_SYSTEM_PROMPT
    return {
        "has_numi_ot": "Numi OT" in text,
        "has_motto": "solve problems" in text.lower(),
        "has_key": "Numi is the key" in text,
        "has_repos": "kraqinc/Numination" in text,
        "rejects_other_vendor_rule": "Never claim another" in text or "never claim another" in text.lower(),
    }


def check_ensure_system() -> bool:
    msgs = ensure_system_message([{"role": "user", "content": "What model are you?"}])
    return msgs[0]["role"] == "system" and "Numi OT" in msgs[0]["content"]


def run_identity_eval() -> dict[str, object]:
    checks = check_runtime_prompt_identity()
    checks["ensure_system"] = check_ensure_system()
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}
