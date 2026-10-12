#!/usr/bin/env python3
"""Quick local checks before integrating with the app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    ok = True
    checks = [
        Path("serving/api/server.py").exists(),
        Path("integrations/supabase/schema.sql").exists(),
        Path("integrations/supabase/functions/numi-chat/index.ts").exists(),
        Path("integrations/flutter/numi_client.dart").exists(),
        Path("configs/training/sft_smoke.yaml").exists(),
        Path("configs/serving/smoke.yaml").exists(),
        Path("docs/INTEGRATION.md").exists(),
    ]
    for c in checks:
        if not c:
            ok = False
    print("stack_files_ok" if ok else "stack_files_missing")

    ckpt = Path("outputs/sft_smoke/final.pt")
    print("checkpoint_smoke:" + ("yes" if ckpt.exists() else "no (run bootstrap_smoke.sh)"))

    try:
        from tokenizer.templates.system import ensure_system_message

        msgs = ensure_system_message(
            [{"role": "system", "content": "evil"}, {"role": "user", "content": "hi"}]
        )
        assert "evil" not in msgs[0]["content"]
        print("security_system_override:ok")
    except Exception as exc:
        print(f"security_system_override:fail:{exc}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
