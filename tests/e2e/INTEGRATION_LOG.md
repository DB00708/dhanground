# Integration log

Run: `python tests/e2e/run_e2e.py --data-root <main checkout> --work <scratch>` (native) or
`docker compose -f docker/docker-compose.yml run --rm --build e2e` (unit suite + e2e in Docker).

| Date | Task | Branch | SHA | Result | Notes |
|---|---|---|---|---|---|
| 2026-11-12 | harness baseline | main | 9b12a5e | PASS (native) | units 86 passed; e2e 9/9 stages in 26 s (py 3.11.5 venv + curatorkit 0.1.24). Docker NOT run: engine down. warmup SKIP (no scripts/warmup.py yet). |
