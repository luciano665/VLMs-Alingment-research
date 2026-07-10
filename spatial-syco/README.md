# spatial_syco — spatially-grounded false-premise sycophancy benchmark

Scaffold for the REU project: separating **perception failure**, **hallucination**, and
**sycophantic capitulation** in medical VLMs when users assert spatially false premises
(wrong laterality / location / organ), using ground-truth SLAKE masks as the arbiter.

See `../REU_Weeks1-5_Report.md` for the full write-up (RQ, baselines, taxonomy, metrics).

## Layout
```
benchmark/
  build_benchmark.py    # SLAKE boxes/masks -> false-premise items (mask-contradicted flips)
  prompt_conditions.py  # 4-tier pressure ladder: neutral/misleading/confident/authority
eval/
  run_eval.py           # harness: runs items x conditions, grades, summarizes (pluggable backends)
  classify_response.py  # free-text -> {C,S,P,H,U} taxonomy (rule-based + LLM-judge hook)
  metrics.py            # FAR, PA-Syco (headline), CR, HR, flip rate, pressure slope
data/
  pilot_items.json      # 10 hand-built pilot items (Week-3 deliverable + test fixtures)
results/                # written by run_eval.py
```

## Quickstart (no GPU / no data needed — validates the whole pipeline)
```bash
python benchmark/build_benchmark.py --selftest     # verify laterality/quadrant logic
python benchmark/prompt_conditions.py              # see the 4 pressure prompts
python eval/classify_response.py                   # taxonomy classifier unit checks
python eval/metrics.py                             # metric schema on synthetic data
python eval/run_eval.py --dry-run --items data/pilot_items.json --out results
```
The dry run uses a deterministic **stub** model, so every number it prints is SYNTHETIC —
it exists to prove the plumbing works end-to-end before spending compute.

## Running for real (Weeks 4–5)
1. Download SLAKE, then:
   `python benchmark/build_benchmark.py --slake_root /path/to/SLAKE --out data/benchmark_v1.json`
2. Implement `generate()` in `OpenAICompatBackend` (API) or `HFBackend` (local VLM) in `run_eval.py`.
3. `python eval/run_eval.py --backend hf --model llava-med --items data/benchmark_v1.json --out results`
4. Repeat per model; compare `results/tables/*_summary.json`.

## The one thing to protect
`PA_Syco` (perception-adjusted sycophancy) is the headline metric: false-agreement rate
computed **only on items the model got right under the neutral prompt**. That conditioning
is what separates "chose to please" from "never could see it," and it's the contribution.
Don't report raw agreement without it.
