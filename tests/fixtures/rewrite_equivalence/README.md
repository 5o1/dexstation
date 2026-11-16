# Rewrite equivalence fixture

`f07aa4c.json.gz` contains 5,408 old-output records from commit
`f07aa4c19ecd2a97967dc972d7a9003287e3cc35`, seed 731109. It is numerical
input/output data, with scalar types, numpy dtypes/shapes, container types and
exception classes/messages preserved. It contains no old implementation code.

The full research comparison checks 113,579 cases. The compact fixture retains
boundary cases and seeded samples, stratifies arm cases by side, startup mode,
calibration and use of StationMath, and retains every tested exception. It is
656,038 bytes. SHA256:
`6caa80bac88847fd56a4c3fabd374c9398b21952e9130037e2cbff08a704d089`.

Run the permanent test without a compiler or preserved source:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_rewrite_equivalence.py
```

To reproduce the full comparison, supply a preserved baseline source tree with
`src/dexstation/{g20_mapping,retarget_motor_clone,arm_mapping,station_math}.py`
and its original `motion_assets` and `retarget_assets` directories. Only this
research command needs a C++17 compiler:

```bash
PYTHONPATH=src python scripts/research/rewrite_equivalence.py \
  --old /path/to/preserved-baseline \
  --freeze /tmp/rewrite-equivalence.json.gz --summary /tmp/rewrite-equivalence.json
```

The generator loads the test's evaluator; tests never import the generator.
The old source is used as an oracle only. Provenance metadata is intentionally
updated for the new code and is checked separately; it is not a frozen numerical
output. Existing R0 fixtures under `tests/fixtures/golden/` are never regenerated.
