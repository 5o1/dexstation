# Platform restructure guards

Run comparisons from the repository root with the station's test interpreter:

```sh
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_golden_scenes.py tests/test_golden_episode.py tests/test_api_inventory.py tests/test_import_layers.py tests/test_calibration_single_copy.py
```

After reviewing a deliberate contract change, run the same command with
`DEXSTATION_UPDATE_GOLDEN=1` to rewrite all golden files. Normal runs never write
goldens and show unified diffs for changed snapshots. Review the resulting Git
diff before accepting updates; refactors should preserve these snapshots.

- `scenes/`: every task discovered at `configs/tasks/*/task.json`, composed with
  `ar5_08_g20_bench_bimanual`, backend name `sapien`, and seed 17. Every dataclass
  field is included. Paths under the repository become relative; paths under
  the temporary test root use `<tmp>`. The existing C2 rehashed bench copy's
  temporary directory maps back to its repository directory. Only
  `identity.code.git_commit` and `identity.code.dirty` are replaced with markers.
  JSON text fields and all hashes remain intact. Physics/assets are not loaded.
  Update mode removes scene goldens whose task profiles have been removed.
- `episode/`: two 50 Hz FakeBackend steps through production `Session`, with
  populated commands and workbench/task events. The relative file list and
  recursive keys of metadata and first journal rows are recorded. Scalar leaves
  become null; arrays contain sorted, distinct key shapes, so values and array
  lengths do not affect the snapshot. Fingertip mesh extraction is stubbed and
  integrity checking uses a thread instead of a subprocess.
- `routes.json`: all registered methods and canonical resource paths, including
  implicit HEAD and static routes; no HTTP server is bound.
- `import_layers.json`: explicit assignments for every Python module, plus the
  current reverse-import allowlist. All syntactic imports count, including lazy
  and type-checking imports. Relative imports resolve against package modules.
  Dynamic string imports are outside this static guard. The five peer layers
  (backends, inputs, recording, render, cameras) share one rank. Task code sits
  between runtime and server. Assign new modules explicitly before running.
  **Update mode never adds reverse imports.** Remove obsolete allowlist entries
  manually or let update mode prune them. Normal mode rejects stale entries.

The calibration guard also scans untracked/ignored repository files, excluding
Git metadata, and permits only the two bench calibration files.
