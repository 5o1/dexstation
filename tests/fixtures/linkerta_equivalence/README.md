# LinkerTA rewrite baseline

`4dbb537.json.gz` contains synthetic inputs and losslessly typed outputs from
`linkerta_reader.py` at `4dbb5373c7ffcf79cb580db00179828deb91f077`.
No old implementation, station source, real CAN capture, or actual ROS data is
included. The temporary oracle's SHA256 is
`acbed71553d87277394b0fb4a87685a72ecacc2bf84fa7bf4d6899b5be5b9511`.

The fixture has 55 cases, including 1,285 reader events and 4,377 fake wire frames;
seed `4354083` drives eight stateful random streams. Cases cover all four IDs,
signed extrema, wrap/jump boundaries, complete/short/overlong/flagged frames,
exact and adjacent expiry times, interleaving, flood limits, DOWN/recovery,
I/O failures, advancing clocks, affine configuration, validation errors, fake
native sockets, graph discovery and cleanup, foreign/unknown GIDs, and a ROS
adapter using only injected stand-ins. An unchanged collection receiver is also
exercised with old/new graph helpers to compare side blocking and decode errors.

Floats are encoded as their exact IEEE-754 binary64 bytes, including signed zero
and NaN; types, container order, error class/message, times, clock calls, status,
query bytes, and resource lifecycle are compared exactly. Fixture input values
include Python JSON's NaN/Infinity extensions for validation cases. Gzip's mtime
is zero. The fixture SHA256 is
`b63472b845b51d30f0ce8cbce79653a1f9468c4857c550922dcc6cf18c99f809`.

Run permanent tests from the repository root:

```sh
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_linkerta_rewrite_equivalence.py
```

For an independent oracle comparison, save the baseline reader and adapter
without executing them, then compare (the supplied interpreter is only an example
of a pytest-capable environment; no ROS/CAN installation is needed):

```sh
mkdir -p /tmp/rw2/old
# Run these only if no preserved baseline already exists.
git show 4dbb5373c7ffcf79cb580db00179828deb91f077:src/dexstation/inputs/linkerta_reader.py > /tmp/rw2/old/linkerta_reader.py
git show 4dbb5373c7ffcf79cb580db00179828deb91f077:scripts/linkerta_reader_node.py > /tmp/rw2/old/linkerta_reader_node.py
PYTHONPATH=src python scripts/research/linkerta_equivalence.py --old /tmp/rw2/old --exhaustive
```

The exhaustive mode additionally compares 524,288 old/new frames: every signed
int16 value, repeated at every joint slot in all four reply groups, once with
fresh positions and once with stateful histories. Its decoded-position byte
stream SHA256 is
`ae8c0fb38de77165645d0b895a8ddcbd122d95e820aefad86de698100cb03c44`.
It also compares receive timestamps, rejection counters and reasons directly.
This is Python old/new equivalence, not C++ float32 equivalence or physical
validation. No fake bus or fake ROS stand-in can open an actual CAN socket or
start a real ROS node.

To regenerate reviewed baseline data, append
`--freeze tests/fixtures/linkerta_equivalence/4dbb537.json.gz` to the research
command. The tool asserts old/new equality before writing. Tests themselves never
read `/tmp/rw2/old` or invoke the oracle loader or exhaustive sweep.

Two isolated temporary pytest mutations were required to fail: signed decoding
changed to unsigned (all four signed-boundary cases failed), and freshness `>`
changed to `>=` (the exact-boundary case failed). Both were applied only to the
imported module in separate Python processes, leaving source files unchanged.
Their invocation and output are recorded in `/tmp/rw2_report.md` and the temporary
evidence files; they are not installed runtime behavior or permanent tests.
