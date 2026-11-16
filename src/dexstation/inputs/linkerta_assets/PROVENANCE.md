# LinkerTA input example provenance

The reader and ROS adapter are project-owned Python implementations, independently
rewritten from the public protocol contract below and our validity specification.
They do not translate the modified station implementation. No upstream C++ source,
vendor binary, symlink, or original-workstation runtime read is retained here.
Privileged interface setup is supplied separately by the operator.

## File-level state

| Maintained file | Basis and ownership |
| --- | --- |
| `src/dexstation/inputs/linkerta_reader.py` | Project MIT rewrite of the public LinkerTA CAN contract and our validity/graph rules. Non-mutating project `CanSystem.check/require` retained. A temporary `bring_up` delegate exists solely for callers/tests being migrated separately. |
| `scripts/linkerta_reader_node.py` | Project MIT ROS adapter rewrite using the public JointState degree interface and our settings/status rules. |
| `src/dexstation/inputs/linkerta_assets/linkerta.yaml` | Project configuration, unchanged by the rewrite. Its old snapshot comment records historical configuration origin, not implementation ownership. |
| `linkerta_assets/sources.json` | Exact pinned public URLs and SHA256 values, file-level implementation state, and license-text source. |
| `linkerta_assets/APACHE-2.0.txt`, `linkerta_assets/NOTICE` | Full Apache-2.0 text and our explanatory attribution notice. The pinned upstream declares Apache-2.0 but has no LICENSE or NOTICE file. Our NOTICE is not represented as an upstream file. |

Project implementations: MIT, Copyright (c) 2026 5o1, under the repository LICENSE.
Public reference: [linker-bot/linkerbot-teleop-arm](https://github.com/linker-bot/linkerbot-teleop-arm),
formerly `linkerhand-tele-arm`, commit
`4354083b27fd69f9d042d78edb6924e67f7f03fc`.
The package's `package.xml:8` declares Apache-2.0. Full reference paths, downloaded
hashes and license-text provenance are recorded in [sources.json](sources.json).
No upstream copyright notice was present in the inspected package files.

## Public protocol trace

Paths and line numbers refer to that exact public commit's `src/linkerta/`.
They do not refer to the old station snapshot.

| Behavior | Public source |
| --- | --- |
| Empty `0x04` query, DLC 0 | `src/LinkerArm.h:23`; `src/LinkerArm.cpp:77–84`; `src/CanBus.cpp:170–173` |
| Reply IDs and order: `0x65` left J1–4, `0x66` left J5–7, `0x67` right J1–4, `0x68` right J5–7 | `src/LinkerArm.h:24–27`; `src/LinkerArm.cpp:164–185` |
| Signed big-endian int16, divide by ten for degrees | `src/LinkerArm.h:60–63` |
| Wrap to `[-180,180)`, reject delta >=180 degrees, round to tenths | `src/LinkerArm.h:63–81` |
| Raw SocketCAN, bind to interface, classic frame ID/DLC/data and close | `src/CanBus.cpp:41–45,69–77,135–139,170–173,188–213` |
| Degree-valued JointState interface, left joint61–67 and right joint51–57 | `src/main_ros2.cpp:55–59,69–81` |

The rewrite preserves the old Python reader's binary64 arithmetic order and
Python rounding, rather than claiming bit identity with the public C++ float32
implementation. The exact old/new Python gate includes every signed int16 value,
all four reply groups, stateful histories, and frozen fake-input scenarios.

## Our specification and configuration

Ruling RW2-1 treats values and standard library/ROS settings as our configuration;
they require no upstream numerical or setting comparison. This includes signs,
offsets, rates, timeouts, topic names, channels and QoS. No values changed.

QoS: best-effort, depth 1, chosen to match the collection subscriber.

The following project-owned rules supplement the public protocol:

- Initialize fourteen positions to NaN, receive times to None, rejection counts
  to zero and rejection reasons to None. Inspect the interface, then capture
  the initial monotonic time. Constructing a reader never opens a socket.
- Accept only exact reply IDs and complete payloads: eight bytes for four joints,
  six through eight for three. Ignore foreign IDs and CAN flags. Fault trailer
  bytes do not affect positions; do not query firmware version or calibrate zero.
- Capture one time per complete reply. The first measurement is accepted without
  comparison to fictional zero. Later jumps of >=180 degrees retain the old
  value and receive time, increment the counter and record the rejection reason.
  An accepted measurement clears that joint's rejection reason.
- Each step inspects the interface first. DOWN closes the owned bus and clears
  position validity while retaining historical receive times and diagnostics.
  Recovery requires new accepted replies. UP lazily opens a nonblocking bus,
  sends the query, then drains at most 256 frames. A missing frame ends the drain.
- Socket receive returns None on would-block, short classic frames or DLC >8.
  A short write raises `OSError('short CAN write')`. Other I/O errors propagate
  to the reader, which records its channel-specific error and closes its bus.
  No station-specific send retry or sleep logic is required or retained.
- Snapshot once per call. Clamp each measured age to at least zero. A side
  publishes only when all seven positions are finite and ages are at most the
  configured limit. Precedence is interface reason, I/O error, then the first
  invalid joint's rejection reason or formatted no-reply reason. Publish degrees
  using `position * negation + offset_deg` in that operation order.
- Status reports include ages, rejection counts/reasons and side validity.
  Transition identity omits changing silence durations and counters. The adapter
  emits status on transitions and at the configured rate, without per-frame logs.
- The fixed reader node identity is `dexhand_linkerta_reader`. Graph evidence
  contains sorted distinct foreign names and own endpoint GIDs per side.
  Startup rejects foreign publishers without stopping them. The collection
  receiver blocks a side with foreign publishers and refreshes immediately for
  an unconfirmed GID before accepting commands. That receiver/startup enforcement
  is outside this module and is not rewritten here.
- `CanSystem.check/require` resolve the configured role, inspect sysfs and issue
  only the read-only `ip ... link show` query. Privileged bring-up lives in
  `station/linkerta_can.py`; the temporary delegate preserves external callers.
  No ordinary test invokes real CAN, sudo, the root helper, or real ROS.

## Rewrite verification

`tests/test_linkerta_rewrite_equivalence.py` replays compact frozen inputs and
baseline outputs from commit `4dbb5373c7ffcf79cb580db00179828deb91f077`.
It checks typed containers, float bits, errors, timings, status, socket operations,
graph evidence and an adapter running exclusively against injected ROS stand-ins.
`tests/fixtures/linkerta_equivalence/README.md` records reproduction commands.
The old implementation is an external temporary oracle, never a runtime/test
source dependency. `scripts/research/linkerta_equivalence.py` can compare it to
this rewrite and run the complete signed-int16 sweep without devices or ROS.

The historical station diff was inspected during the rewrite. Its constructor,
bring-up, queue, retry, diagnostics and send-delay changes supply no required
station-only decoding or validity logic. The initial QoS stop was resolved by
RW2-1 as project configuration; it did not require a protocol change.
