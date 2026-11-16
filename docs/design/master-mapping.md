# Master-to-robot arm mapping by pose (LinkerTA → AR5)

Status: decided by the user on 2026-11-15 (English renderings: "Use B; when unreachable, stay at the last reachable state"; the sweep definition: "“large swing” should be defined in terms of spatial sweep distance", 2 cm for elbow, wrist and palm, "Execute allowed branch switches automatically"; hold recovery: "Use option A for hold handling"; timing: "The highest frame rate in the current research is 50 Hz; twice that is 100 Hz", "Accept over-run frames for now"). Deployed to both targets: our workstation, and the original workstation through a bridge node on our Git branch (WP13).

**Scope:** the LinkerTA → AR5 arm mapping only (the arm's 7 joints). The dexterous hand is driven by the retargeting algorithm and is out of scope; no finger state or fingertip point is used here. "Palm" below means the arm's end frame (the hand mount, rigid with link 7).

## 1. Problem

The original mapping copies joint angles: `robot = master + constant` per joint, then clamps each joint to the AR5 range. LinkerTA and the AR5 have different joint ranges (reference: `configs/workbenches/ar5_08_g20_bench_bimanual/calibration/master_linkerta_range.json`):

| Joint | AR5 | LinkerTA (observed) |
|---|---|---|
| J3 | ±178° | ±170°, but the mapping adds ±90° |
| J4 | −60 ~ 145° | about ±120° |
| J6 | ±60° | about ±107° |
| J7 | ±60° | continuous, any number of turns |

Clamping one joint while the others pass through changes the arm's shape. Two cases were seen on 2026-11-15 (station teleop logs):
- Rolling the right upper arm with the forearm fixed: master J3 +θ and J5 −θ. Once master J3 > ~85°, robot J3 is clamped at 175° while J5 still moves, so the robot turns its forearm instead of its upper arm.
- The elbow passing through straight: master J4 changes sign while J3/J5 swing by about 180° (the same arm shape, described by different joint numbers). The robot copies the swing and runs into the J3/J4 limits.

Rejecting a whole frame whenever any joint is out of range (the alternative the user considered) would have held the arm for 31.4% of the left-arm frames on 2026-11-05 (J6) and 28.5% of tonight's right-arm samples (J3, J4, J2), although most of those poses are reachable.

## 2. Mapping

Each control cycle, per arm:

1. **Intended pose.** `q_req = master + mapping constants + teleop calibration` (the current joint-copy result, **without clamping**, in degrees). Forward kinematics of the AR5 model at `q_req` gives:
   - the hand pose `T_req`: position and orientation of the arm tip;
   - the elbow point `e_req`: the J4 origin.

   This needs no LinkerTA kinematic model: LinkerTA was designed to correspond to the AR5 joint by joint. Joint sets that describe the same shape (J3±180°, −J4, J5±180°) give the same `T_req`, and full turns of J7 do not change it.
2. **Robot joints by optimization** over a feasible set, then lexicographic.
   - **Sweep.** The arm moves from its current state `q_now` (the motion reference in simulation, the measured joints on the real robot) toward a candidate `q`, each joint at most at the station's arm speed limit (`v_arm`, 40°/s), joints with less change arriving first. Along this executed path, sample the elbow point, the wrist center and the palm point. The sweep `σ(q)` is the largest distance of any of these points from the straight segment between its start and end positions. Moving straight from A to B is commanded motion; bulging away from that segment is a swing.
   - **Feasible set:** joint ranges with the 3° margin; hand error within the reachability tolerance (`ε_p`, `ε_R`); sweep `σ(q) ≤ σ_max` (2 cm). A branch switch near a straight elbow (the whole arm rolls about its own axis) sweeps little and is feasible; the same switch with a bent elbow swings the forearm and is not.
   - **Within the feasible set**, in order:
     1. hand error `E_hand(q) = ‖p(q) − p_req‖² / s_p² + ‖log(R_reqᵀ R(q))‖² / s_R²`, with a physical band `δ`: a candidate stays when its position error ≤ best + 0.5 mm and its orientation error ≤ best + 0.1°;
     2. elbow error `‖e(q) − e_req‖`, with a band `δ_e` (2 mm) so that negligible elbow differences do not cause switches;
     3. distance to the master's own joints `Σ_j ((q_j − q_req,j)/v_j)²` (continuous joints compared by full turns). This resolves singular configurations: at J2 = 0 the AR5's J1 and J3 are coaxial, and level 3 keeps the master's own split instead of letting J1/J3 drift;
     4. execution time `τ(q) = max_j |q_j − q_now,j| / v_arm` (smaller wins).
   - **Branch switches** that are feasible and win the order above happen automatically, without operator confirmation; a ~180° J3/J5 roll takes about 4.5 s at 40°/s. During it the hand stays within the sweep bound.
   - When `q_req` is inside the ranges and the previous output was the previous joint copy, `q_req` is zero on levels 1-3 and its sweep is the master's own small motion, so it is the optimum; no pass-through rule is needed.
   - Seeds: the previous output; `q_req` wrapped into range; its equivalent sets (J3±180°, −J4, J5±180°); J7 shifted by full turns.
   - Solver: warm-started damped Gauss-Newton with box constraints. The Levenberg damping is a numerical aid only; it is not part of the objective.
   - Speed, exact: an in-range `q_req` reached from its own previous joint copy, with `σ(q_req) ≤ σ_max`, is returned without solving.
   - Speed, bounded: otherwise the solver first runs only from the previous output. If that result is feasible, inside the level-1 band of zero, and has an elbow error of at most 1 mm (`elbow_certificate_m`), it is accepted. This approximation is accepted against the full search on reachability, the level-1 band and elbow error within 1 mm.
3. **Hold.**
   - When the feasible set is empty, the arm **stays at its last reachable output**. The reason is `unreachable` (no candidate within `ε_p`, `ε_R`) or `sweep` (reachable only through a path that swings more than `σ_max`). The output reports the reason, the error and the joint at its limit, for example `LEFT WRIST LIMIT`, shown large on screen with audio.
   - **Recovery (user decision A):** following resumes only when a feasible candidate exists from the held state, so a resumed motion also obeys the sweep limit. In practice the operator brings the master back near the held pose. No large catch-up motion is ever made. The operator must see which arm is holding, why, and how far the master is from the held pose: on our workstation as a large on-screen prompt with audio, combined with takeover Matching (if the resumed target is farther than the deviation budget from the held state, the arm goes to Matching instead of jumping); on the original workstation in the bridge's status topic and console log.
   - Replays of logs recorded under the old mapping cannot show recovery: the recorded operator never reacted to a hold. There, a hold often lasts to the end of the log (Nov 5 left arm: all frames). Recovery is validated in closed loop with an operator.
4. The result then passes through the existing safety layer (WP5 limits: collision and angle; takeover) unchanged.

## 3. Parameters

| Name | Meaning | Proposed |
|---|---|---|
| `s_p` | position scale | 0.01 m |
| `s_R` | orientation scale | 0.05 rad |
| `s_e` | elbow position scale | 0.05 m |
| `δ` | hand-error band for levels 2 and 3: a candidate is in the band when its position error ≤ best + 0.5 mm and its orientation error ≤ best + 0.1° | 0.5 mm, 0.1° |
| `v_j` | joint velocity limits used in level 3 | AR5 URDF velocity limits |
| `v_arm` | station arm speed limit used for the executed path and `τ` | 40°/s (motion model `arm.max_speed_deg_s`) |
| `σ_max` | sweep limit for the elbow, wrist center and palm | 0.02 m |
| `δ_e` | elbow band at level 2 | 2 mm |
| `ε_p`, `ε_R` | reachability tolerance | 2 mm, 1° |

All terms are logged per cycle. Sensitivity to `δ`, `δ_e`, `σ_max` and the seeds is reported before the values are fixed.

## 4. Acceptance

- Within the AR5 range, in continuous operation that starts in range (the previous output is the previous joint copy), the output equals the joint-copy result (difference ≤ 1e-6 rad). When the previous output sits on another joint set with the same pose (for example right after recovering from unreachable), the motion cost keeps it there; such frames are counted and their hand and elbow errors reported.
- The equivalent-set cases from 2026-11-15 (logs `rokae_right_impedance_teleop_node_128682`, `_122551`, interpolated to 100 Hz) give the intended hand pose: every frame is within `ε_p`, `ε_R` of it or holds (δ only ranks candidates; frames between δ and ε are counted). The old joint copy's hand error in the same frames is reported for comparison. With the elbow near straight, rolling the master's upper arm while holding the forearm is null-space motion (hand and elbow unchanged), so the robot is not required to follow that roll; the upper-arm roll difference is reported as information.
- J7 past ±180° gives the same output as the equivalent angle within range.
- An unreachable request (J6 = 90°) holds the last reachable output and reports which joint and how far.
- **Timing.** The target is 100 Hz (twice the fastest research method's 50 Hz task rate), i.e. one 10 ms cycle for everything computed per cycle. The mapping completes every query in the same cycle (no deferred search, no hold to wait for computation). Measured on the station CPU, one core, per arm: p50 about 0.13 ms (direct following), p95 7.1-7.5 ms and max 10.4-10.6 ms on the 2,902-frame worst-case window; full-search frames, especially while holding, take up to about 30 ms. The user accepts these over-run frames for now: a late cycle only delays the next update, and consumers process the latest input only (no backlog). Timing is measured and reported, not asserted.
- **Sweep certificate.** A candidate is sweep-feasible only with a proof that every tracked point stays within `σ_max` of its start-end segment: a closed-form arc-length or axis-radius bound, or sampling on the executed path (10 ms grid plus joint-arrival times) with the curvature error `B·h²/8` and local refinement of uncertified intervals. `B` includes both the centripetal and the rotating-axis terms (7.01 m/s² for the AR5). A refinement cap exists only as a numerical guard; on all replays it is never reached.
- Replaying the recorded teleop data (1 Hz logs interpolated to the 100 Hz master stream): the hold fraction, the hand error distribution and the largest per-cycle joint change are reported; every frame's sweep is ≤ `σ_max` or the frame holds with a reason; branch switches are counted, with their duration and sweep, and `sweep` holds are counted separately.
