"""Teleop motor values decode back to the joints the station HandCore encoded.

Station values pinned from linkerhand_retarget config/hand_config.yml (g20). A decode that
disagreed with them reversed the left thumb roll disc and finger side swing (2026-11-04,
the maintained reference configuration). Since 2026-11-12 the sim decodes left motor 5 as
the real hand does, and teleop values are mirrored on that slot first (TELEOP_MOTOR_REVERSE),
so the teleop path still inverts HandCore exactly.
"""
import unittest

from dexstation.examples.ar5_g20 import mapping as g

STATION_COMMAND = {
    "right": ([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, None, None, None, None, 255, 255, 255, 255, 255],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0, 0, 0, 0]),
    "left": ([255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, None, None, None, None, 255, 255, 255, 255, 255],
             [0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, None, None, None, None, 0, 0, 0, 0, 0]),
}


class StationInverseTests(unittest.TestCase):
    def test_command_tables_match_station_except_mirrored_teleop_slots(self):
        for hand, (lower, upper) in STATION_COMMAND.items():
            mirrored = set(g.TELEOP_MOTOR_REVERSE[hand])
            for m in range(g.G20_MOTOR_COUNT):
                expected = (upper[m], lower[m]) if m in mirrored else (lower[m], upper[m])
                self.assertEqual((g.G20_COMMAND_LOWER[hand][m], g.G20_COMMAND_UPPER[hand][m]), expected, (hand, m))
        self.assertEqual(g.TELEOP_MOTOR_REVERSE, {"left": (5,), "right": ()})


    def test_round_trip_through_station_encode(self):
        for hand, (lower, upper) in STATION_COMMAND.items():
            for frac in (0.0, 0.25, 0.6, 1.0):
                motors = [255.0] * g.G20_MOTOR_COUNT
                expected = {}
                for m in range(g.G20_MOTOR_COUNT):
                    if lower[m] is None:
                        continue
                    name = g.G20_JOINT_NAMES[g.G20_URDF_INDEX[hand][m]]
                    lo, hi = g.G20_JOINT_LIMITS[hand][name]
                    q = lo + frac * (hi - lo)
                    expected[name] = q
                    motors[m] = lower[m] + (q - lo) / (hi - lo) * (upper[m] - lower[m])
                decoded = dict(zip(g.G20_JOINT_NAMES, g.motor_positions_to_joint_positions(
                    hand, g.teleop_motor_positions(hand, motors))))
                for name, q in expected.items():
                    self.assertAlmostEqual(decoded[name], q, places=6, msg=f"{hand} {name} frac={frac}")


if __name__ == "__main__":
    unittest.main()
