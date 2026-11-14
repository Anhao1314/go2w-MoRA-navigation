#!/usr/bin/env python3
"""Live interactive view of the SO-101 tabletop task (MuJoCo viewer window).

Opens a real MuJoCo viewer window and runs the scripted expert in real time
(30 fps wall-clock): the arm picks the four cubes and places them in the green
box.  Drag with the mouse to orbit/zoom.  The window closes when all four
cubes are placed (plus a short linger) or when you close it.
"""

from __future__ import annotations

import sys
import time

PROJECT_ROOT = __file__ and __import__("pathlib").Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    import mujoco
    import mujoco.viewer  # explicit: the viewer subpackage is not auto-imported

    from vla.env import CUBE_BODIES, SO101TabletopSession, make_so101_raw_env
    from vla.expert import PickPlacePhaseMachine

    env, _ = make_so101_raw_env()
    session = SO101TabletopSession(env)
    obs = session.reset(seed=0)
    world = session._ensure_world()
    print("viewer opening — drag to orbit, scroll to zoom")

    machines = {cube: PickPlacePhaseMachine() for cube in CUBE_BODIES}
    queue = list(CUBE_BODIES)
    current = queue[0]

    fps = 30.0
    with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
        obs = session.reset(seed=0)
        next_tick = time.time()
        placed_log = []
        while viewer.is_running() and queue:
            machine = machines[current]
            command = machine.update(session.expert_observation(obs, current))
            if command.done:
                placed = session.cubes_in_box()
                placed_log.append((current, placed))
                print(f"[live] {current}: done — cubes in box: {placed}/4")
                queue.pop(0)
                current = queue[0] if queue else None
                if current:
                    machines[current].reset()
                continue
            action = session.expert_action(command)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            next_tick += 1.0 / fps
            lag = time.time() - next_tick
            if lag > 0:
                time.sleep(min(lag, 0.1))
            else:
                next_tick = time.time()
        # linger so the last placement stays visible
        linger_until = time.time() + 8.0
        while viewer.is_running() and time.time() < linger_until:
            viewer.sync()
            time.sleep(1.0 / fps)
    print("done:", placed_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
