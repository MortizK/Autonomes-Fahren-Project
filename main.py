import argparse

import cv2
import gymnasium as gym
import numpy as np
from car import Car
from env_wrapper import CarRacingEnvWrapper
from telemetry import TelemetryLogger


def _wheel_metrics(wheel) -> list[str]:
    forward = wheel.GetWorldVector((0, 1))
    side = wheel.GetWorldVector((1, 0))
    velocity = wheel.linearVelocity
    forward_speed = forward[0] * velocity[0] + forward[1] * velocity[1]
    side_speed = side[0] * velocity[0] + side[1] * velocity[1]
    road_friction = max(
        (getattr(tile, "road_friction", 0.0) for tile in wheel.tiles),
        default=0.0,
    )
    grip = road_friction if road_friction else 0.6
    on_road = bool(wheel.tiles)

    return [
        f"omega {wheel.omega:7.2f} rad/s",
        f"tire speed {wheel.omega * wheel.wheel_rad:7.2f} m/s",
        f"wheel speed {np.linalg.norm(velocity):7.2f} m/s",
        f"forward {forward_speed:7.2f}  side {side_speed:7.2f} m/s",
        f"gas {wheel.gas:5.2f}  brake {wheel.brake:5.2f}  steer {wheel.steer:5.2f}",
        f"grip {grip:5.2f}  road {'yes' if on_road else 'no'}",
    ]


def _text(panel, text, position, scale=0.45, color=(255, 255, 255)) -> None:
    cv2.putText(
        panel,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_debug_overlay(env, state_image: np.ndarray, info: dict, action=None) -> None:
    """Draw the observation and telemetry in a separate debug window."""
    car = env.unwrapped.car
    hull_velocity = car.hull.linearVelocity
    panel = np.zeros((620, 700, 3), dtype=np.uint8)
    state_bgr = cv2.cvtColor(state_image, cv2.COLOR_RGB2BGR)
    state_bgr = cv2.resize(state_bgr, (288, 288), interpolation=cv2.INTER_NEAREST)
    panel[8:296, 8:296] = state_bgr

    lines = [
        f"speed {info['speed']:.2f} m/s",
        f"velocity {hull_velocity[0]:.2f}, {hull_velocity[1]:.2f} m/s",
        f"position {car.hull.position[0]:.2f}, {car.hull.position[1]:.2f}",
        f"angle {car.hull.angle:.2f}  angular {car.hull.angularVelocity:.2f} rad/s",
        f"fuel {car.fuel_spent:.2f}",
    ]
    if action is not None:
        lines.append(f"action {action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}")

    _text(panel, "DEBUG TELEMETRY", (312, 24), 0.6, (255, 220, 120))
    for index, line in enumerate(lines):
        _text(panel, line, (312, 47 + index * 19), 0.55)

    for wheel_index, wheel in enumerate(car.wheels):
        x = 8 + wheel_index % 2 * 350
        y = 320 + wheel_index // 2 * 135
        _text(panel, f"WHEEL {wheel_index}", (x, y), 0.6, (120, 220, 255))
        for metric_index, metric in enumerate(_wheel_metrics(wheel)):
            _text(panel, metric, (x, y + 17 + metric_index * 16), 0.55)

    cv2.imshow("Car Racing - Debug", panel)
    cv2.waitKey(1)


def evaluate(env, eval_runs=50, eval_length=600, display_debug=True, telemetry_dir=None):
    seeds = [373468, 693592, 197453, 180888]
    telemetry = TelemetryLogger(telemetry_dir) if telemetry_dir else None

    episode_rewards = []
    for episode in range(eval_runs):

        seed = int(np.random.randint(0, int(1e6)))
        # TODO: Difficult seed for testing, remove when done
        if episode < len(seeds):
            seed = seeds[episode]
        state_image, info = env.reset(seed=seed)

        car = Car()
        if telemetry:
            telemetry.start_episode(episode, seed)

        episode_rewards.append(0.0)
        time_finished = 0
        for t in range(eval_length):

            action = car.next_action(state_image, info)

            state_image, r, done, trunc, info = env.step(action)
            episode_rewards[-1] += r
            if telemetry:
                telemetry.record(
                    step=t,
                    episode=episode,
                    seed=seed,
                    info=info,
                    action=action,
                    reward=r,
                    terminated=done,
                    truncated=trunc,
                    env=env,
                    car=car,
                )
            if display_debug:
                draw_debug_overlay(env, state_image, info, action)

            if done or trunc:
                time_finished = t
                break

        print(
            f"episode: {episode:02d}     "
            f"seed: {seed:06d}     "
            f"time: {time_finished:03d}     "
            f"reward: {episode_rewards[-1]:06.2F}     "
            f"avg 5 reward: {np.mean(np.asarray(episode_rewards[-5:])):06.2f}     "
            f"avg reward: {np.mean(np.asarray(episode_rewards)):06.2f}     "
        )

    print("---------------------------")
    print(" avg score: %f" % (np.mean(np.asarray(episode_rewards))))
    print(" std diff:  %f" % (np.std(np.asarray(episode_rewards))))
    print(" max score: %f" % (np.max(np.asarray(episode_rewards))))
    print(" min score: %f" % (np.min(np.asarray(episode_rewards))))
    print("---------------------------")
    print(" top 5 avg score: %f" % (np.mean(np.sort(np.asarray(episode_rewards))[-5:])))
    print(" low 5 avg score: %f" % (np.mean(np.sort(np.asarray(episode_rewards))[:5])))
    print("---------------------------")
    if telemetry:
        telemetry.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    parser.add_argument("--domain_randomize", action="store_true", default=False)
    parser.add_argument("--telemetry_dir", default=None, help="write per-step JSONL telemetry")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"
    env = CarRacingEnvWrapper(
        gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            domain_randomize=args.domain_randomize,
        )
    )

    evaluate(
        env,
        eval_runs=50,
        eval_length=2000,
        display_debug=not args.no_display,
        telemetry_dir=args.telemetry_dir,
    )

    env.reset()


if __name__ == "__main__":
    main()
