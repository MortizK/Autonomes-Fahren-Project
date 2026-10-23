import argparse
import time

import cv2
import gymnasium as gym
import numpy as np
from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection


def run(env, input_controller: InputController):
    seed = int(np.random.randint(0, int(1e6)))
    seed = 197453   # Seed with S-Curves and U-Turns
    print(f"seed: {seed:06d}")
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0
    
    # Initialization time of the lane detection algorithm
    start_time = time.time()
    lane_detection = LaneDetection()
    end_time = time.time()
    detection_times = []

    while not input_controller.quit:
        start_time = time.time()
        lane_detection.detect(state_image)
        end_time = time.time()
        detection_times.append(end_time - start_time)

        cv_image = np.asarray(lane_detection.debug_image, dtype=np.uint8)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(
            cv_image,
            np.asarray(state_image.shape[:2]) * 6,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Car Racing - Lane Detection", cv_image)
        cv2.waitKey(1)

        # Step the environment
        input_controller.update()
        a = [
            input_controller.steer,
            input_controller.accelerate,
            input_controller.brake,
        ]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # done = True     # Debug to Capture the first frame of each run

        # Reset environment if the run is skipped
        input_controller.update()
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")


            input_controller.skip = False
            lane_detection = LaneDetection()
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0

    # Print Boxplot values of the detection times
    print(f"p25 : {np.percentile(detection_times, 25):.4f} s   std : {np.std(detection_times):.4f} s")
    print(f"p50 : {np.percentile(detection_times, 50):.4f} s")
    print(f"p75 : {np.percentile(detection_times, 75):.4f} s")
    print(f"p95 : {np.percentile(detection_times, 95):.4f} s")
    print(f"p99 : {np.percentile(detection_times, 99):.4f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.no_display else "human"
    env = CarRacingEnvWrapper(
        gym.make("CarRacing-v3", render_mode=render_mode, domain_randomize=False)
    )
    input_controller = InputController()

    run(env, input_controller)
    env.reset()


if __name__ == "__main__":
    main()
