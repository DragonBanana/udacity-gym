import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent
from udacity_gym.agent_callback import LogObservationCallback, PauseSimulationCallback, ResumeSimulationCallback

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "/home/banana/projects/self-driving-car-sim/Builds/udacity_linux.x86_64"

    # Track settings
    track = "lake"
    daytime = "day"
    weather = "sunny"

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=simulator_exe_path,
        host=host,
        port=port,
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )
    simulator.start()
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    log_observation_callback = LogObservationCallback(pathlib.Path(f"udacity_dataset_7/{track}_{weather}_{daytime}"))
    agent = PIDUdacityAgent(
        kp=0.06, kd=0.8, ki=0.000001,
        before_action_callbacks=[],
        after_action_callbacks=[log_observation_callback],
    )

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(2000)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")
