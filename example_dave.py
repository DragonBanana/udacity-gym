import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, DaveUdacityAgent
from udacity_gym.agent_callback import LogObservationCallback

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "/home/banana/projects/self-driving-car-sim/Builds/udacity_linux.x86_64"

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=simulator_exe_path,
        host=host,
        port=port,
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
        track="lake",
    )
    simulator.start()
    observation, _ = env.reset(track="lake")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    log_observation_callback = LogObservationCallback(pathlib.Path("dataset2"))
    agent = DaveUdacityAgent(checkpoint_path="dave2.ckpt",
                            before_action_callbacks=[],
                            after_action_callbacks=[log_observation_callback])

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(20000)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        print(observation.get_metrics())
        print(action.steering_angle)
        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")
