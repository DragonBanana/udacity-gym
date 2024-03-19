import time

from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "/media/banana/data/project/udacity-self-driving-car-dev/Builds/udacity.x86_64"

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
    while observation.input_image is None or observation.input_image.sum() == 0:
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    agent = PIDUdacityAgent(kp=0.055, kd=0.75, ki=0.000001)

    # Interacting with the gym environment
    for _ in range(20000000):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    simulator.close()
    env.close()
    print("Experiment concluded.")
