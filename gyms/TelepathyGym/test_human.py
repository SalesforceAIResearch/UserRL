import os

os.environ["OPENAI_API_KEY"] = "XXX"

if __name__ == "__main__":
    import telepathygym
    config = telepathygym.config.get_default_config()
    config.data_mode = "single"
    config.data_source = "Albert Einstein"
    env = telepathygym.env.TelepathyEnv(config)
    obs, info = env.reset()
    print("Observation: ", obs)
    while True:
        print("Choose from 1: search, 2: action, 3: answer, 4: finish")
        human_input = input()
        human_input = int(human_input)
        print("Give me the contents")
        contents = input()
        if human_input == 1:
            string_to_send = f"[search] {contents}"
        elif human_input == 2:
            string_to_send = f"[action] {contents}"
        elif human_input == 3:
            string_to_send = f"[answer] {contents}"
        elif human_input == 4:
            string_to_send = "[finish]"
        else:
            print("Invalid input")
            continue
        observation, reward, terminated, truncated, info = env.step(string_to_send)
        feedback = observation["feedback"]
        print("Feedback: ", feedback)
        print("Reward: ", reward)
        print("--------------------------------")

        if terminated or truncated:
            print("Episode finished")
            break
    env.close()