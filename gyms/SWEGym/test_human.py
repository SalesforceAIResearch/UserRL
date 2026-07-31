import os

os.environ["OPENAI_API_KEY"] = "XXX"

if __name__ == "__main__":
    import swegym
    config = swegym.config.get_default_config()
    config.data_mode = "single"
    config.data_source = "triton-msvc-c4267-warnings"
    env = swegym.env.SWEEnv(config)
    obs, info = env.reset()
    print("Task: ", obs["task_id"])
    print("Instruction: ", obs["instruction"])
    while True:
        print("Choose from 1: action, 2: answer, 3: finish")
        human_input = input()
        human_input = int(human_input)
        if human_input == 3:
            string_to_send = "[finish]"
        else:
            print("Give me the contents")
            contents = input()
            if human_input == 1:
                string_to_send = f"[action] {contents}"
            elif human_input == 2:
                string_to_send = f"[answer] {contents}"
            else:
                print("Invalid input")
                continue
        observation, reward, terminated, truncated, info = env.step(string_to_send)
        feedback = observation["feedback"]
        print("Feedback: ", feedback)
        print("Reward: ", reward)
        print("Intents elicited: ", observation["intents_elicited"], "/", observation["total_intents"])
        print("Best score: ", observation["best_score"])
        print("--------------------------------")

        if terminated or truncated:
            print("Episode finished")
            break
    env.close()
