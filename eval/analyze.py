import json
import os

path = "./outputs" # path to your outputs
files = os.listdir(path)

id2task2avg = {}
task2uid = {}

for file in files:
    if not file.endswith("_reward_cache.json"):
        continue

    id = file.split("_reward_cache")[0]
    id2task2avg[id] = {}
    with open(os.path.join(path, file), "r") as f:
        data = json.load(f)
    
    all_travel_avg = []

    for task in data.keys():
        if task == "interact":
            continue

        all_avg = []

        for uid, content in data[task].items():
            if "travel" not in task:
                if task not in task2uid:
                    task2uid[task] = []
                if uid not in task2uid[task]:
                    task2uid[task].append(uid)
            else:
                if "travel" not in task2uid:
                    task2uid["travel"] = []
                if uid not in task2uid["travel"]:
                    task2uid["travel"].append(uid)

            if "travel" not in task:
                rewards = content["reward"]
                final_rewards = []
                for reward in rewards:
                    final_rewards.append(sum(reward))
                avg_final_reward = sum(final_rewards) / len(final_rewards) if final_rewards else 0
                all_avg.append(avg_final_reward)
            else:
                histories = content["history"]
                final_rewards = []
                aspect_num = len(task.split("travel")[-1])
                for history in histories:
                    answer_cache = {}
                    for turn in history:
                        if turn["choice"] == "answer" and turn["content"] != "":
                            initial = turn["content"][0]
                            if initial not in answer_cache:
                                answer_cache[initial] = 0.0
                            answer_cache[initial] = max(answer_cache[initial], turn["reward"])
                    reward = sum(list(answer_cache.values())) / aspect_num
                    final_rewards.append(reward)
                avg_final_reward = sum(final_rewards) / len(final_rewards) if final_rewards else 0
                all_travel_avg.append(avg_final_reward)
        
        if "travel" not in task:
            id2task2avg[id][task] = sum(all_avg) / len(all_avg) if all_avg else 0
    
    id2task2avg[id]["travel"] = sum(all_travel_avg) / len(all_travel_avg) if all_travel_avg else 0

task2number = {task: len(uids) for task, uids in task2uid.items()}

# For each diction, sort id based on all the tasks's weighted avg reward based on number in each task
id2task2avg = dict(sorted(id2task2avg.items(), key=lambda item: sum(item[1][task] * task2number[task] for task in item[1]) / sum(task2number[task] for task in item[1]), reverse=True))
id2weightedavg = {id: sum(task2number[task] * task2avg for task, task2avg in task2avg_dict.items()) / sum(task2number[task] for task in task2avg_dict) for id, task2avg_dict in id2task2avg.items()}

print("Weighted Avg Reward Ranking:")
avg_sorted_list = sorted(id2weightedavg.items(), key=lambda item: item[1], reverse=True)
for rank, (id, score) in enumerate(avg_sorted_list, start=1):
    print(f"Rank {rank}: {id} with score {score:.4f}")
print()


# TODO: Print two Latex Tables showing the results. One for max and one for avg.
# tasks = {"travel": "TravelGym", "turtle": "TurtleGym", "function": "FunctionGym", "tau": "TauGym", "persuasion": "PersuadeGym", "intention": "IntentionGym", "telepathy": "TelepathyGym", "bamboogle": "SearchGym"}

# print("\n\nAvg Reward Table:")
# header = "Model"
# for task in list(tasks.values()):
#     header += f" & {task}"
# header += " & Avg. \\\\ \n\\midrule"
# print(header)
# for id, task2avg in id2task2avg.items():
#     row = id
#     for task in list(tasks.keys()):
#         if task in task2avg:
#             row += f" & {(task2avg[task]):.4f}"
#         else:
#             row += " & 0.00"
#     row += f" & {id2weightedavg[id]:.4f}"
#     row += " \\\\ \n\\midrule"
#     print(row)
