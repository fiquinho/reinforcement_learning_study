from subprocess import call
from pathlib import Path

configs_dir = Path("C://", "Users", "Fico", "rl_experiments",
                   "policy_gradient", "Catcher", "experiments_configurations", "16-10-20")

experiments = [
    {
        "name": "00.05",
        "env": "Catcher",
        "agent": "naive",
        "config_file": str(Path(configs_dir, "00.05_naive_default.json")),
        "desc": "Normalizing the state values (x/32).",
    },
    {
        "name": "00.01",
        "env": "Catcher",
        "agent": "reward_to_go",
        "config_file": str(Path(configs_dir, "00.01_reward_to_go_default.json")),
        "desc": "Normalizing the state values (x/32).",
    },
    {
        "name": "00.01",
        "env": "Catcher",
        "agent": "REINFORCE",
        "config_file": str(Path(configs_dir, "00.01_REINFORCE_default.json")),
        "desc": "Normalizing the state values (x/32).",
    }

]

for config in experiments:
    call_list = ["python", "train_agent.py",
                 "--name", config["name"],
                 "--config_file", config["config_file"],
                 "--env", config["env"],
                 "--agent", config["agent"],
                 "--desc", config["desc"],
                 "--replace"]

    call(call_list)
