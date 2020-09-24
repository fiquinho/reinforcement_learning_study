from subprocess import call
from pathlib import Path

configs_dir = Path("C://", "Users", "Fico", "rl_experiments",
                   "deep_q_learning", "mountain_car", "experiments_configurations", "24-08")

for file in configs_dir.iterdir():
    call_list = ["python", "train_agent.py",
                 "--name", str(file.stem),
                 "--config_file", str(file),
                 "--replace"]

    call(call_list)
