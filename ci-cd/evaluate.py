import os

import neptune.new as neptune

# Fetch project
project = neptune.get_project(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    name="common/project-rl",
)

# Find run with best reward
runs_table_df = project.fetch_runs_table().to_pandas()
runs_table_df = runs_table_df.sort_values(by="training/episode/reward", ascending=False)
best_run_id = runs_table_df["sys/id"].values[0]

# Resume run
run = neptune.init(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    project="common/project-rl",
    run=best_run_id,
)

# Download agent
run["agent/policy_net"].download("policy_net.pth")

# ToDo
# Run evaluation logic, say 10 episodes
