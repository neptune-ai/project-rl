import os
import random

import neptune.new as neptune

# Fetch project
project = neptune.get_project(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    name="common/project-rl",
)

# Find run with "in-prod" tag
runs_table_df = project.fetch_runs_table(tag="in-prod").to_pandas()
best_run_id = runs_table_df["sys/id"].values[0]

# Resume run
run = neptune.init(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project="common/project-rl",
    run=best_run_id,
)

# Run monitoring logic
# ... and log metadata to the run
run["production/monitoring/reward"].log(random.random() * 100)
