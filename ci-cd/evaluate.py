import neptune.new as neptune

run = neptune.init(
    project="common/project-rl",
    name="evaluate",
    tags=["tmp"],
)

for i in range(100):
    run["test"].log(1.2)
