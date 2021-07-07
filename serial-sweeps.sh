count=$1
shift

for x in $@
do
    wandb agent thasthika/mer/$x --count $count
done
