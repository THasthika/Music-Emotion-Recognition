pp=$1
shift

for x in $@
do
    python ./exec.py train $pp.$x
done