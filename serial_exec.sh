pp=$1
shift

for x in $@
do
    echo python ./exec.py train $pp.$x
done