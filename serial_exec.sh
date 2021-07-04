for x in $@
do
    python ./exec.py train $x
done