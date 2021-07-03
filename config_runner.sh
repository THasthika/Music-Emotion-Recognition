function config_runner() {
    run_f=$1
    shift
    for r in $@
    do
        python $run_f $r
    done
}