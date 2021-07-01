function run_py_model() {
    rd=$1
    x=$2
    shift
    shift
    python $rd/model_pkgs/$x/simple_run.py --temp-folder x --dataset x --split x --check --only-shape $@
}