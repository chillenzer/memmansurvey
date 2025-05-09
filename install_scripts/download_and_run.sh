set -x

if [ $# -eq 0 ]; then
  echo "No arguments supplied - architecture must be specified"
  exit 1
fi

dt=$(date +"%Y-%m-%d-%H:%M:%S")
filename=log_${dt}.txt

function main() {
  # We'll gather some information upfront:
  echo Running on ${dt} on $(hostname)
  git remote -v
  git log | head -n 1
  git status
  git diff
  nvidia-smi
  lscpu

  echo "Building for architecture $1"
  echo "Updating submodules"
  git submodule init
  git submodule update

  if [[ ! -f ./tests/graph_tests/data/orkut.mtx ]]; then
    ./install_scripts/graph_curl.sh
  fi

  echo "Downloads complete, compiling"
  mkdir -p build
  yes | python3 init.py
  python3 cleanAll.py
  python3 setupAll.py --cc $1
  mkdir -p results

  echo "Download finished, running experiments"

  yes | python3 testAll.py -mem_size 8 -device 0 -runtest -genres

  echo "Building PDF"

  ./install_scripts/process_results.sh
}

main $1 > >(tee -a $filename) 2>&1
