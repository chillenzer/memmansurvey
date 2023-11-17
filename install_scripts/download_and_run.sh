


if [ $# -eq 0 ]
  then
    echo "No arguments supplied - architecture must be specified"
    exit 1
fi

echo "Building for architecture $1"

echo "Updating submodules"
git submodule init
git submodule update


./install_scripts/graph_curl.sh



echo "Downloads complete, compiling"
mkdir -p build
yes | python3 init.py
python3 cleanAll.py
python3 setupAll.py --cc $1
mkdir results


echo "Download finished, running experiments"

python3 testLarge.py -mem_size 8 -device 0 -runtest -genres


echo "Generating results"

python3 install_scripts/process_for_latex.py results artifact_data