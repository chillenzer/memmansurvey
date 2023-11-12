


if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
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


echo "Download finished, running experiments"

python3 testLarge.py -mem_size 8 -device 0 -runtest -genres