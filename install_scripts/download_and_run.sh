
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