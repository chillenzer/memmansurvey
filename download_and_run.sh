
echo "Starting Downloads"
git submodule init
git submodule update
git lfs pull



echo "Downloads complete, compiling"
mkdir -p build
python3 init.py
python3 cleanAll.py
python3 setupAll.py --cc $1


echo "Download finished, running experiments"

python3 testLarge.py -mem_size 8 -device 0 -runtest -genres