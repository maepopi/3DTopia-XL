CURRENT=$(pwd)
cd dva/mvp/extensions/mvpraymarch
make -j4
cd ../utils
make -j4
cd ${CURRENT}
pip install ./simple-knn
git clone https://github.com/ashawkey/cubvh
cd cubvh
pip install .
cd ${CURRENT}