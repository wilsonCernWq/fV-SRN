echo CMake
mkdir build
cd build
export CC=gcc-9
export CXX=g++-9
/home/qadwu/Software/cmake-3.22.1-linux-x86_64/bin/cmake -DTORCH_PATH=${CONDA_PREFIX}/lib/python3.8/site-packages/torch \
      -DGLM_INCLUDE_DIR=/home/qadwu/Software/glm-0.9.9.8/glm \
      -DRENDERER_BUILD_GUI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_CLI=OFF -DRENDERER_BUILD_TESTS=OFF -DRENDERER_BUILD_OPENGL_SUPPORT=OFF ..
make -j32 VERBOSE=true
cd ..

echo Setup-Tools build
python setup.py build
cp build/lib.linux-x86_64-cpython-38/pyrenderer.cpython-38-x86_64-linux-gnu.so bin/

echo Test
cd bin
python -c "import torch; print(torch.__version__); import pyrenderer; print('pyrenderer imported')"
