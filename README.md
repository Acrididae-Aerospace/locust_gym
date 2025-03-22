# Simulation environment for launching

# Setup
Follow JSBsim setup [here](https://github.com/JSBSim-Team/jsbsim)
Download FlightGear [here](https://www.flightgear.org/download/)

# rename
echo 'alias fgfs="/Applications/FlightGear.app/Contents/MacOS/fgfs"' >> ~/.zshrc && source ~/.zshrc

# Compile from src
cd jsbsim
mkdir build
cd build
cmake -DINSTALL_PYTHON_MODULE=ON ..
make -j$(nproc)
sudo make install

# Mac
python -m venv locust_launch
source locust_launch/bin/activate
pip install -r requirements.txt

# verify
JSBSim --version
/Applications/FlightGear.app/Contents/MacOS/fgfs --version
import jsbsim (in python env)
