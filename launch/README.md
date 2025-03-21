# Simulation environment for launching

# Setup
Follow JSBsim setup [here](https://github.com/JSBSim-Team/jsbsim)
Download FlightGear [here](https://www.flightgear.org/download/)

# Compile from src
mkdir build
cd jsbsim
cmake -DINSTALL_PYTHON_MODULE=ON .
make

# Mac
python -m venv locust_launch
source locust_launch/bin/activate
pip install -r requirements.txt