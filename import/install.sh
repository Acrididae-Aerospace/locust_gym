cd PyFlyt --no-cache-dir
pip install -e .
cd ..
cd stable-baselines3 --no-cache-dir
pip install -e .
cd ..
cd custom_gyms
pip install -e . --config-settings editable_mode=compat --no-cache-dir
cd ..