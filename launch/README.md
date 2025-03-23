# Drone RL
Use PPO to test launching of drone

### Setup
login to wandb using
```bash
wandb login
```
The use the API key found in your dashboard

### Usage
For training
```python
python train.py --model <path to model>
```
For testing
```python
python test.py
```