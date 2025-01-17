# N-MNIST Classification


# Get started

1. install dependencies

```
pip install -r requirements.txt
```

# Training the pruned network
```
python main.py

```

# Testing the pruned network on N-MNIST dataset using input-aware dynamic early stop policy

```
python early_stop.py
```

You can adjust the early stop threshold (ee_threshold) in ConvNet_ee of network.py file
