# PRIME in MNIST Inpainting

# Get started

1. install dependencies

```
pip install -r requirements.txt
```


# Training PRIME on MNIST Inpainting
```
python main_fsvae_denoise.py

```

Training settings are defined in `NetworkConfigs/*.yaml`.

args:
- name: [required] experiment name
- config: [required] config file path
- checkpoint: checkpoint path (if use pretrained model) 
- device: device id of gpu, default 0

You can watch the logs with below command and access http://localhost:8009/ 

```
tensorboard --logdir checkpoint --bind_all --port 8009
```

# Test PRIME on MNIST Inpainting with Early Stop
```
python denoise_ee_pred.py

```

# 




