# sCCN (sEMG-based Contrastive Convolutional Network)
Code of the paper IEEE JBHI "A Real-Time Hand Gesture Recognition System  for Low-Latency HMI via Transient HD-sEMG  and In-Sensor Computing"

dataset.py contains a dataloader class
1. data: [Batch, Height, width, Channel]
2. label: [Batch]
3. activations: [Batch]

evaluation.py shows how to evaluate the trained model using test dataset

model.py is the structure of the proposed sCCN
