# sCCN (sEMG-based Contrastive Convolutional Network)
Code of the paper IEEE JBHI "A Real-Time Hand Gesture Recognition System  for Low-Latency HMI via Transient HD-sEMG  and In-Sensor Computing"

dataset.py contains a dataloader class
  -data: [Batch, Height, width, Channel]
  -label: [Batch]
  -activations: [Batch]
evaluation.py shows how to evaluate the trained model using test dataset
model.py is the structure of the proposed sCCN

