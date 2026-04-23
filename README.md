# sCCN (sEMG-based Contrastive Convolutional Network)
Code of the paper IEEE JBHI "A Real-Time Hand Gesture Recognition System  for Low-Latency HMI via Transient HD-sEMG  and In-Sensor Computing"

## dataset.py 
It contains a dataloader class
- data: [Batch, Height, Width, Channel]
- label: [Batch]
- activations: [Batch]

## evaluation.py 
It shows how to evaluate the trained model using test dataset

## model.py 
The structure of the proposed sCCN
