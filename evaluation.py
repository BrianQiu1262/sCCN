from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

DATA_PATH = './DATA/' # TODO
MODEL_PATH = './MODEL/' # TODO
DEVICE = '' # TODO

y_test = []
y_pred = []

train_loss = tf.keras.metrics.Mean(name='training_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='training_accuracy')
test_loss = tf.keras.metrics.Mean(name='val_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

test_data = DATA_PATH
test_dataset = Dataset(test_data, DEVICE)
test_dataset.dataset = test_dataset.dataset.batch(1)

model = tf.keras.models.load_model(MODEL_PATH)

@tf.function
def val_step(x, labels, act):
    x = x[0]
    act = act[0]

    x = x[act==1]

    preds1 = model(x, training=False)
    preds = tf.reduce_mean(preds1, 0, keepdims=True)
    t_loss = loss_func(labels, preds)

    test_loss(t_loss)
    test_accuracy(labels, preds)
    return preds

for val_data, val_labels, act in test_dataset.dataset:
    preds = val_step(val_data, val_labels, act)
    preds_cls = tf.argmax(preds[0]).numpy()
    y_pred.append(preds_cls)
    y_test.append(val_labels.numpy()[0])

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm = np.around(cm)
print(cm)

sns.heatmap(cm, annot=True, fmt="g", linewidths=0.3, cmap=plt.cm.Blues)
plt.savefig('cm.png', dpi=300)