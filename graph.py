from train_chatbot import *
import matplotlib.pyplot as plt
%matplotlib inline
acc = hist.history['accuracy']
loss = hist.history['loss']
epochs=200
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Accuracy and Loss')
plt.show()