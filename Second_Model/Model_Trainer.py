### Train your model here.
### Feel free to use as many code cells as needed.
import time
from sklearn.utils import shuffle
from Mininet import *
##import Mininet
from Preprocessor import *
import tensorflow as tf


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int64, (None))

EPOCHS = 10000
BATCH_SIZE = 128
#LEARNING_RATE = 0.00006
#LEARNING_RATE = 0.0009
LEARNING_RATE = 0.000009

logits = MiniNet(x)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

inference_operation = tf.argmax(logits, 1)
correct_prediction = tf.equal(inference_operation, y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    inference_data = np.array([])
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        accuracy, loss, inference = sess.run([accuracy_operation, loss_operation, inference_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
        inference_data = np.append(inference_data, inference)
    return total_accuracy / num_examples, total_loss / num_examples, inference_data

with tf.Session() as sess:
    print("Training with {} inputs...".format(len(X_training)))
    print()
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        start_time =  time.time()
        num_examples = len(X_training)
        X_training, y_training = shuffle(X_training, y_training)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training[offset:end], y_training[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy, validation_loss, inference_data = evaluate(X_validation, y_validation)

        if (i % 100 == 0):
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Validation Loss = {:.3f}".format(validation_loss))
            print("Time Taken = {:.2f} sec".format(time.time() - start_time))
            print()
            saver.save(sess, '.\lenet')

            
    #saver.save(sess, '.\lenet')
    print("Model saved")