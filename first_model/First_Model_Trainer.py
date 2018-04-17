### Train your model here.
### Feel free to use as many code cells as needed.
import time
from sklearn.utils import shuffle
from Alexnet import * 
from First_Model_Data_Loader import *

x = tf.placeholder(tf.float32, (None, 227, 227, 3))
y = tf.placeholder(tf.int64, (None))

EPOCHS = 151
BATCH_SIZE = 128
LEARNING_RATE = 0.0009

logits = Alexnet(x)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
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
    print("Training with {} inputs...".format(len(X_train)))
    print()
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        start_time =  time.time()
        num_examples = len(X_train)
        X_training, y_training = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training[offset:end], y_training[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy, validation_loss, inference_data = evaluate(X_training, y_training)

        if (i % 10 == 0):
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Validation Loss = {:.3f}".format(validation_loss))
            print("Time Taken = {:.2f} sec".format(time.time() - start_time))
            print()

            
    saver.save(sess, './alexnet')
    print("Model saved")
