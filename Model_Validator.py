from sklearn.metrics import confusion_matrix
from Model_Trainer import *

# Test model accuracy
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_accuracy, test_loss, inference_data = evaluate(X_test_normalized, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print("Test Loss = {:.3f}".format(test_loss))
    
    plt.title('Confusion Matrix')
    plt.imshow(confusion_matrix(y_true = y_test,y_pred = inference_data))