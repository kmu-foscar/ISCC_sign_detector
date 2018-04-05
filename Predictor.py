### Run the predictions here.
### Feel free to use as many code cells as needed.

from Model_Tester import *

original_five = own_images[0:5]
sample_five = own_images_normalized[0:5]

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    inference_output = sess.run(inference_operation, feed_dict={x: sample_five})
    print(inference_output)
    
for (image, evaluated)  in zip(original_five, inference_output):
    plt.figure(figsize=(1,1))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(sign_names[evaluated])
    plt.imshow(image)
    plt.show()