import tensorflow as tf
from keras import backend as K
import numpy as np
if __name__ == "__main__":
    tf.enable_eager_execution()
    tf.executing_eagerly() 
    x = [[2.]]
    m = tf.matmul(x, x)
    print("hello, {}".format(m))

    # [BATCH_SIZE, num_feature * num_groups]
    logits = np.ones((64, 4 * 6), dtype=np.float32)
    logits_ = tf.expand_dims(logits, -2)
    print(logits_.shape)
    sub_logits = logits_[:,:,0*6:(0+1)*6]
    print(sub_logits.shape)
    uniform = tf.random_uniform(shape =(64, 1, 6),
                minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
                maxval = 1.0)

    gumbel = - tf.log(-tf.log(uniform))
    noisy_logits = (gumbel + sub_logits)/0.1
    print(noisy_logits.shape)
    samples = K.softmax(noisy_logits)
    print(samples.shape)
    samples = K.max(samples, axis = 1)
    print(samples.shape)
    samples = tf.expand_dims(samples, -1)
    ss = tf.concat([samples, samples], 2)
    print(ss)
    print(tf.nn.top_k(logits[:,0:6], 1, sorted = True)[0][:,-1])