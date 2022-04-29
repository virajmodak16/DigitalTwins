#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
#tf.random.set_seed(42)

def prepare_data(df, input_cols,output_var,test_size=0.3,scaley=False):
    """
    """
    df1 = df.copy()
    X = df1[input_cols]
    y = df1[str(output_var)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return (X_train, X_test, y_train, y_test)


def build_model(num_layers=1, architecture=[32],act_func='relu', 
                input_dim=2):
    """
    Dense Neural Network
    """
    layers=[tf.keras.layers.Dense(input_dim,input_dim=input_dim)]
    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(architecture[i], activation=tf.nn.relu))
        layers.append(tf.keras.layers.Dense(1))

    model = tf.keras.models.Sequential(layers)
    return model

def build_model_CNN():
    """
    CNN Keras
    """
    print("building CNN model with Conv1D layer")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 2, activation="relu", input_shape=(6, 1)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model


def compile_train_model(model,x_train, y_train, learning_rate=0.001,batch_size=1,epochs=10,verbose=0):
    """
    Compile dense
    """
    tf.random.set_seed(42)
    model_copy = model
    model_copy.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     loss="mse", metrics=["mse"])
    start = time.time()
    model_copy.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                   verbose=verbose)
    train_time = time.time()-start
    return train_time,model_copy
    
def compile_train_model_CNN(model,x_train, y_train, learning_rate=0.001,batch_size=1,epochs=10,verbose=0):
    """
    Compile model
    """
    tf.random.set_seed(42)
    model_copy = model
    model_copy.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     loss="mse", metrics=["mse"])
    model_copy.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                   verbose=verbose)
    return model_copy

 
def model_TF_raw(x_train, y_train, pred_df, batch_size, epochs, learning_rate, verbose):
    """
    Trains a Tensorflow model with no Keras
    """
    print(type(x_train))
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.set_random_seed(42)
    n = len(x_train)
    
    x0 = tf.placeholder("float",[None,6])
    y0 = tf.placeholder("float",[None,1])
    
    # Layer 1 = the 3x4 hidden sigmoid
    m1 = tf.Variable( tf.random.uniform( [6,16] , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    b1 = tf.Variable( tf.random.uniform( [16]   , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    h1 = tf.matmul( x0,m1 ) + b1

    m2 = tf.Variable( tf.random.uniform( [16,16] , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    b2 = tf.Variable( tf.random.uniform( [16]   , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    h2 = tf.matmul( h1,m2 ) + b1    

    m3 = tf.Variable( tf.random.uniform( [16,8] , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    b3 = tf.Variable( tf.random.uniform( [8]   , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    h3 = tf.matmul( h2,m3 ) + b3 
    
    # Layer 2 = the 4x1 sigmoid output
    m4 = tf.Variable( tf.random.uniform( [8,1] , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    b4 = tf.Variable( tf.random.uniform( [1]   , minval=0.01 , maxval=0.9 , dtype=tf.float32  ))
    y_out = tf.matmul( h3,m4 ) + b4
    
    loss = tf.reduce_sum(tf.pow(y_out-y0, 2))/(2*n)
    
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        TF_train_times = []
        for counter in [1,2,3]:
            TF_train_start = time.time()
            sess.run( tf.global_variables_initializer() )
            for epoch in range(epochs):
                sess.run(train, feed_dict = {x0:x_train.values, y0:y_train.values.reshape(-1,1)})
                result = sess.run(loss, feed_dict = {x0:x_train.values, y0:y_train.values.reshape(-1,1)})
                print("Step", (epoch + 1), ": loss =", result)
            TF_train_end = time.time()
            TF_train_times.append(TF_train_end-TF_train_start)
        saved_path = saver.save(sess, 'my-model') # save last model to be used for prediction
    
    with tf.Session() as sess:
        saver.restore(sess, './my-model')
        TF_pred_times = []
        for counter in [1,2,3]:
            TF_pred_start = time.time()
            prediction = sess.run(y_out, feed_dict = {x0:pred_df.values})
            TF_pred_end = time.time()
            TF_pred_times.append(TF_pred_end-TF_pred_start)

    return TF_train_times, TF_pred_times