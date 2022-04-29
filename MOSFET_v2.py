if __name__ == "__main__":
    import argparse
    import logging
    from itertools import product
    from nn_package_v2 import *
    import os
    import time
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="",
                        required=True,
                        help="type of model to be trained ks:keras Seq, kc:keras CNN, tf:TF raw, pt: pytorch")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available")

    parser.add_argument('-s',
                        '--save_model_dir',
                        default=None,
                        type=str,
                        required=False,
                        help="directory to save model to")


    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()
    
    if FLAGS.intel:
        logging.debug("Loading intel libraries...")
        import pandas as pd
        import numpy as np
        np.random.seed(1)
        # import tensorflow as tf
        # from tensorflow.python.util import _pywrap_util_port
        os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
        os.environ['DEVICE_COUNT'] = '1'
        os.environ['OMP_NUM_THREADS'] = '16'
        os.environ['KMP_SETTINGS'] = 'TRUE'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact'
        os.environ['KMP_BLOCKTIME'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' 
        print(torch.__version__)
        if '1.8.0' in torch.__version__:
            import intel_pytorch_extension as ipex
            device = ipex.DEVICE
        else:
            import intel_extension_for_pytorch as ipex


    else:
        logging.debug("Loading stock libraries...")
        import pandas as pd
        import numpy as np
        np.random.seed(1)
        '''
        import tensorflow as tf
        tf.random.set_seed(42)
        from tensorflow.python.util import _pywrap_util_port
        os.environ.pop('TF_ENABLE_MKL_NATIVE_FORMAT',None)
        os.environ.pop('TF_NUM_INTEROP_THREADS',None)
        os.environ.pop('TF_NUM_INTRAOP_THREADS',None)
        os.environ.pop('OMP_NUM_THREADS',None)
        os.environ.pop('KMP_AFFINITY',None)
        os.environ.pop('KMP_BLOCKTIME', None)
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        
        '''

    #logger.debug('MKL Enabled : %r', _pywrap_util_port.IsMklEnabled())
    logger.info("Starting Benchmarking tests")
    logger.info("\n")
    
    def sub_vth(w_l,vgs,vth,temp=300):
        """
        Helper function to calculate sub-vth current analytically.
        
        Uses randomized parameters to mimic 
        measurement noise and manufacturing/material variability
        """
        
        # Electron charge
        q = 1.60218e-19
        # Boltzman constant
        k = 1.3806e-23
        # Capacitance factor (randomized to mimic manufacturing variability)
        eta = 1.2+0.01*np.random.normal()
        # Mobility factor/coefficient (randomized to mimic material and manufacturing variability)
        w_l = w_l*(1+0.01*np.random.normal())
        # Mobility factor/coefficient (randomized to mimic material and manufacturing variability)
        temp = temp*(1+0.1*np.random.normal())
        v_th = w_l*np.exp(q*(vgs-vth)/(eta*k*temp))
        
        return eta, temp, v_th
    
    class MOSFET:
        """
        This defines a MOSFET object class which will have the properties of a MOSFET transistor.
        This is the virtual version of the digital twin       
        """
        def __init__(self,params=None,terminals=None):
            
            # Params
            if params is None:
                self._params_ = {'BV':20,
                                 'Vth':1.0,
                                 'gm':1e-2}
            else:
                self._params_ = params
            
            # Terminals
            if terminals is None:
                self._terminals_ = {'source':0.0,
                            'drain':0.0,
                            'gate':0.0}
            else:
                self._terminals_ = terminals
            
            # Determine state
            self._state_ = self.determine_state()
            
            # Leakage model trained?
            self._leakage_ = False
            self.leakage_model = None
        
        def __repr__(self):
            return "Digital Twin of a MOSFET"
        
        def determine_state(self,vgs=None):
            """
            """
            if vgs is None:
                vgs = self._terminals_['gate'] - self._terminals_['source']
            else:
                vgs = vgs
            if vgs > self._params_['Vth']:
                return 'ON'
            else:
                return 'OFF'
        
        def id_vd(self,vgs=None,vds=None,rounding=True):
            """
            Calculates drain-source current from terminal voltages and gm 
            """        
            if vds is None:
                vds = self._terminals_['drain'] - self._terminals_['source']
            else:
                vds = vds
            if vgs is None:
                vgs = self._terminals_['gate'] - self._terminals_['source']
            else:
                vgs = vgs
            
            vth = self._params_['Vth']
            state = self.determine_state(vgs=vgs)
            self._state_ = state
            
            if state=='ON':
                if vds <= vgs - vth:
                    ids = self._params_['gm']*(vgs - vth - (vds/2))*vds
                else:
                    ids = (self._params_['gm']/2)*(vgs-vth)**2
                if rounding:
                    return round(ids,3)
                else:
                    return ids
            else:
                return sub_vth(w_l=self._params_['gm'],
                               vgs=vgs,
                               vth=vth)
                #return 0.0
            
            
        def train_leakage(self,data=None,
                          batch_size=32,
                          epochs=15,
                          learning_rate=2e-5,
                          verbose=1):
            """
            Train Densely connected NN model
            """
            if data is None:
                return "No data to train with"
            X_train, X_test, \
            y_train, y_test = prepare_data(data,
                                                 input_cols=['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins'],
                                                 output_var='log-leakage',
                                                         scaley=False)
            # Deep-learning model
            model = build_model(num_layers=5,
                                architecture=[2,16,32,64,128,256,128,64,32,16,2],
                                input_dim=6)
            # Compile and train
            train_time, model_trained = compile_train_model(model,
                                                X_train,
                                                y_train,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                verbose=verbose)
            self.leakage_model = model_trained
            self.train_time_ks = train_time
            self._leakage_ = True

        def train_leakage_CNN(self,data=None,
                          batch_size=32,
                          epochs=10,
                          learning_rate=2e-4,
                          verbose=1):
            """
            Train CNN model
            """
            if data is None:
                return "No data to train with"
            X_train, X_test, \
            y_train, y_test = prepare_data(data,
                                                 input_cols=['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins'],
                                                 output_var='log-leakage',
                                                         scaley=False)
            X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
                                                         
            # Deep-learning model
            model = build_model_CNN()
            # Compile and train
            model_trained_CNN = compile_train_model_CNN(model,
                                                X_train,
                                                y_train,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                verbose=verbose)
            self.leakage_model_CNN = model_trained_CNN
            self._leakage_CNN_ = True
        
        def leakage(self,
                    w_l=1e-2,
                    vgs=None,
                    vth=None):
            """
            Calculates leakage current using the deep learning model
            """
            if not self._leakage_:
                return "Leakage model is not trained yet"
            # Vgs
            if vgs is None:
                vgs = self._terminals_['gate'] - self._terminals_['source']
            else:
                vgs = vgs
            # Vth
            if vth is None:
                vth = self._params_['Vth']
            else:
                vth = vth
            # Predict
            x = np.array([w_l,vgs,vth])
            ip = x.reshape(-1,3)
            result = float(self.leakage_model.predict(ip))
            
            return result


        def train_leakage_TF(self,data=None,
                          pred_df=None,
                          batch_size=32,
                          epochs=20,
                          learning_rate=2e-6,
                          verbose=1):
            """
            Train TF Raw model
            """
            if data is None:
                return "No data to train with"
            
            X_train, X_test, \
            y_train, y_test = prepare_data(data,
                                                 input_cols=['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins'],
                                                 output_var='log-leakage',
                                                         scaley=False)
            # Deep-learning model
            
            
            
            train_times, pred_times = model_TF_raw(X_train, y_train, pred_df,  
                                 batch_size=batch_size, epochs=epochs,
                                 learning_rate=learning_rate, verbose=verbose)
            
            self.train_times_TF = train_times
            self.pred_times_TF = pred_times
            self._leakage_ = True          
    
    logger.info("Generating synthetic data for Deep Learning model build")
    
    w_l_list = [1e-3*i for i in np.linspace(1,30.5,60)]
    vgs_list = [0.01*i for i in np.linspace(1,100.5,200)]
    vth_list = [0.05*i for i in np.linspace(21,40.5,40)]
    comb = list(product(w_l_list,vgs_list,vth_list))
    
    w_l_bins = [x*0.001 for x in [0, 6, 11, 16, 21, 26, 31]]
    w_l_labels = [1,2,3,4,5,6]
    vgs_bins = [x*0.01 for x in [0, 21, 41, 66, 81, 101]]
    vgs_labels = [1,2,3,4,5]
    vth_bins = [x*0.05 for x in [20, 26, 31, 36, 41]]
    vth_labels = [1,2,3,4]

    data_dict = {'w_l':[],'vgs':[],'vth':[],'eta':[],'temp':[],'sub-vth':[]}
    for c in comb:
        data_dict['w_l'].append(c[0])
        data_dict['vgs'].append(c[1])
        data_dict['vth'].append(c[2])
        eta, temp, v_th = sub_vth(c[0],c[1],c[2])
        data_dict['eta'].append(eta)
        data_dict['temp'].append(temp)
        data_dict['sub-vth'].append(v_th)
        
    df = pd.DataFrame(data=data_dict,columns=['w_l','vgs','vth', 'eta', 'temp','sub-vth'])
    df['w_l_bins'] =  pd.cut(df['w_l'], w_l_bins, labels=w_l_labels)
    df['vgs_bins'] =  pd.cut(df['vgs'], vgs_bins, labels=vgs_labels)
    df['vth_bins'] =  pd.cut(df['vth'], vth_bins, labels=vth_labels)

    df['log-leakage'] = -np.log10(df['sub-vth'])
    
    logger.info("Synthetic data generation complete")
    logger.info("=========> dataset shape:")
    logger.info(df.shape)
    logger.info(df.head())
    logger.info(df.tail())
    logger.info("\n")

    mosfet = MOSFET() # define mosfet object
    
    if FLAGS.model == 'ks':
        logger.info("Training a Keras Model")
        keras_train_times = []
        for counter in [1,2,3]:
            mosfet.train_leakage(df)
            keras_train_times.append(mosfet.train_time_ks)
        logger.info("Keras Model training complete")
        logger.info("=========> Keras Model Training times : {} secs".format(' '.join(map(str, keras_train_times))))
        logger.info("\n")

        logger.info("Inferencing a Keras Model")
        df_for_prediction = pd.concat([df,df,df,df])
        logger.info("=========> Pred Data Shape: ")
        logger.info(df_for_prediction.shape)
        arr_for_prediction = df_for_prediction[['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins']].to_numpy()
        keras_pred_times = []
        for counter in [1,2,3,4,5]:
            keras_pred_start = time.time()
            prediction = mosfet.leakage_model.predict(arr_for_prediction)
            keras_pred_end = time.time()
            keras_pred_times.append(keras_pred_end-keras_pred_start)
        logger.info("Keras Model Inferencing complete")
        logger.info("=========> Keras Model Inferencing time : {} secs".format(' '.join(map(str, keras_pred_times))))
        logger.info("\n")

    if FLAGS.model == 'kc':
        logger.info("Training a Keras CNN Model")
        keras_CNN_train_times = []
        for counter in [1,2,3]:
            keras_CNN_train_start = time.time()
            mosfet.train_leakage_CNN(df)
            keras_CNN_train_end = time.time()
            keras_CNN_train_times.append(keras_CNN_train_end-keras_CNN_train_start)
        logger.info("Keras CNN Model training complete")
        logger.info("=========> Keras CNN Model Training time : {} secs".format(' '.join(map(str, keras_CNN_train_times))))
        logger.info("\n")
        
        logger.info("Inferencing a Keras CNN Model")
        df_for_pred = pd.concat([df,df,df,df,df])
        arr_for_prediction = df_for_pred[['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins']].to_numpy()
        arr_for_prediction_reshaped = arr_for_prediction.reshape(arr_for_prediction.shape[0],arr_for_prediction.shape[1],1)
        mosfet.leakage_model_CNN.predict(arr_for_prediction_reshaped)
        keras_CNN_pred_times = []
        for counter in [1,2,3]:
            keras_CNN_pred_start = time.time()
            prediction = mosfet.leakage_model_CNN.predict(arr_for_prediction_reshaped)
            keras_CNN_pred_end = time.time()
            keras_CNN_pred_times.append(keras_CNN_pred_end-keras_CNN_pred_start)
        logger.info("Keras CNN Model Inferencing complete")
        logger.info(arr_for_prediction.shape)
        logger.info("=========> Keras CNN Model Inferencing time : {} secs".format(' '.join(map(str, keras_CNN_pred_times))))
        logger.info("\n")       

    
    if FLAGS.model == 'tf':
        df_for_pred = pd.concat([df,df,df,df,df])
        df_for_pred = df_for_pred[['w_l','vgs','vth','w_l_bins','vgs_bins','vth_bins']]
        logger.info("Starting Raw TF model build and inference")
        mosfet.train_leakage_TF(df,df_for_pred)
        logger.info("Raw TF model build and inference complete")
        logger.info("=========> TF Model Training time : {} secs".format(' '.join(map(str, mosfet.train_times_TF))))
        logger.info("=========> TF Model Inference time : {} secs".format(' '.join(map(str, mosfet.pred_times_TF))))
        logger.info("\n")
    
    if FLAGS.model == 'pt':
        """
        pytorch training and inference
        """
        
        class MLP(nn.Module):
            '''
            Multilayer Perceptron for regression.
            '''
            def __init__(self, num_layers):
                super(MLP, self).__init__()
                self.num_layers = num_layers
                self.layers = nn.ModuleList([nn.Linear(8, 32)])
                self.layers.extend([nn.Linear(32, 32) for i in range(1, self.num_layers-1)])
                self.layers.append(nn.Linear(32, 1))

            def forward(self, x):
                y = x
                for i in range(len(self.layers)):
                    y = self.layers[i](y)
                return y    
                 
        
        
        class Torch_data(torch.utils.data.Dataset):

            def __init__(self, data):
                X_train, X_test, y_train, y_test = prepare_data(data, input_cols=['w_l','vgs','vth','eta','temp','w_l_bins','vgs_bins','vth_bins'], output_var='log-leakage', scaley=False)        
                X = X_train.values
                y = y_train.values.reshape(-1,1)
                self.X = torch.from_numpy(X)
                self.y = torch.from_numpy(y)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, i):
                return self.X[i], self.y[i]    

        # building the dataset
        dataset = Torch_data(df)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True) # load data         
        logger.info("Starting PyTorch model build and inference")    
        # Run the training loop
        pytorch_train_times = []
        pytorch_pred_times = []
        mlp = MLP(num_layers=50)
        from torchsummary import summary
        summary(mlp, input_size=(1, 8))
        for counter in [1]:
            torch.manual_seed(42) # Set fixed random number seed
      
            
            # Definiting a multi layer perceptron
            mlp = MLP(num_layers=50) # Initialize the MLP
            loss_function = nn.MSELoss() # Define the loss function and optimizer
            optimizer = torch.optim.Adam(mlp.parameters(), lr =0.00005 ) 
            if '1.8.0' in torch.__version__:
                mlp = mlp.to(device)
            elif FLAGS.intel:
                mlp, optimizer = ipex.optimize(mlp, optimizer=optimizer)
                print("IPEX opt for training done with torch version", torch.__version__)
            

            current_loss = []
            pytorch_train_start_time = time.time()
            for epoch in range(0, 100): # run for five epochs
            # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):

                    # Get and prepare inputs
                    inputs, targets = data        
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))
                    if '1.8.0' in torch.__version__:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    
                    optimizer.zero_grad() # Zero the gradients
                    outputs = mlp(inputs) # Perform forward pass
                    loss = loss_function(outputs, targets) # Compute loss
                    loss.backward() # Perform backward pass
                    optimizer.step() # Perform optimization


                    # Print statistics
                    current_loss.append(loss.item())
                if epoch%10 == 0:
                    print('Loss after epoch %5d: %.3f' %(epoch, loss.item()))
            pytorch_train_end_time = time.time()
            pytorch_train_times.append(pytorch_train_end_time-pytorch_train_start_time)
        
        torch.save(mlp, 'PyTorch_model.h5')
        mlp_pred = torch.load('PyTorch_model.h5')
        if '1.8.0' in torch.__version__:
            mlp_pred.to(device)
        mlp_pred.eval()
        if '1.8.0' not in torch.__version__:
            if FLAGS.intel:
                mlp_pred = ipex.optimize(mlp_pred, dtype=torch.float32)
                print("IPEX opt for prediction done")
            
        for counter in [1,2,3,4,5]:
        
        # Prediction
            pytorch_pred_start_time = time.time()
            prediction = mlp_pred(torch.tensor(df[['w_l','vgs','vth','eta','temp','w_l_bins','vgs_bins','vth_bins']].values).float())
            pytorch_pred_end_time = time.time()
            pytorch_pred_times.append(pytorch_pred_end_time-pytorch_pred_start_time)
        logger.info("PyTorch model build and inference complete")
        logger.info("=========> PyTorch Model Training time : {} secs".format(' '.join(map(str, pytorch_train_times))))
        logger.info("=========> PyTorch Model Inference time : {} secs".format(' '.join(map(str, pytorch_pred_times))))
        logger.info("\n")
        