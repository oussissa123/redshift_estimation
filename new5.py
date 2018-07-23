from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, concatenate, Flatten, Dense, Add
from utils import *
from keras.models import Model
import keras
from keras.callbacks import LearningRateScheduler

def layer1(input_img, size = 32):
    #print(input_img.shape)
    #size = int(input_img.shape[3])
    tower_1 = Conv2D(size, 1, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(input_img) 
    tower_1 = Conv2D(size, 1, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_1) 
    tower_1 = Conv2D(size, 1, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_1) 
	
    tower_2 = Conv2D(size, 3, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(input_img) 
    tower_2 = Conv2D(size, 3, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_2) 
    tower_2 = Conv2D(size, 3, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_2)

    tower_3 = Conv2D(size, 5, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(input_img) 
    tower_3 = Conv2D(size, 5, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_3) 
    tower_3 = Conv2D(size, 5, activation='relu', strides = 1, kernel_regularizer=keras.regularizers.l2(0.00001), padding = "same")(tower_3) 
	
    tower = concatenate([tower_1, tower_2, tower_3])
    return tower   
	
def layer2(input_img):
    tower_1 = MaxPooling2D(pool_size = (2, 2), strides = 2)(input_img)    
    return tower_1  	

def layer(input_img, size = 32):
    sortie = layer1(input_img, size)
    sortie = layer2(sortie)
    return sortie
	
def get_final_model():
    shape = (32,32, 5)
    input_img = Input(shape=shape)    
    input = input_img
    dim = 8
    for i in range(5):
        dim = dim*2
        input_img = layer(input_img, size = dim)
    print(input_img.shape)
    FullyConnect = Flatten()(input_img)
    #FullyConnect = Dense(100, activation = "relu", kernel_regularizer=keras.regularizers.l2(0.00001))(input_img)
    #FullyConnect = Dense(100, activation = "relu", kernel_regularizer=keras.regularizers.l2(0.00001))(FullyConnect)
    FullyConnect = Dense(1)(FullyConnect)
    
    model = Model(inputs = input, outputs = FullyConnect)
    opt = keras.optimizers.SGD(lr=0.001, momentum = 0.9, decay = 0.2)
    opt = keras.optimizers.Adam()
    model.compile(loss=rmse_loss_keras, optimizer=opt)#, metrics=['accuracy'])
    print(model.output_shape)
    print('Okay ...')
    return model 
	
print('------------------------------------------------------------------ starting -----------------')

def scheduler(epoch, lr):
    lr_t = lr
    if epoch == 0:
         lr_t = 0.0001
    if epoch%2 == 0:
        lr_t = 0.5*lr_t    
    return lr_t
change_lr = LearningRateScheduler(scheduler, verbose = 1)

batch_size = 256
dir_img = '/home/peta/ouissa/images/QSO*.npy'
epoch = 30
model = get_final_model()

print('           |Model built')

X_Train, Y_Train, X_Test, Y_Test, X_Valid, Y_Valid = get_train_test_valid_data_galaxy1(dir_img,test_size=0.3, valid_size = 0.5)
stopearly = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='min')
history = model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[change_lr, stopearly])
#dire = '/home/etud/ouissa/stage/redshit_estimation/new/stage/modeles')
#batch_size = 10#128
#data = '../data/csvs/galaxies/all1_.csv';
#dir_img = '../data/images/galaxies1/all'

#Saving ploting
#save_model(model, './model_cnn_inception_module.json')
#plot_history(history);

print('             |Training ok ...')

#Testing and ploting of result
predict = model.predict(X_Test, batch_size=batch_size).reshape(-1)
result = compute_metrics(Y_Test, predict, 'Redshift')     
print(result)
save_pr_hist(predict, Y_Test, history)
#plot_result(Y_Test, predict)

print('----------------------------------------- End ---------------------------------------')
