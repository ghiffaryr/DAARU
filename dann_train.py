#checkpointer = ModelCheckpoint(filepath='F:/123/UNET/model/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
from keras_unet.metrics import iou, iou_thresholded

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        start = (int)(start)
        end = (int)(end)
        yield [d[start:end] for d in data]

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def dann_train(discriminator_model, source_classification_model, model, train_mode,epoch,source_batch,target_batch,val_batch,batch_size,save_dir,classifier_name):
    domain_history=[]
    class_history=[]
    val_class_history=[]
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
    y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))
    sample_weights_adversarial = np.ones((batch_size * 2,))
    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    j=0
    cost_class_prev=0
    cost_val_class_prev=0
    epoch = epoch*100
    for i in range(epoch):
        if (i % 100 == 0):
            start_time = int(time.time())
        if train_mode == 'dann':
            X0, y0 = source_batch.__next__()
            X1, y1 = target_batch.__next__()
            validation_x, validation_y = val_batch.__next__()
            validation_x, validation_y = validation_x.astype('float32'), validation_y.astype('float32')
            
            X_adv = np.concatenate([X0, X1])
            y_class = np.concatenate([y0, y0])
            #save class_weight and train domain_dann to get common feature
            class_weights = []
            for layer in model.layers:
                if (layer.name.startswith("class")):
                    class_weights.append(layer.get_weights())

            cost_domain = discriminator_model.train_on_batch(X_adv,y_adversarial_1)
            #update weight except class_weight
            k = 0
            for layer in model.layers:
                if(layer.name.startswith("class")):
                    layer.set_weights(class_weights[k])
                    k += 1

            #update class_weights
            adv_weights = []
            for layer in model.layers:
                if(layer.name.startswith("dis")):
                    adv_weights.append(layer.get_weights())
            for z in range(2):
                cost_class = source_classification_model.train_on_batch(X0,y0)
            k = 0
            for layer in model.layers:
                if(layer.name.startswith("dis")):
                    layer.set_weights(adv_weights[k])
                    k += 1
            
            if ((i + 1) % 100 == 0):
                end_time = int(time.time())
                iterate = int((j+1)/100)
                print('Epoch {:d}/{:d}'.format(iterate, int(epoch/100)))
                # print('Domain loss: %s [%.4f, %.4f]' % (discriminator_model.metrics_names, cost_domain[0], cost_domain[1]))
                # domain_history.append(cost_domain)
                # print('Segment loss: %s [%.4f, %.4f, %.4f]' % (source_classification_model.metrics_names, cost_class[0], cost_class[1], cost_class[2]))
                # class_history.append(cost_class)
                
                # y_output = source_classification_model.predict(validation_x)
                # val_loss = dice_loss(validation_y,y_output)
                # val_iou = iou(validation_y,y_output)
                # val_iou_thresholded = iou_thresholded(validation_y,y_output)
                # cost_val_class = [val_loss, val_iou, val_iou_thresholded]
                # val_class_history.append(cost_val_class)
                # print("Segment val loss: ['val_loss', 'val_iou', 'val_iou_thresholded'] [%.4f, %.4f, %.4f]" % (val_loss, val_iou, val_iou_thresholded))
                
                domain_history.append(cost_domain)
                class_history.append(cost_class)
                
                y_output = source_classification_model.predict(validation_x)
                val_loss = dice_loss(validation_y,y_output)
                val_iou = iou(validation_y,y_output)
                val_iou_thresholded = iou_thresholded(validation_y,y_output)
                cost_val_class = [val_loss, val_iou, val_iou_thresholded]
                val_class_history.append(cost_val_class)                

                print('{:d}s - domain_loss: {:.4f} - domain_acc: {:.4f} - class_loss: {:.4f} - class_iou: {:.4f} - class_iou_thresholded: {:.4f} - class_val_loss: {:.4f} - class_val_iou: {:.4f} - class_val_iou_thresholded: {:.4f}\n'.format(end_time-start_time, cost_domain[0], cost_domain[1], cost_class[0], cost_class[1], cost_class[2], val_loss, val_iou, val_iou_thresholded))
                
                if cost_val_class[2]>cost_val_class_prev :
                    source_classification_model.save_weights(save_dir+"/weight/best_weight_dann-"+classifier_name+".hdf5")
                    source_classification_model.save(save_dir+"/model/best_model_dann-"+classifier_name+".hdf5")
                    print('Epoch {:05d}: val_iou_thresholded improved from {:.4f} to {:.4f}, saving weight and model to {:s}'.format(iterate, cost_val_class_prev, cost_val_class[2], save_dir))
                    cost_val_class_prev = cost_val_class[2]
          
            if ((i + 1) % epoch == 0):
                model_json = source_classification_model.to_json()
                json_name = save_dir+"/model/best_model_dann-"+classifier_name+"+" + str(int(j/100)) + ".json"
                print(json_name)
                with open(json_name,"w") as json_file:
                    json_file.write(model_json)
                return domain_history,class_history,val_class_history
            j+=1
            
        if train_mode == 'class':
            X0, y0 = source_batch.__next__()
            X1, y1 = target_batch.__next__()
            cost_class = source_classification_model.train_on_batch(X0,y0)
            if ((i + 1) % 100 == 0):
                end_time = int(time.time())
                iterate = int((j+1)/100)
                print('Epoch {:d}/{:d}'.format(iterate,int(epoch/100)))
                class_history.append(cost_class)
                print('{:d}s - class_loss: {:.4f} - class_iou: {:.4f} - class_iou_thresholded: {:.4f}'.format(end_time-start_time, cost[0], cost[1], cost[2]))
                
                if cost_class[2]>cost_class_prev :
                    source_classification_model.save_weights(save_dir+"/weight/best_weight_class-"+classifier_name+".hdf5")
                    source_classification_model.save(save_dir+"/model/best_model_class-"+classifier_name+".hdf5")
                    print('Epoch {:05d}: iou_thresholded improved from {:.4f} to {:.4f}, saving weight and model to {:s}'.format(iterate, cost_class_prev, cost_class[2], save_dir))
                    cost_class_prev = cost_class[2]
                
            if ((i + 1) % epoch == 0):
                model_json = source_classification_model.to_json()
                json_name = "model+" + str(i) + ".json"
                print (json_name)
                with open(json_name,"w") as json_file:
                    json_file.write(model_json)
                return _,class_history
            j+=1

        if train_mode == 'domain':
            X0, y0 = source_batch.__next__()
            X1, y1 = target_batch.__next__()
            X_adv = np.concatenate([X0, X1])
            cost = discriminator_model.train_on_batch(X_adv,y_adversarial_1)
            write_log(callback, train_names, cost, i)
            if ((i + 1) % 100 == 0):
                end_time = int(time.time())
                iterate = int((j+1)/100)
                print('Epoch {:d}/{:d}'.format(iterate,int(epoch/100)))
                domain_history.append(cost)
                print ('{:d}s - domain_loss: {:.4f} - domain_acc: {:.4f}'.format(end_time-start_time, cost_domain[0], cost_domain[1]))
                return domain_history,_   
            j+=1

