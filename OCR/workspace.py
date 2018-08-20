from persistence import *
from test_network import create_confusion_matrix
from table_def import *
import numpy as np
import itertools
import cv2
import math
from keras.preprocessing.image import ImageDataGenerator
import os
import keras.backend.tensorflow_backend as ktf
from sqlalchemy.orm import sessionmaker
from sqlalchemy import *
from back_end import *
import argparse

'''This file is for experimenting with various functions within python.
This file will not run on your computer.
I have included this file in my submission to show how I have developed
some of the algorithms used.

'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Used in creation of confusion_matrix
def get_percent_amt(test_label, mapping):
    numbers = [0] * len(mapping)
    for label in test_label:
        numbers[(int(np.argmax(label)))] += 1
    return numbers

'''
plot a confusion matrix using pyplot
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdYlGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    '''
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 '''
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'''
create the confusion matrix
'''
def get_graph():
    loc = "c:/matlab/emnist/emnist-bymerge"
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)
    model = load_yaml("c:/github/myproject/model/emnist-bymerge_12epoch_rec_transform")
    numbers = get_percent_amt(test_label)
    cm = create_confusion_matrix(model,test_img,test_label)
    print(cm)
    print(cm.shape)
    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            cm[x,y] = 100 * float(cm[x,y])/float(numbers[x])
    plt.figure(figsize=(15,10))
    plot_confusion_matrix(cm,mapping)
    plt.savefig('plot.png', dpi=200)

'''
get largest from a list
'''
def get_largest(ipt):
    return np.where(ipt==max(ipt))[0][0]

'''
get
'''
def get_centroids():
    loc = "c:/matlab/emnist/emnist-byclass"
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)
    centroids = [np.full((28,28,1),255, np.uint8)] * len(mapping)
    num_classes = get_percent_amt(test_label, mapping)
    for i in range(len(test_img)):
        pos = get_largest(test_label[i])
        newimg = test_img[i]
        centroids[pos] = (np.array(centroids[pos]) + np.array(test_img[i]))
    for j in range(len(centroids)):
        centroids[j] = centroids[j] / num_classes[j]
    return centroids


'''
plot all images in the data set on top of each other to show average pixel
distribution
'''
def plot_centroids(centroids):
    x = 0
    y = 0
    f, axarr = plt.subplots(16, 4)#math.ceil(len(centroids)/5)+1)
    plt.suptitle("Plot of Centroids of all Classes in EMNIST")
    print (axarr.shape)
    for i in range(len(centroids)):
        axarr[x,y].imshow(cv2.cvtColor(centroids[i], cv2.COLOR_GRAY2BGR), cmap='gray')
        x += 1
        if x == 16:
            x = 0
            y += 1
    x = axarr.shape[0]
    y = axarr.shape[1]
    for j in range(x):
        for p in range(y):
            axarr[j,p].axis('off')
    plt.show()

'''
print the sizes of the testing and training data sets
'''
def print_sizes():
    loc = ROOT_DIR+"training_data/emnist-bymerge"
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)
    print("test size" + str(len(test_img)))
    print("train size" + str(len(train_img)))

'''
plot the loss of a model
'''
def plot_loss():
    history = load_history(ROOT_DIR+'/model/emnist-letters_100epoch_lenet/history.pickle')
    print(history['acc'])
    for x in range(len(history['acc'])):
        history['acc'][x] = (1 - history['acc'][x])*100
        history['val_acc'][x] = (1 - history['val_acc'][x])*100
    print(history['acc'])
    plot_model_p(history)

'''
create a mapping dictionary
'''
def create_dict(loc):
    d = {}
    with open(loc) as f:
        for line in f:
           (key, val) = line.split()
           d[int(key)] = int(val)
    return d

'''
evaluate the model using custom data (unfinished)
'''
def eval_gen_mat():
    test_datagen = ImageDataGenerator(rescale=(1/255))
    val_gen = test_datagen.flow_from_directory(
            'c:/matlab/test_data',
            target_size=(28, 28),
            batch_size=10,
            color_mode='grayscale',
            shuffle='False',
            class_mode='categorical')

    model = load_yaml("c:/github/myproject/model/emnist-letters_1epoch_rec")
    #model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])
    history = model.predict_generator(val_gen)
    val_gen.classes
    confusion = np.zeros((26,26))
    for n in range(len(history)):
        enum_history = history[n]
        result = np.where(enum_history==max(enum_history))[0][0]
        actual = np.where(test_label[n]==max(test_label[n]))[0][0]
        confusion[actual][result] +=1
    confusion = confusion.astype(int)
    #history = model.evaluate_generator(val_gen)
    print(val_gen.classes)



'''
evaluate the model using custom data and plot a confusion matrix of
results (unfinished)
'''
def eval_gen():
    test_datagen = ImageDataGenerator(rescale=(1/255))
    val_gen = test_datagen.flow_from_directory(
            'c:/matlab/test_data',
            target_size=(28, 28),
            batch_size=10,
            color_mode='grayscale',
            shuffle='false',
            class_mode='categorical')

    model = load_yaml("c:/github/myproject/model/emnist-letters_1epoch_rec")
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])
    #history = model.predict_generator(val_gen)
    history = model.evaluate_generator(val_gen)
    print(model.metrics_names)
    print(history)

'''
manually evaluate a network (unfinished)
'''
def manual():
    image_list = []
    loc = 'c:/matlab/test_data/A'
    directory = os.fsencode(loc)
    model = load_yaml("c:/github/myproject/model/emnist-letters_24epoch_lenet")
    mapping = create_dict('c:/github/myproject/model/emnist-letters_12epoch_lenet/mapping.txt')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        im = cv2.imread('c:/matlab/test_data/A/' + filename, 0)
        im = np.invert(im)
        im = imresize(im, (28, 28))      # resize image
        im = im.reshape(28,28,1)       # shape to fit network input
        image_list.append(im)
    data = np.asarray(image_list)
    history = model.predict(data)
    for his in history:
        res = chr(mapping[(int(np.argmax(his)))])
        print(res)

'''
loop through all images in directories and invert them.
Used for generating my own data set
'''
def invert():
    loc = 'c:/matlab/letters/'
    directory = os.fsencode(loc)
    for subdir, dirs, files in os.walk(loc):
        for file in files:
            filename = os.path.join(subdir, file)
            im = cv2.imread(filename, 0)
            im = np.invert(im)
            cv2.imwrite(filename, im)

'''
resize all images in a folder
used for generating my own data set
'''
def resizeimgs(loc = 'c:/matlab/temp/'):
    directory = os.fsencode(loc)
    for subdir, dirs, files in os.walk(loc):
        for file in files:
            filename = os.path.join(subdir, file)
            im = cv2.imread(filename, 0)
            preshape = im.shape
            im = cv2.resize(im, (0,0), fx=0.25, fy=0.25)
            cv2.imwrite(filename,im)
            print(filename + ' was resized from: ' + str(preshape) + ' to: ' + str(im.shape))

'''
Move all images from one dir to another, change their names to not overwrite
images already in the file.
'''
def move_img_change_name(old = 'c:/matlab/letters/', new = 'c:/matlab/test_data'):
    all_img = []
    for subdir, dirs, files in os.walk(old):
        letter_list = []
        for file in files:
            filename = os.path.join(subdir, file)
            img = cv2.imread(filename, 0)
            letter_list.append(img)
        all_img.append(letter_list)
    count = 0
    for subdir, dirs, files in os.walk(new):
        for img in all_img[count]:
            newname = str(len(os.listdir(subdir))+1)
            filename = os.path.join(subdir, newname + '.jpg')
            cv2.imwrite(filename, img)
            print('image saved to: ' + filename)
        count += 1

'''
prevent memory errors with tensorflow
'''
def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

'''
save image as a thumbnail
'''
def save_thumbnail(user, loc):
        img = cv2.imread(loc)
        max_height = 200
        if(img.shape[0] < img.shape[1]):
            img = np.rot90(img)
        hpercent = (max_height/float(img.shape[0]))
        wsize = int((float(img.shape[1])*float(hpercent)))
        img = cv2.resize(img,(wsize,max_height))
        newname = str(len(os.listdir('static/users/{}/'.format(user)))+1)
        cv2.imwrite('static/users/{}/{}.png'.format(user,newname),img)

'''
save an image
'''
def save_image(user, img):
    newname = str(len(os.listdir('static/users/{}/'.format(user)))+1)
    cv2.imwrite('static/users/{}/{}.png'.format(user,newname),img)

'''
test the database
'''
def db_test():
    import uuid
    result = 'ABCD\nDEFG'
    print(result)
    engine = create_engine('sqlite:///tutorial.db', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    pred = Prediction("admin","password")
    session.add(user)


'''
print example of one-hot encoding to demonstate in report
'''
def print_one_hot():
    model = load_yaml("c:/github/myproject/model/emnist-letters_24epoch_lenet")
    loc = "c:/matlab/emnist/emnist-letters"
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)
    print(mapping)
    image = np.expand_dims(test_img[0], axis=0)
    res = model.predict(image)
    print(res.astype(float))

'''
Create the graphic for figure x.x in the report
'''
def get_incorrect_images(model, loc):
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)
    history = model.predict(test_img)
    incorrect = []
    for i in range(len(history)):
        if list(history[i]).index(max(history[i])) != list(test_label[i]).index(max(test_label[i])):
            incorrect.append((test_img[i], mapping[list(history[i]).index(max(history[i]))], mapping[list(test_label[i]).index(max(test_label[i]))]))
    interval = len(incorrect) / 100
    print(len(incorrect))
    cut = []
    for i in range(len(incorrect)):
        #pos = int(i * interval)
        cut.append(incorrect[i])

    plot_incorrect_class(cut)


def plot_incorrect_class(incorrect):
    rows = 6
    for num, tup in enumerate(incorrect):
        plt.subplot(rows,20,num+1)
        plt.title(str(tup[1]) + " -> " + str(tup[2]))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(tup[0],cv2.COLOR_GRAY2RGB), cmap='gray')
    plt.show()

def get_best_epoch(loc):
    history = load_history(loc)
    list = history.get('val_acc')
    print(list.index(max(list)))

def resize_to_scale():
    img = cv2.imread('c:/matlab/resize_test.jpg')
    copy = img.copy()
    max_height = 28
    hpercent = (max_height/float(img.shape[0]))
    wsize = int((float(img.shape[1])*float(hpercent)))
    img = cv2.resize(img,(wsize,max_height))
    print(img.shape)
    blank_image = np.full((28,28,3),255, np.uint8)
    img = cv2.resize(img, (0,0), fx=0.7, fy=0.7)
    x_offset=y_offset=7
    blank_image[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    cv2.imshow('scale',blank_image)
    cv2.waitKey(0)
    cv2.imshow('org',cv2.resize(copy,(28,28)))
    cv2.waitKey(0)

'''
test delete database
'''
def db_delete():
    import uuid
    engine = create_engine('sqlite:///tutorial.db', echo=True)
    Session = sessionmaker(bind=engine)
    s = Session()
    s.query(Prediction).filter(Prediction.id == '112').delete()
    s.commit()
    for r in result:
        print(r)
def ok():
    res = s.query(User).filter(User.username=="admin")
    print(type(res))
    identity = ""
    for r in res:
        identity = r.username
    if identity == 'admin':
        print('ok')

if __name__ == '__main__':
    ktf.set_session(get_session())
    loc = 'test_train_data/emnist/emnist-mnist'
    load_hdf5("model/emnist-digits_200epoch_lenet_rotate_zoom")
    (test_img, test_label), (train_img, train_label), mapping, number_classes = load_data(loc)



    #get_incorrect_images(model, loc)



    '''
    #classifierloc = ROOT_DIR+"/model/emnist-letters_300epoch_lenet_transform_rotate"
    #historyloc = ROOT_DIR+"/model/emnist-letters_200epoch_lenet_6/history.pickle"
    parser = argparse.ArgumentParser(usage='A workspace utility program')
    parser.add_argument('-m', '--model', type=str,
                        help='path to a saved model default is best performing')
    parser.add_argument('-get','--get_incorrect_images', action='store_true', default=False,
                        help='generate a graphic of incorrect images')
    parser.add_argument('--print_one_hot', action='store_true', default=False,
                        help='prints an example of one-hot encoding')
    parser.add_argument('-cm''--confusion_matrix', action='store_true', default=False,
                        help='create and display a confusion matrix')
    args = parser.parse_args()
    Session = sessionmaker(bind=engine)
    s = Session()
    res = s.query(User).filter(User.username == 'adin').first()

    if res == None:
        print('yay')
    if res != None:
        print('boo')

    #print(type(res))
    #res = s.query(User).filter(User.username == 'admin').first()
    #print(type(res))
    #print(type(res))
    #print(res.username)



    #resize_to_scale()
    #get_best_epoch(historyloc)
    '''


































#print uuid.uuid4()
#load_file_data()
#resizeimgs()
#invert()
#EOF
