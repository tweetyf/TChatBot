#!/usr/bin/python3
#encoding=utf8

import tensorflow as tf
from tensorflow import keras

import numpy as np # linear algebra
import xml.etree.ElementTree as ET # for parsing XML
import matplotlib.pyplot as plt # to show images 

from PIL import Image
import os
import sys
import pickle,time
import utils

print(tf.__version__)

#test, if size too large, the shrink it.
IMAGE_SIZE=(128,128)
IMAGE_SHAPE=(128,128,3)
LABELS={'English_setter': 117, 'Irish_wolfhound': 62, 'basset': 57, 'Appenzeller': 112, 
'cairn': 114, 'clumber': 71, 'silky_terrier': 108, 'Afghan_hound': 54, 'miniature_poodle': 97,
 'Blenheim_spaniel': 10, 'bull_mastiff': 119, 'groenendael': 78, 'flat-coated_retriever': 46, 
 'Border_collie': 118, 'Staffordshire_bullterrier': 29, 'Irish_setter': 3, 'Airedale': 61, 'Pomeranian': 93, 
 'Sealyham_terrier': 38, 'Yorkshire_terrier': 21, 'collie': 116, 'Gordon_setter': 84, 'miniature_schnauzer': 60, 
 'Welsh_springer_spaniel': 51, 'giant_schnauzer': 36, 'Kerry_blue_terrier': 42, 'Weimaraner': 14, 'Labrador_retriever': 73, 
 'Doberman': 59, 'cocker_spaniel': 31, 'Bernese_mountain_dog': 24, 'boxer': 26, 'Newfoundland': 105, 
 'beagle': 7, 'toy_terrier': 96, 'Dandie_Dinmont': 103, 'Brittany_spaniel': 22, 'EntleBucher': 107, 
 'borzoi': 75, 'dingo': 43, 'Japanese_spaniel': 102, 'French_bulldog': 86, 'Siberian_husky': 45, 'Border_terrier': 87, 
 'Greater_Swiss_Mountain_dog': 23, 'otterhound': 41, 'black-and-tan_coonhound': 63, 'whippet': 15, 'Mexican_hairless': 55, 
 'pug': 66, 'briard': 99, 'kuvasz': 40, 'Bedlington_terrier': 44, 'toy_poodle': 76, 'Norwegian_elkhound': 37, 'Lhasa': 91, 
 'komondor': 68, 'curly-coated_retriever': 13, 'Samoyed': 18, 'miniature_pinscher': 2, 'bluetick': 83, 'malinois': 52, 
 'Lakeland_terrier': 95, 'Saluki': 5, 'wire-haired_fox_terrier': 74, 'English_foxhound': 11, 'Ibizan_hound': 85, 
 'Leonberg': 104, 'Saint_Bernard': 34, 'keeshond': 27, 'malamute': 69, 'Maltese_dog': 106, 'Tibetan_terrier': 32, 
 'Irish_terrier': 47, 'standard_schnauzer': 20, 'Brabancon_griffon': 88, 'English_springer': 81, 'Walker_hound': 28,
  'Irish_water_spaniel': 49, 'Old_English_sheepdog': 115, 'Australian_terrier': 1, 'German_short-haired_pointer': 56, 
  'Scotch_terrier': 0, 'Shetland_sheepdog': 70, 'redbone': 16, 'vizsla': 67, 'golden_retriever': 25, 'Eskimo_dog': 94, 
  'Rhodesian_ridgeback': 58, 'West_Highland_white_terrier': 8, 'Chesapeake_Bay_retriever': 72, 'Sussex_spaniel': 90, 
  'Pekinese': 39, 'German_shepherd': 65, 'affenpinscher': 9, 'Bouvier_des_Flandres': 101, 'Tibetan_mastiff': 100, 
  'papillon': 4, 'Scottish_deerhound': 17, 'African_hunting_dog': 6, 'Norwich_terrier': 30, 'Boston_bull': 77, 'Rottweiler': 113, 
  'Shih-Tzu': 53, 'standard_poodle': 110, 'American_Staffordshire_terrier': 33, 'chow': 79, 'Cardigan': 19, 'Italian_greyhound': 80, 
  'dhole': 48, 'schipperke': 82, 'Great_Dane': 111, 'kelpie': 50, 'basenji': 92, 'Pembroke': 64, 'Norfolk_terrier': 12, 'bloodhound': 109, 
  'soft-coated_wheaten_terrier': 89, 'Chihuahua': 98, 'Great_Pyrenees': 35}
LABELS_R = {}
for itm in LABELS.keys():
    LABELS_R[LABELS[itm]] = itm

#============================Utils======================================
def showImg(img):
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def cleanWorkspace():
    print("cleanWorkspace")
    utils.run("rm -rf ./dogdataset/output/")
    utils.run("mkdir ./dogdataset/output")
    utils.run("mkdir ./dogdataset/output/chk")

#============================data preparation.==========================
def testReadSamples():
    breed_list = os.listdir('./dogdataset/input/Annotation/')
    figplot = plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(331 + i) # showing 9 random images
        breed = np.random.choice(breed_list) # random breed
        dog = np.random.choice(os.listdir('./dogdataset/input/Annotation/' + breed)) # random image 
        img = Image.open('./dogdataset/input/Images/' + breed + '/' + dog + '.jpg') 
        tree = ET.parse('./dogdataset/input/Annotation/' + breed + '/' + dog) # init parser for file given
        root = tree.getroot() # idk what's it but it's from documentation
        objects = root.findall('object') # finding all dogs. An array
        plt.imshow(img) # displays photo
        for o in objects:
            bndbox = o.find('bndbox') # reading border coordinates
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # showing border
            plt.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # printing breed
    figplot.savefig("./output/random_dog.png",format='png')

def splitSamples():
    # 9/10 for training, and 1/10 for testing
    divider=20
    annoObj=[]
    testObj=[]
    breed_list = os.listdir('./dogdataset/input/Annotation/')
    for breed in breed_list:
        dogs = os.listdir('./dogdataset/input/Annotation/' + breed) # image infos. 
        for dog in dogs:
            imgdir = './dogdataset/input/Images/' + breed + '/' + dog + '.jpg'
            infotree = ET.parse('./dogdataset/input/Annotation/' + breed + '/' + dog) # init parser for file given
            nodeInfo={"breed":breed, "dog":dog, "imgdir":imgdir, "infotree":infotree}
            _magic = np.random.choice(range(divider))
            if _magic==0: testObj.append(nodeInfo)
            else: annoObj.append(nodeInfo)
    np.random.shuffle(annoObj)
    np.random.shuffle(testObj)
    return annoObj,testObj

def loadRawData():
    # if the data not been loaded. then load, and save as python objects.
    anfileTrain ="./dogdataset/output/annoObj.train.obj"
    anfileTest ="./dogdataset/output/annoObj.test.obj"
    annoObjTrain = utils.loadObjsIfExist(anfileTrain)
    annoObjTest = utils.loadObjsIfExist(anfileTest)
    if not annoObjTrain:
        print("raw data not exist, loading....")
        annoObjTrain, annoObjTest = splitSamples()
        print("raw data loaded, saving as files: ",anfileTrain, anfileTest)
        utils.saveObj(annoObjTrain,anfileTrain)
        utils.saveObj(annoObjTest,anfileTest)
    print("raw data loaded.")
    return annoObjTrain, annoObjTest

def preProcessBatch(tag, annoObj):
    #if the data has not been pre-processed, the process the data.
    #1. generate the label sets.
    #2. save configure info.
    confFile ="./dogdataset/output/"+tag+"config.obj"
    confs= utils.loadObjsIfExist(confFile)
    if not confs:
        confs = {}
    labels = confs.get("labels", {})
    labelcnt=0
    #3. generate batchs, 200 items each. 
    # too many images, so we should not load them all at once, we need to load needed images as we training.
    batchs=[]
    bacthcnt=0
    objbatch=[]
    objcnt=0
    for pic in annoObj:
        tree = pic.get("infotree")
        root = tree.getroot()
        objects = root.findall('object') # finding all dogs. An array
        for o in objects:
            bndname = o.find('name').text
            imgdir = pic.get("imgdir")
            if(not labels.get(bndname, None)):
                labelcnt+=1
                labels[bndname]=labelcnt
            objcnt+=1
            objbatch.append((bndname, imgdir, o))
            if (objcnt%8192==0 or objcnt == len(annoObj)):
                batchfile="./dogdataset/output/objbatch_"+tag+str(bacthcnt)
                batchs.append(batchfile)
                utils.saveObj(objbatch, batchfile)
                print(objcnt, bacthcnt, len(objbatch))
                objbatch=[]
                bacthcnt+=1
    #3. save confs
    confs["objcnt"] = objcnt
    confs["batchs"] = batchs
    confs["bacthcnt"] = bacthcnt
    confs["labels"] = labels
    confs["labelcnt"] = len(labels.keys())
    print(confs)
    utils.saveObj(confs, confFile)
    pass
    
#============================Images process=============================
def readPic(picdir, obj):
    bndbox = obj.find('bndbox')
    bndname = obj.find('name').text
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    #1. find picture
    img = Image.open(picdir)
    #imgobj= img.resize(IMAGE_SIZE)
    #2. find the object in picure.
    imgobj = img.crop((xmin, ymin, xmax, ymax))           
    #3. resize into fixed size
    imgobj= imgobj.resize(IMAGE_SIZE)
    #4. normalnize the data between 0,1
    imgobj = np.array(imgobj)/ 255.0
    if (imgobj.ndim < 3) or (imgobj.shape != IMAGE_SHAPE):
        #exception: this picture is not a RGB picture. 
        return None
    #5. split data set into different batch
    return bndname, imgobj


def loadImagesBatch(batchfile):
    #1.load images as numpy's ndarray
    objbatch =utils.loadObjsIfExist(batchfile)
    imgarray = []
    labels = []
    for (bndname, imgdir, o) in objbatch:
        imginfo = readPic(imgdir, o)
        if not imginfo: 
            continue # in case picture is wrong.
        bndname, imgobj = imginfo
        imgarray.append(imgobj)
        labels.append(LABELS[bndname])
    imgarray = np.array(imgarray)
    labels=np.array(labels)
    return imgarray,labels

def loadConf(tag):
    confFile ="./dogdataset/output/"+tag+"config.obj"
    confs= utils.loadObjsIfExist(confFile)
    return confs

#============================Tensorflow=================================
def buildModel():
    #1. build model.
    #2. compile the model
    tfmodel = tf.keras.models.Sequential([
    #keras.layers.Flatten(input_shape=IMAGE_SHAPE),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=IMAGE_SHAPE),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #keras.layers.Dropout(0.02), 
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.keras.activations.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(1024, activation=tf.keras.activations.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(len(LABELS), activation=tf.keras.activations.softmax)
    ])
    tfmodel.compile(optimizer=tf.keras.optimizers.Adadelta(lr=0.1),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return tfmodel

def loadWeights(tfmodel):
    checkpoint_path="./dogdataset/output/chk/cp-{epoch:04d}.ckpt"
    if os.path.exists("./dogdataset/output/chk/checkpoint"):
        latest = tf.train.latest_checkpoint("./dogdataset/output/chk/")
        #tfmodel.load_weights(checkpoint_path.format(epoch = 0))
        tfmodel.load_weights(latest)
        print("tfmodel loaded from: ",latest)
    return tfmodel

def trainModel(tfmodel, train_images, train_labels, epochs=50):
    checkpoint_path="./dogdataset/output/chk/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=10)
    result = tfmodel.fit(train_images, train_labels,
          epochs = epochs, callbacks = [cp_callback],
          #validation_data = (test_images,test_labels),
          verbose=1)
    # return the fit history.
    return result, tfmodel
    
def evaluteAccuracy(tfmodel, test_images, test_labels):
    loss, acc = tfmodel.evaluate(test_images, test_labels, verbose=1)
    return loss, acc

#============================Main logic============================

def training():
    #1. load model
    tfmodel = buildModel()
    tfmodel = loadWeights(tfmodel)
    tfmodel.summary()
    #2. load config
    trainconf = loadConf("train")
    testconf = loadConf("test")
    trainBatchs = trainconf["batchs"]
    testBatchs = testconf["batchs"]
    #3. train model with each batch
    history =[]
    nBatch =0
    for batchfile in trainBatchs:
        imgarray,label = loadImagesBatch(batchfile)
        print(imgarray.shape, label.shape)
        h1,tfmodel = trainModel(tfmodel,imgarray,label,epochs=40)
        #4. evaluate the model with random test batch
        testbatch = np.random.choice(testBatchs)
        test_images,test_labels = loadImagesBatch(testbatch)
        print(test_images.shape, test_labels.shape)
        loss, acc = evaluteAccuracy(tfmodel, test_images,test_labels)
        #5. record the training history for further analysis.
        history.append((h1, loss, acc))
        nBatch+=1
        print(nBatch, "loop of training finished, loss and acc is:",loss, acc)

def evaluateResult():
     #1. load model
    tfmodel = buildModel()
    tfmodel = loadWeights(tfmodel)
    tfmodel.summary()
    #2. load config
    testconf = loadConf("test")
    testBatchs = testconf["batchs"]
    history =[]
    nBatch =0
    #4. evaluate the model with random test batch
    testbatch = np.random.choice(testBatchs)
    test_images,test_labels = loadImagesBatch(testbatch)
    print(test_images.shape, test_labels.shape)
    loss, acc = evaluteAccuracy(tfmodel, test_images,test_labels)
    print("loss and acc is:",loss, acc)

def predict(imgdir):
    tfmodel = buildModel()
    tfmodel = loadWeights(tfmodel)
    tfmodel.summary()
    test_images =[]
    figplot = plt.figure(figsize=(10, 10))
    imgs = os.listdir(imgdir)
    for imgpath in imgs:
        #1. Read data. 3. Make prediction.
        img = Image.open(imgdir+ imgpath)
        #2. resize into fixed size
        imgobj= img.resize(IMAGE_SIZE)
        #3. normalnize the data between 0,1
        imgobj = np.array(imgobj)/ 255.0
        test_images.append(imgobj)
    test_images = np.array(test_images)
    predictions = tfmodel.predict(test_images,verbose=1)
    for i in range(len(predictions)):
        img = test_images[i]
        pre = predictions[i]
        mostlikely = np.argmax(pre)
        txt = LABELS_R[mostlikely]
        plt.subplot(331 + i) # showing 9 random images
        plt.imshow(img) # displays photo
        plt.text(0, 0, txt, bbox={'ec': None}) # printing breed
    figplot.savefig("./dogdataset/output/001.png",format='png')
    pass


#============================Main interfaces============================
HELP='''Another dogo recogenition tool:\n\nOPTIONs:
 -h Show this help
 -p prepare the data.
 -t train the model.
 -e <imagepath> to show the prediction result.
 --test simply test the files.
 --clean to clean the workspace under output/*'''
def main():
    argc=len(sys.argv)
    if(argc<2):
        print(HELP)
    elif (argc>=2 and sys.argv[1]== '-h'):
        print(HELP)
    elif (argc>=2 and sys.argv[1]== '-p'):
        annoObjTrain, annoObjTest = loadRawData()
        preProcessBatch("train", annoObjTrain)
        preProcessBatch("test", annoObjTest)
    elif (argc>=2 and sys.argv[1]== '-t'):
        training()
    elif (argc>=3 and sys.argv[1]== '-e'):
        predict(sys.argv[2])
    elif (argc>=2 and sys.argv[1]== '--test'):
        evaluateResult()
        #testReadSamples()
    elif (argc>=2 and sys.argv[1]== '--clean'):
        cleanWorkspace()
    else:
        print(HELP)
    pass

if __name__ == '__main__':
    main()
