import fitz
import os
import tqdm
import pandas as pd
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import json
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import cv2
import time

def dump_load_map(map,flag):
    if flag==0:
        with open("output_map.json", "w") as outfile:
            json.dump(map, outfile)
    else: 
        with open("output_map.json", "r") as outfile:
            map=json.load(outfile)
        return map
        

def get_data_image(path):
    doc=fitz.open(path)
    page=doc.load_page(0)
    doc=page.get_pixmap()
    return doc

def inference_image_get(path):
    img=get_data_image(path)
    img.save('t_img.jpg')
    img=Image.open('t_img.jpg')
    img=img.resize((224,224))
    img=np.asarray(img)
    try:
        os.remove('t_img.jpg')
    except:
        pass
    return img

def make_local_storage(vendor_name):
    try:
        os.mkdir('Dataset')
        
    except:
        pass

    try:
        os.mkdir(f'Dataset/{vendor_name}')
    except:
        pass


def prepare_dataset(path,vendor_name,vendor_map):
    images_list=[]
    vendor_list=[]
    make_local_storage(vendor_name)
    for file in tqdm.tqdm(os.listdir(path),desc = 'dirs'):
        if file.endswith('.pdf'):
            image=get_data_image(path+'/'+file)
            file=file.replace('pdf','.jpg')
            image.save(f'Dataset/{vendor_name}/{file}')

            

            images_list.append(f'Dataset/{vendor_name}/{file}')
            vendor_list.append(vendor_map[vendor_name])
    return images_list,vendor_list

def prepare_all_dataset(path_to_dataset):
    df=pd.DataFrame(columns=['image','vendor'])
    images_list=[]
    vendors_list=[]
    vendors=[path_to_dataset+'/'+i for i in os.listdir(path_to_dataset)]
    vendor_map={}
    for i in range(len(vendors)):
        vendor_map[vendors[i].split('/')[-1]]=i
    
    print(f"Here are vendors: {vendors}")
    for i in vendors:
        v=i.split('/')[-1]
        print('\n'+v+' vendor Processing ')
        t_image_list,t_vendor_list=prepare_dataset(i,v,vendor_map)
        images_list+=t_image_list
        vendors_list+=t_vendor_list

    return images_list,vendors_list,vendor_map

def plot_metrics_for_training(history):
    try:
        os.mkdir('History_Graphs')
    except:
        pass

    plt.figure(figsize=(30, 10))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')
    plt.xticks(np.arange(len(history['accuracy'])), np.arange(1, len(history['accuracy'])+1))
    #plt.xticks(np.arange(len(history['val_accuracy'])), np.arange(1, len(history['val_accuracy'])+1))
    plt.xlabel('Epoch')
    plt.ylabel('val_accuracy')
    plt.ylim([0.0, 1.10])
    plt.legend(loc='lower right')
    plt.savefig('History_Graphs/Accuracy_history.jpg')

    plt.figure(figsize=(30, 10))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label = 'val_loss')
    plt.xticks(np.arange(len(history['loss'])), np.arange(1, len(history['loss'])+1))
    #plt.xticks(np.arange(len(history['val_accuracy'])), np.arange(1, len(history['val_accuracy'])+1))
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.ylim([0,3.5])
    plt.legend(loc='lower right')
    plt.savefig('History_Graphs/Loss_History.jpg')



def get_images_array(X):
    images_output=[]
    for i in X:
        img=Image.open(i)
        img=img.resize((224,224))
        img=np.asarray(img)
        images_output.append(img)
    
    return np.asarray(images_output)

def get_class_weights(Y):
    y_integers = np.argmax(Y, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))
    return d_class_weights

def inference_testing_model(model_name,map,path_pdf):
    loaded_model=tf.keras.models.load_model('Efficient_net_custom_doc_type.h5')
    map=dump_load_map({},1)
    #print(f'File Uploaded: {path_pdf}')
    test_img=inference_image_get(path_pdf)
    preds=loaded_model.predict(test_img.reshape(1,test_img.shape[0],test_img.shape[1],test_img.shape[2]))
    op=np.argmax(preds)
    print(list(preds[0])[op])
    if list(preds[0])[op]<=0.79:
        return 'This file belongs to none of the vendors '
    else: 
        pred_class=list(filter(lambda x: map[x] == op, map))[0]
        print(pred_class)
        v_name=pred_class.split('_')[0]
        with open('document_type_mapping.json','r') as default_:
            f=json.load(default_)
        
        d_type=f[pred_class.upper()]
        return f'This file belongs to  \n  Vendor: {v_name}  \n  Doc Type: {d_type}'



def get_data_image_black_bar(path):
    doc=fitz.open(path)
    page=doc.load_page(0)
    img=page.get_pixmap()
    img.save('test.jpg')
    img = Image.open('test.jpg').convert('RGB') 
    img=img.resize((224,224))
    img = np.array(img) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200,200, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.drawContours(img,[cnt],0,(0,0,0),3)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),-1)

    img=Image.fromarray(img)
    return img

if __name__=='__main__':
    #prepare_dataset('../Dataset/Medline','Medicine')
    #op=prepare_all_dataset('Dataset_EDA')
    #print(op)

    start_time = time.time()
    img=get_data_image_black_bar(input('Enter path:'))
    print("--- %s seconds ---" % (time.time() - start_time))
    #img.show()
    


'''
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()

'''