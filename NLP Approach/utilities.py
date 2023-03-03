import fitz 
import os
import pandas as pd
import tqdm
import nltk
from nltk.corpus import stopwords
import string

def get_titles(list_of_titles):
    list_of_titles=list(set(list_of_titles))
    return list_of_titles

def get_content_from_pdf(path):
    doc=fitz.open(path)
    page=doc[0]
    content=page.get_text()
    translator = str.maketrans('', '', string.punctuation)
    nopunc = content.translate(translator)
    content = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    content=' '.join(content)
    return content

def get_all_doc_info(final_path,vendor):
    print(final_path)
    contents=[]
    title=[]
    image_address=[]
    for file in tqdm.tqdm(os.listdir(final_path)):
        try:
            content=get_content_from_pdf(f'{final_path}/{file}')
            contents.append(content)
            title.append(vendor)
            image_address.append(f'{final_path}/{file}')
        except:
            pass
    return contents,title,image_address

def get_all_vendors_name(path):
    vendors=os.listdir(path)
    return vendors



def prepare_dataset_(path):
    contents=[]
    img_address=[]
    title=[]
    vendors=get_all_vendors_name(path)
    dataset=pd.DataFrame(columns=['Img Address','Text','Title'])
    
    for i in vendors:
        t_c,t_t,t_i=get_all_doc_info(f'{path}/{i}',i)
        contents+=t_c 
        img_address+=t_i
        title+=t_t 
    
    dataset['Img Address']=img_address
    dataset['Text']=contents
    dataset['Title']=title
    dataset.to_csv('dataset_final.csv')
    return dataset


#prepare_dataset_(input('Enter path of dataset: '))