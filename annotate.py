import os
import pyautogui
import time

def switch_tab():
    pyautogui.keyDown('alt')
    time.sleep(.002)
    pyautogui.press('tab')
    time.sleep(.002)
    pyautogui.keyUp('alt')

def show_all_files(path):
    count=0
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            file=path+'/'+file
            print(file)
            os.system(file)
            if count>=1:
                time.sleep(0.05)
                switch_tab()
            t_name=input('Enter Template Number: ')
            t_name='_T'+t_name+'.pdf'
            os.rename(file,file.split('.')[0]+t_name)
            
            count+=1
    os.rename(path,path+'_Complete')
    print(f'renamed {count} files')
show_all_files(input('Enter Path of the file: '))