from shutil import copyfile
from os import walk
import os


def struct_data(src,dest,sub_size):
    try:
        os.mkdir(dest+"/"+"training_set")
        os.mkdir(dest+"/"+"test_set")
        os.mkdir(dest+"/"+"validation_set")
    except:
        print("")
    
    for (dirpath, dirnames, filenames) in walk(src):
        break
    
    for i in range(1,sub_size+1):
        searching = 'S' + str(i).zfill(3)
        try:
            os.mkdir(dest+"/training_set/"+searching)
            os.mkdir(dest+"/test_set/"+searching)
            os.mkdir(dest+"/validation_set/"+searching)
        except:
            print("")
        
        found = False
        counter = 0
        useful_files = []
        for fnames in filenames:
            if(searching in fnames):
                useful_files.append(fnames)
                found = True
                counter = counter + 1
            if(found and not(searching in fnames)):
                break
        
        new_counter = 0
        for fs in useful_files:
            if(new_counter <= int(round(counter*(70/100))) ):
                copyfile(src+"/"+fs, dest+"/training_set/"+searching+"/"+fs)
            elif(new_counter <= int(round(counter*(20/100))) + int(round(counter*(70/100))) ):
                copyfile(src+"/"+fs, dest+"/test_set/"+searching+"/"+fs)
            else:
                copyfile(src+"/"+fs, dest+"/validation_set/"+searching+"/"+fs)
            new_counter = new_counter + 1
        

#Update your path here ...
struct_data("H:/Dataset","H:/CNN_DATA",50)