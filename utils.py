import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 

#utils functions used to plot random image in dataset
def showFolderImageSample(base_folder,img_number=2):
    '''
    base_folder(String) : directory in which we'll look for sample
    img_number : for each folder how many img to show
    '''
    nrows = ncols = img_number
    
    fig = plt.gcf()
    fig.set_size_inches(ncols * img_number , nrows * img_number)
    dir_names =os.listdir(base_folder)
    
    #get sample directory
    sample_dir=[]
    for i in range(img_number):
        sample_dir.append(dir_names[randint(0,61)])
    dir_names = [os.path.join(base_folder,dname) for dname in sample_dir]
    
    print("total sample directory : {}".format(len(dir_names)))
    #for each sample directory get img_number random image
    
    img =[]
    for d in dir_names :
        for i in range(img_number):
            img.append(os.path.join(d,os.listdir(d)[randint(0,len(os.listdir(d)))]))
    
    for i,img_path in enumerate(img):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)
            
        img = mpimg.imread(img_path)
        plt.imshow(img)
            
    plt.show()
    print("Size of one random image : {}".format(img.shape))