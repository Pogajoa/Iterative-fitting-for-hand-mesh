from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
import json
import sys
import torch
import os
import numpy as np 
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
sys.path.append('/home/cvlab/anaconda3/lib/python3.9/site-packages')
import json
def load_dataset(fp_data='./data/youtube_test_jq.json'):
    """Load the YouTube dataset.
    Args:
        fp_data: Filepath to the json file.
    Returns:
        Hand mesh dataset.
    """    
    with open(fp_data, "r") as file:
        data = json.load(file)

        #dict_example = json.loads(data)
        #data = sorted(data, key = lambda x:int(x['images']['id']))
    return data 

def retrieve_sample(data, ann_index):
    """Retrieve an annotation-image pair from the dataset.
    Args:
        data: Hand mesh dataset.
        ann_index: Annotation index.

    Returns:
        A sample from the hand mesh dataset.
    """    
    ann = data['annotations'][ann_index]
    images = data['images']
    img_idxs = [im['id'] for im in images]
    img = images[img_idxs.index(ann['image_id'])]
    name = img['name'].split('/')[-1]
    return ann, img, name

def viz_sample(data, ann_index, faces=None, db_root='./data/'):
    """Visualize a sample from the dataset.
    Args:
        data: Hand mesh dataset.
        ann_index: Annotation index.
        faces: MANO faces.
        db_root: Filepath to the youtube parent directory.
    """
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from os.path import join
    ann, img, name = retrieve_sample(data, ann_index)
    image = imageio.v2.imread(join(db_root, img['name']))
    vertices = np.array(ann['vertices'])
    
    if faces is None:
        plt.plot(vertices[:, 0], vertices[:, 1], 'D', color='blue', markersize=0.05)
    else:
        import matplotlib.colors
        plt.triplot(vertices[:, 0], vertices[:, 1], faces, lw=0.1)
        c = np.ones(len(vertices))
        # 성공
        # create a colormap with a single color
        cmap = matplotlib.colors.ListedColormap("white")
        # tripcolorplot with a single color filling:
        plt.tripcolor(vertices[:, 0], vertices[:, 1],faces, c, edgecolor="k", lw=0.1, cmap=cmap)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    return plt, name

if __name__ == "__main__":
    data = load_dataset() 
    
    index_list = []   
    for im in data['images']:
        if im['name'].split('/')[1] == 'G26Ly2CTUy0':
            index_list.append(im['id'])
    index_list.sort()
                
    mano_root='./manopth/mano/models'
    side =  'right'
    if side == 'right':
            mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
    elif side == 'left':
            mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
    smpl_data = ready_arguments(mano_path)
    face = torch.Tensor(smpl_data['f'].astype(np.int32)).long()
    images = data['images'] 
    img_name = [im['name'] for im in images]        
    img_idxs = [im['id'] for im in images]
    #list_start = img_idxs.index(2)
    #list_end =  img_idxs.index(1255)
    
    #img_name = img_name[list_start:list_end+1]
    name_list=[]
    
    for index in range(len(img_name)):
        new = img_name[index].split('/')
        name_list.append(new[-1])
        
    img_idxs = [im['id'] for im in images]
    #list_start = img_idxs.index(17702)
    #list_end =  img_idxs.index(18555)
    
    #img_idxs = img_idxs[list_start:list_end+1]
    
    num = 0
    for index in range(len(data['images'])):
        #print(img_idxs)
        for id in index_list:
            if retrieve_sample(data, index)[1]['id'] == id:
                num += 1
                print(index)
                plt, name = viz_sample(data,index,face)
                filename = './test/' + name
                plt.savefig(filename, dpi=500 ,bbox_inches='tight', pad_inches=0)
                
    '''
    # version2
    for index in range(len(data['images'])):
        if data['images'][index]['name'].split('/')[1] == '3mdlb7bViOI':
                num += 1
                plt = viz_sample(data,index,face)
                filename = './test/' + name_list[num]
                #plt.show()
                plt.savefig(filename, dpi=300 ,bbox_inches='tight', pad_inches=0)
    ''' 
    print("Data keys:", [k for k in data.keys()])
    print("Image keys:", [k for k in data['images'][0].keys()])
    print("Annotations keys:", [k for k in data['annotations'][0].keys()])

    print("The number of images:", len(data['images']))
    print("The number of annotations:", len(data['annotations']))