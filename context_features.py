import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, KFold
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from numpy import asarray
import math 
import getopt, sys
from scipy.io import loadmat

sys.path.append('./CLIP') 


target_col = 'class_label'  # Default column with labels
input_resolution = 224    # Default input resolution


# %%
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
def to_rgb(image):
    return image.convert("RGB")


# General transformation applied to all models
preprocess_image = Compose(
    [
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        to_rgb,
        ToTensor(),
    ]
)


# %%
def torch_hub_normalization():
    # Normalization for torch hub vision models
    return Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


# %%
def clip_normalization():
    # SRC https://github.com/openai/CLIP/blob/e5347713f46ab8121aa81e610a68ea1d263b91b7/clip/clip.py#L73
    return Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


# %%
# Definel "classic" models from torch-hub
def load_torch_hub_model(model_name):
    # Load model
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           model_name, pretrained=True)

    # Put model in 'eval' mode and sent to device
    model = model.eval().to(device)

    # Check for features network
    if hasattr(model, 'features'):
        features = model.features
    else:
        features = model

    return features, torch_hub_normalization()


def load_mobilenet():
    return load_torch_hub_model('mobilenet_v2')


def load_densenet():
    return load_torch_hub_model('densenet121')


def load_resnet():
    return load_torch_hub_model('resnet101')


def load_resnext():
    return load_torch_hub_model('resnext101_32x8d')


def load_vgg():
    return load_torch_hub_model('vgg16')


# %%
# Define CLIP models (ViT-B and RN50)
def load_clip_vit_b():
    model, _ = clip.load("ViT-B/32", device=device)

    return model.encode_image, clip_normalization()


def load_clip_rn50():
    model, _ = clip.load("RN50", device=device)

    return model.encode_image, clip_normalization()


# %%
# Dataset loader
class ImagesDataset(Dataset):
    def __init__(self, df, preprocess, input_resolution):
        super().__init__()
        self.df = df
        self.preprocess = preprocess
        self.empty_image = torch.zeros(3, input_resolution, input_resolution)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        try:
            image = self.preprocess(Image.fromarray(row['Filename']))
        except:
            image = self.empty_image

        return image, row[target_col]


# %%
# Define model loaders
MODELS_LOADERS = {
     'mobilenet': load_mobilenet,
     'densenet': load_densenet,
     'resnet': load_resnet,
     'resnext': load_resnext,
     'vgg': load_vgg,
     'clip_vit_b': load_clip_vit_b,
     'clip_rn50': load_clip_rn50
}




# %% [markdown]
# ## Patches generation

def generate_patchs(im, N, s):
    
    W = input_resolution  #window width
    H = input_resolution  #window height
    
    s = int(s * W)   # Sliding window stride
    
    tam = N * W
    width, height = im.size #image original dimensions size

    #calculing the new size
    if width < height:
        mult = tam/width
        new_w = tam
        new_h = math.ceil(height * mult)
    else:
        mult = tam/height
        new_w = math.ceil(width * mult)
        new_h = tam

    orig_im = asarray(im)

    all_tiles = []
    
    #resize image
    transform = T.Resize([int(new_h), int(new_w)], interpolation=Image.BICUBIC)
    im = transform(im)
    im = asarray(im)

    #patches generation
    tiles = [im[x:x+W,y:y+H] for x in range(0,im.shape[0],W) for y in range(0,im.shape[1],H)]
    
    all_tiles = all_tiles + tiles

    for strt in range(s,W,s):
        tiles = [im[x:x+W,y:y+H] for x in range(strt,im.shape[0],W) for y in range(0,im.shape[1],H)]
        all_tiles = all_tiles + tiles

    for strt in range(s,W,s):
        tiles = [im[x:x+W,y:y+H] for x in range(0,im.shape[0],W) for y in range(strt,im.shape[1],H)]
        all_tiles = all_tiles + tiles

    #drop unsized patches
    final_tiles = []
    for img in all_tiles:
        im_test = Image.fromarray(img)
        width, height = im_test.size
        if width * height == W * H:
            final_tiles.append(img)
    
    print("Num Patchs:")
    print(len(final_tiles))

    if len(final_tiles) == 1:
        tile = []
        tile.append(orig_im)
        return tile
    else:
        return final_tiles


#open image and send it to patch split
def df_patchs(row,columns, N, s):
    df2 = pd.DataFrame(data=None, columns=columns)
    try:
        image = Image.open(row['Filename'])
    except ValueError:
        image = torch.zeros(3, input_resolution, input_resolution)
        transf = T.ToPILImage()
        image = transf(image)

    patchs = generate_patchs(image, N, s)
    for patch in patchs:
        row['Filename'] = patch
        df2 = df2.append(row,ignore_index=True)

    len_patchs = len(patchs)


    return df2, len_patchs    






# %%
# Main function to generate features as describe in paper code
def generate_features(model_loader,N, s, df_all):
    
    # Create model and image normalization
    model, image_normalization = model_loader()
    patchs = []
    preprocess = Compose([preprocess_image, image_normalization])
    
    # Sample one output from model just to check output_dim
    x = torch.zeros(1, 3, input_resolution, input_resolution, device=device)
    with torch.no_grad():
        x_out = model(x)
    output_dim = x_out.shape[1]

    #Feature vector
    V_X = np.empty((len(df_all), output_dim), dtype=np.float32)
    V_y = np.empty(len(df_all), dtype=np.int32)

    j = 0

    #Process each image to create the context features vector
    for index in range(len(df_all)):
        print(f'Image ({index}/{len(df_all)})')
        row = df_all.iloc[index]

        new_df, len_patch = df_patchs(row, df_all.columns, N, s) #function to split image into patches
        patchs.append(len_patch)
 
    
        ds = ImagesDataset(new_df, preprocess, input_resolution)
        dl = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=True)

        # Features data
        X = np.empty((len(ds), output_dim), dtype=np.float32)

        # Begin feature generation
        i = 0
        for images, cls in tqdm(dl):
            
            n_batch = len(images)

            with torch.no_grad():
                emb_images = model(images.to(device))
                if emb_images.ndim == 4:
                    emb_images = emb_images.reshape(
                        n_batch, output_dim, -1).mean(-1)
                emb_images = emb_images.cpu().float().numpy()

            # Save normalized features
            X[i:i+n_batch] = emb_images / \
                np.linalg.norm(emb_images, axis=1, keepdims=True)
            y = cls

            i += n_batch
        

        y = int(y[0])

        #Patches features agregation by average
        sup_vector = np.median(X,axis=0)
            

        V_X[j] = sup_vector
        V_y[j] = y
        j += 1

        del ds, dl, sup_vector, X
    
    del model, image_normalization,

    return V_X, V_y

def sub(x):
	return x - 1

def load_stanford_dataset():
    annots = loadmat('stanford/cars_train_annos.mat')

    annotations = annots['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = 'stanford/cars_train/' + str(annotation[0][5][0])
        car_class = annotation[0][4][0]
        bboxes.append((fname,bbox_x1, bbox_x2, bbox_y1, bbox_y2, int(list(car_class)[0])))
        
        
    df_train = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','class_label'])

    annots = loadmat('stanford/cars_test_annos_withlabels_eval.mat')

    annotations = annots['annotations']
    annotations = np.transpose(annotations)

    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = 'stanford/cars_test/' + str(annotation[0][5][0])
        car_class = annotation[0][4][0]
        bboxes.append((fname,bbox_x1, bbox_x2, bbox_y1, bbox_y2, int(list(car_class)[0])))
        
        
    df_test = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','class_label'])

    df_all = pd.concat([df_train,df_test])

    LABELS_MAP_name = loadmat('stanford/cars_annos.mat')

    ann = LABELS_MAP_name['class_names']
    ann = np.transpose(ann)

    LABELS_MAP = []

    for name in ann:
        LABELS_MAP.append(name[0][0])

    

    df_all['class_label'] = df_all['class_label'].apply(sub)
    print(df_all)
    return df_all, LABELS_MAP


def classifier(X, y, out_name, df_all, LABELS_MAP):
    shape = X.shape
    shape = shape[1]

    # Classifier definition
    class BaseNeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(shape, len(LABELS_MAP))     
            )

        def forward(self, x):
            return self.network(x)
    
    # Define folds
    folds = list(KFold(5, shuffle=True, random_state=42).split(df_all))
    with open("folds","wb") as fl:
        pickle.dump(folds, fl)

    EPOCHS = 100

    print("Epochs", EPOCHS)

    prediction_df = []
    
    print("Features shape: ", X.shape)

    print("Context Features generated sucessfully")
    print("Starting the Classifier train/test")
    
    final_matrix = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)

    final_test = []
    final_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        
        #Spliting train and validation sets
        np.random.shuffle(train_idx)
        val = int(len(train_idx) * 0.9)
        training, valid = train_idx[:val], train_idx[val:]

        train_X = X[training]
        train_y = y[training]

        valid_X = X[valid]
        valid_y = y[valid]

        test_X = X[test_idx]
        test_y = y[test_idx]

        test_df = df_all.iloc[test_idx]

        print("Train X shape", train_X.shape)
        print("Train y shape", train_y.shape)
        print("Valid X shape", valid_X.shape)
        print("Valid y shape", valid_y.shape)
        print("Test X shape", test_X.shape)
        print("Test y shape", test_y.shape)

        fold_model = BaseNeuralNetwork()
        fold_model.to(torch_device)

        print(fold_model)
        print()


        #Training criterion/optimizer/parameters
        fold_criterion = nn.CrossEntropyLoss()
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)
        learning_c_train = []
        learning_c_valid = []
        last_loss = 100
        patience = 5
        trigger_times = 0
        min_improvement = 0.001

        print("Loss", fold_criterion)
        print("Optimizar", fold_optimizer)
        print()

        print("Start training ...")

        for epoch in range(EPOCHS):  # loop over the dataset multiple times

            running_loss = 0.0

            train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=True)

            valid_loader = DataLoader(list(zip(valid_X, valid_y)), batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=True)

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                
                inputs, label_index = data

                multilabel_values = np.zeros((len(label_index),len(LABELS_MAP))).astype(float)
                #print(multilabel_values.shape)

                for k, idx in enumerate(label_index):
                    
                    multilabel_values[k][idx] = 1.0


                tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                # zero the parameter gradients
                fold_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = fold_model(inputs.to(torch_device))
                pred = outputs.cpu().argmax()

                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())

                fold_loss.backward()
                fold_optimizer.step()

                # print statistics
                running_loss += fold_loss.item()
                
                if i == len(train_loader) - 1:
                    print('[%d, %5d] Train loss: %.5f' %
                        (epoch + 1, i + 1, running_loss / len(train_loader)))
                    learning_c_train.append(running_loss / len(train_loader))
                    running_loss = 0.0
            
            #Validation
            valid_loss = 0.0
            fold_model.eval() 
            for i, data in enumerate(valid_loader, 0):

                inputs, label_index = data
                multilabel_values = np.zeros((len(label_index),len(LABELS_MAP))).astype(float)

                for k, idx in enumerate(label_index):
                    multilabel_values[k][idx] = 1.0


                tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                fold_optimizer.zero_grad()

                outputs = fold_model(inputs.to(torch_device))
                pred = outputs.cpu().argmax()

                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                valid_loss += fold_loss.item()
                current_loss = valid_loss / len(valid_loader)
                
                #print statistics
                if i == len(valid_loader) - 1:   
                    print('[%d, %5d] Valid loss: %.5f' %
                        (epoch + 1, i + 1, valid_loss / len(valid_loader)))
                    valid_loss = 0.0

            
            #Early stopping verification
            learning_c_valid.append(current_loss)
            minimal = last_loss - (last_loss * min_improvement)
            print("Minimal:")
            print(minimal)

            if current_loss > minimal:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break

            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss


        #Plot the learning curve before the test
        plt.plot(np.array(learning_c_valid), 'r', label = "valid loss")
        plt.plot(np.array(learning_c_train), 'b', label = "train loss")
        plt.legend()
        plt.savefig(str(fold_idx) + out_name + "_lc.jpg")
        plt.clf()

        corrects = 0
        fold_cm = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)
        y_pred = []
        test_predictions = []

        print("Start testing ...")

        for x_item, y_item in list(zip(test_X, test_y)):

            item_input = torch.from_numpy(x_item).to(torch_device)

            preds = fold_model(item_input)

            pred_index = preds.cpu().argmax()

            fold_cm[y_item][pred_index] += 1

            if pred_index == y_item:
                corrects += 1

            y_pred.append(pred_index)
            test_predictions.append(preds.detach().cpu().numpy().tolist())

        
        #Calculatin the metrics
        y_pred = np.array(y_pred)
        accuracy_score = corrects / len(test_y)

        fold_predictions = list(zip(test_df['Filename'].values, test_df['class_label'].values, test_predictions))
        fold_predictions = [[p[0], p[1], *p[2]] for p in fold_predictions]

        prediction_df += fold_predictions        
        
        print(f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.5f}")
        print('Finished fold training')

        final_matrix = np.add(final_matrix, fold_cm)
        final_test = np.concatenate((final_test, test_y))
        final_pred = np.concatenate((final_pred, y_pred)) 

        print("Raw matrix:")
        print(fold_cm.tolist())
        print()

        print("Fold Classification Report:")
        print(metrics.classification_report(test_y, y_pred, target_names=LABELS_MAP))
        print()

        #Saving the classifier model
        torch.save(fold_model.state_dict(), f"{out_name}-fold-{fold_idx+1}.model")

    print("Final Result")
    print("Final Raw matrix:")
    print(final_matrix.tolist())
    print()

    print("Final Classification Report:")
    cr = metrics.classification_report(final_test, final_pred, target_names=LABELS_MAP,output_dict=True)
    print(cr)
    df_report = pd.DataFrame(cr).transpose()

    df_report.to_csv(out_name + '_results.csv')

def context_features(Nlist, FElist, s):

    # Function to load stanford cars dataset
    # If you want to load another dataset you need to keep this format:
    # - df_all: pandas dataframe with a column 'Filename' with the image file path and a column 'class_labels' with the label value of each image
    # - LABELS_MAP: list with the name of all classes in the dataset (according to dataframe labels)
    df_all, LABELS_MAP = load_stanford_dataset()
    
    n = len(FElist)
    for i, (model_name) in enumerate(FElist, 1):
        model_loader = MODELS_LOADERS[model_name]
        
        print(f'[{i}/{n}] Evaluating on {model_name}...')
        
        out_name = str(model_name)+'_s' + str(s) + '_N' + ('_'.join(str(n) for n in Nlist))
        
        features_list = [] 
        
        if Path(f"X_features_{out_name}.npy").is_file():
            with open(f"X_features_{out_name}.npy",'rb') as fe:      
                X = np.load(fe)
            features_list.append(X)

            with open(f"y_labels_{out_name}.npy",'rb') as f:      
                y = np.load(f)
        else:
            for index in range(len(Nlist)):         
                X, y = generate_features(model_loader, int(Nlist[index]), s, df_all)
                features_list.append(X)

            if len(features_list) > 1:
                X = np.concatenate((features_list), axis = 1)
            else:
                X = features_list[0]
    
            #Saving features
            np.save(f"X_features_{out_name}.npy", X)
            np.save(f"y_labels_{out_name}.npy", y)

        
        
        classifier(X, y, out_name, df_all, LABELS_MAP)



def main():
    argumentList = sys.argv[1:]
    # Options
    options = "hN:s:"
    # Long options
    long_options = ["Help", "Output=", "FE="]
    out_name = ''
    s = 0
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--help"):
                print ("Multiscale context features\n")
                print ('Usage:')
                print('python context.py -h | --help')
                print("python context.py -N <approach_parameter> --FE <model_name> [-s <stride_value>]")
                print('\nOptions:')
                print("-h --help    Show this screen")
                print("-N           Approach parameter (If you want use multiscale you can pass a list of N values separed by a comma, ex: 1,2)")
                print("--FE         Feature extractor model name [models available below] (If you want apply multiple models you can pass a list of names separed by a comma, ex: densenet,resnext)")
                print("-s           Slinding window stride value. Range: (0,1) [default: 0.01]")
                print("\nAvailable feature extractors (call by the name in the right): ")
                print("MobileNet_v2:   mobilenet")
                print("DenseNet_121:   densenet")
                print("ResNet_101:     resnet")
                print("ResNeXt_101:    resnext")
                print("Vgg_16:         vgg")
                print("Clip ViT-B/32:  clip_vit_b")
                print("Clip_RN50:      clip_rn50")
                print("\nMore details of the approach and the implementation:")
                print("LINKS")
                sys.exit(2)

            elif currentArgument in ("-N"):
                if ',' in currentValue:
                    Nlist = currentValue.split(',')
                else:
                    Nlist = [currentValue]

            elif currentArgument in ("--FE"):
                if ',' in currentValue:
                    FElist = currentValue.split(',')
                else:
                    FElist = [currentValue]

            elif currentArgument in ("-s"):
                s = float(currentValue)
                
            
        if s == 0:
            s = 0.01
        print("N: ", Nlist)
        print('Feature Extractors: ', FElist)
        print("Sliding window stride: ", s)

    except UnboundLocalError:
        print("You forget to define something")
        sys.exit(2)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    context_features(Nlist, FElist, s)
    


if __name__ == "__main__":
   main()
