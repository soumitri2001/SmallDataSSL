# Exploring Self-Supervised Representation Learning For Low-Resource Medical Image Analysis, ICIP 2023
**Soumitri Chattopadhyay**, Soham Ganguly*, Sreejit Chaudhury*, Sayan Nag*, Samiran Chattopadhyay, [**Exploring Self-Supervised Learning for Low-Resource Medical Image Analysis**](https://arxiv.org/abs/2303.02245), _IEEE International Conference on Image Processing_ **(ICIP)**, 2023. **(Accepted)**

(* denotes equal contribution)

## Abstract
The success of self-supervised learning (SSL) has mostly been attributed to the availability of unlabeled yet large-scale datasets. However, in a specialized domain such as medical imaging which is a lot different from natural images, the assumption of data availability is unrealistic and impractical, as the data itself is scanty and found in small databases, collected for specific prognosis tasks. To this end, we seek to investigate the applicability of self-supervised learning algorithms on small-scale medical imaging datasets. In particular, we evaluate 4 state-of-the-art SSL methods on three publicly accessible small medical imaging datasets. Our investigation reveals that in-domain low-resource SSL pre-training can yield competitive performance to transfer learning from large-scale datasets (such as ImageNet). Furthermore, we extensively analyse our empirical findings to provide valuable insights that can motivate for further research towards circumventing the need for pre-training on a large image corpus. To the best of our knowledge, this is the first attempt to holistically explore self-supervision on low-resource medical datasets.

## Running codes

### Clone repository
``` git clone https://github.com/soumitri2001/SmallDataSSL.git ```

``` cd SmallDataSSL ```
### Install dependencies
``` pip install -r requirements.txt ```
### Datasets
The datasets used in this paper can be found at: [BreaKHis400X](https://www.kaggle.com/datasets/forderation/breakhis-400x) | [Colorectal](https://zenodo.org/record/53169) | [PneumoniaCXR](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Download, unzip and place the datasets as-is under ```./data/``` directory. The dataset utility functions in ```dataset.py``` have been written keeping in mind their original structures. 

Note: Be sure to change the default value of ```--data_root``` and ```--out_root``` in main.py and linear.py with your own paths before running!
### Self-supervised pre-training
Our code supports 4 self-supervised learning methods: SimCLR, DCLW, SimSiam and VICReg (refer to [paper](https://arxiv.org/abs/2303.02245) for more details). These are available as argument choices for ```--model```, along with other training hyperparameters such as ```--batchsize```, ```--lrate```, ```--opti``` and  ```--epochs```. Datasets are available through the argparse token ```--dataset```, along with its color nature ```--dstype``` (RGB/grayscale). 

e.g. to train SimCLR, run the following command:

``` 
python main.py  --model  'SimCLR' --dataset 'breast' --dstype 'color' --epochs 200 --batchsize 64 --opti 'SGD' --lrate 0.01
```
### Linear Evaluation
Using the ```--load_model``` argument the pre-trained encoder weights can be loaded for linear protocol downstream. Hyperparameter arguments such as ```--batchsize```, ```--opti``` and ```--epochs``` are also provided.

``` 
python linear.py --load_model "/path/to/saved_model.pth"
```


## Citation
If you find this article useful in your research, consider citing us:
```
@inproceedings{chattopadhyay2023exploring,
    author = {Soumitri Chattopadhyay and Soham Ganguly and Sreejit Chaudhury and Sayan Nag and Samiran Chattopadhyay},
    title = {Exploring Self-Supervised Representation Learning For Low-Resource Medical Image Analysis},
    booktitle = {IEEE International Conference on Image Processing (ICIP)},
    year = {2023}
}
```
