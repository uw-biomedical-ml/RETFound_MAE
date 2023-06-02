# RETFound_MAE (Official Keras Implementation)

## Release notes

Keras implementation of [RETFound_MAE by Yukun Zhou](https://github.com/rmaphoh/RETFound_MAE).

Please contact 	**yk73@uw.edu** if you have questions.

## Installation

Create enviroment with conda:

```
conda create -n retfound_mae python=3.9 -y
conda activate retfound_mae
```

Install Tensorflow 2.8.3 (cuda 11.1)
```
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
python -m pip install tensorflow==2.8.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# To list the number of visible GPUs on the host.
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Install others
```
git clone https://github.com/uw-biomedical-ml/RETFound_MAE
cd RETFound_MAE
pip install -r requirement.txt
```


### Fine-tuning with RETFound weights

- RETFound pre-trained weights

<table>
  <tr>
    <th colspan="2">Download</th>
  </tr>
<tr>
    <td><a href="https://drive.google.com/file/d/194RKGSKZr-zJfeaSpD1QXHqzQvEFkDf-/view?usp=sharing">Colour fundus image</a></td>
    <td><a href="https://drive.google.com/file/d/10Pehch-CndYhcRHjslPd7SOEzbQJAouK/view?usp=sharing">OCT</a></td>
  </tr>
</table>


- Start fine-tuning (use IDRiD as example). A fine-tuned checkpoint for best model on validation loss will be saved during training. 
```
python main_finetune.py
    --data_path ./IDRiD_data/ \
    --nb_classes 5 \ 
    --finetune ./RETFound_cfp_weights.h5 \ 
    --task ./finetune_IDRiD/ \
```

- For evaluation
```
python main_finetune.py
    --data_path ./IDRiD_data/ \
    --nb_classes 5 \ 
    --task ./internal_IDRiD/ \    
    --eval \
    --resume ./finetune_IDRiD/
```


### Load the model
```
import tfimm
from models_vit import *
# call the model
keras_model = tfimm.create_model( # apply global pooling without class token
    "vit_large_patch16_224_mae",
    nb_classes = opt.nb_classes
    )
```

If you want to train the classifier with a class token (default is using global average pooling without this class token), 
```
import tfimm
from models_vit import *
# call the model
keras_model = tfimm.create_model(
    "vit_large_patch16_224",
    nb_classes = opt.nb_classes
    )
```
