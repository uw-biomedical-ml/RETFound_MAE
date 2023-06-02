# RETFound_MAE -- Official Keras Implementation

## Release notes

Keras implementation of [RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE) by Yukun Zhou.

Please contact 	**yk73@uw.edu** if you have questions.

## Installation

Create enviroment with conda:

```
conda create -n retfound_keras python=3.9
conda activate retfound_keras
```

Install Tensorflow 2.8.3 (cuda 11.1)
```
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
python -m pip install tensorflow==2.8.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# The following example lists the number of visible GPUs on the host.
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Install others
```
git clone https://github.com/uw-biomedical-ml/RETFound_MAE
cd RETFound_MAE
python setup.py
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
    --data_path ./IDRiD_data/ \
    --task ./finetune_IDRiD/ \
```

- For evaluation
```
python main_finetune.py
    --data_path ./IDRiD_data/ c
    --nb_classes 5 \ 
    --data_path ./IDRiD_data/ \
    --task ./internal_IDRiD/ \    
    --resume ./finetune_IDRiD/   
```


### Load the model and weights (if you want to call the model in your code)
```
import tfimm
from models_vit_tfimm import *
# call the model
keras_model = tfimm.create_model( # apply global pooling withoug class token
    "vit_large_patch16_224_mae",
    nb_classes = opt.nb_classes
    )
```

If you want to train the classifier with a class token (default is usign global average pooling without this class token), 
```
import tfimm
from models_vit_tfimm import *
# call the model
keras_model = tfimm.create_model( # apply global pooling withoug class token
    "vit_large_patch16_224",
    nb_classes = opt.nb_classes
    )
```
