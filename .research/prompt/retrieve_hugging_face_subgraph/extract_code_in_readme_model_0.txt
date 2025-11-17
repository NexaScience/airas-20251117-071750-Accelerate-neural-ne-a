
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
tags:
- image-classification
- timm
- transformers
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
---
# Model card for mobilenetv2_100.ra_in1k

A MobileNet-v2 image classification model. Trained on ImageNet-1k in `timm` using recipe template described below.

Recipe details:
 * RandAugment `RA` recipe. Inspired by and evolved from EfficientNet RandAugment recipes. Published as `B` recipe in [ResNet Strikes Back](https://arxiv.org/abs/2110.00476).
 * RMSProp (TF 1.0 behaviour) optimizer, EMA weight averaging
 * Step (exponential decay w/ staircase) LR schedule with warmup


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 3.5
  - GMACs: 0.3
  - Activations (M): 6.7
  - Image size: 224 x 224
- **Papers:**
  - MobileNetV2: Inverted Residuals and Linear Bottlenecks: https://arxiv.org/abs/1801.04381
  - ResNet strikes back: An improved training procedure in timm: https://arxiv.org/abs/2110.00476
- **Dataset:** ImageNet-1k
- **Original:** https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'mobilenetv2_100.ra_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 16, 112, 112])
    #  torch.Size([1, 24, 56, 56])
    #  torch.Size([1, 32, 28, 28])
    #  torch.Size([1, 96, 14, 14])
    #  torch.Size([1, 320, 7, 7])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'mobilenetv2_100.ra_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1280, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
```bibtex
@inproceedings{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and Jegou, Herve},
  booktitle={NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future}
}
```

Output:
{
    "extracted_code": "from urllib.request import urlopen\nfrom PIL import Image\nimport timm\n\nimg = Image.open(urlopen(\n    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'\n))\n\nmodel = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=True)\nmodel = model.eval()\n\n# get model specific transforms (normalization, resize)\ndata_config = timm.data.resolve_model_data_config(model)\ntransforms = timm.data.create_transform(**data_config, is_training=False)\n\noutput = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1\n\ntop5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)\n\nfrom urllib.request import urlopen\nfrom PIL import Image\nimport timm\n\nimg = Image.open(urlopen(\n    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'\n))\n\nmodel = timm.create_model(\n    'mobilenetv2_100.ra_in1k',\n    pretrained=True,\n    features_only=True,\n)\nmodel = model.eval()\n\n# get model specific transforms (normalization, resize)\ndata_config = timm.data.resolve_model_data_config(model)\ntransforms = timm.data.create_transform(**data_config, is_training=False)\n\noutput = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1\n\nfor o in output:\n    # print shape of each feature map in output\n    # e.g.:\n    #  torch.Size([1, 16, 112, 112])\n    #  torch.Size([1, 24, 56, 56])\n    #  torch.Size([1, 32, 28, 28])\n    #  torch.Size([1, 96, 14, 14])\n    #  torch.Size([1, 320, 7, 7])\n\n    print(o.shape)\n\nfrom urllib.request import urlopen\nfrom PIL import Image\nimport timm\n\nimg = Image.open(urlopen(\n    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'\n))\n\nmodel = timm.create_model(\n    'mobilenetv2_100.ra_in1k',\n    pretrained=True,\n    num_classes=0,  # remove classifier nn.Linear\n)\nmodel = model.eval()\n\n# get model specific transforms (normalization, resize)\ndata_config = timm.data.resolve_model_data_config(model)\ntransforms = timm.data.create_transform(**data_config, is_training=False)\n\noutput = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor\n\n# or equivalently (without needing to set num_classes=0)\n\noutput = model.forward_features(transforms(img).unsqueeze(0))\n# output is unpooled, a (1, 1280, 7, 7) shaped tensor\n\noutput = model.forward_head(output, pre_logits=True)\n# output is a (1, num_features) shaped tensor"
}
