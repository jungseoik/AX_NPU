<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/model_zoo.html (원본) -->

# Mobilint Model Zoo

Mobilint's Model Zoo (`mblt-model-zoo`) is a curated collection of deep learning models optimized by Mobilint’s Neural Processing Units (NPUs).

Designed to help developers accelerate deployment, Mobilint's Model Zoo offers access to **over 490** public, pre-trained, and pre-quantized **vision, language, and multimodal models** in `.mxq`* formats, optimized for Mobilint’s runtime. Along with performance results, we provide pre- and post-processing tools to help developers evaluate, fine-tune, and integrate the models with ease.

New models are added regularly to support evolving AI workloads and customer requests.

```{seealso}
To download and access the Model Zoo, [**click here**](https://github.com/mobilint/mblt-model-zoo).
```

```{tip}
To check which MXQ files are compatible with your runtime environment, refer to the [MXQ compatibility matrix](compatibility.md). 
```

## Supported Models

The following is a list of models (non-exhaustive) currently available in Mobilint's Model Zoo.

### Transformer Models

You can find `transformers` models available in **mblt-model-zoo** in our [HuggingFace Group Page](https://huggingface.co/mobilint).

### Vision Models

#### Image Classification (ImageNet)
| Model                 | Original Source                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| AlexNet               | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html)             |
| ConvNeXt\_Tiny        | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)       |
| ConvNeXt\_Small       | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_small.html)      |
| ConvNeXt\_Base        | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html)       |
| ConvNeXt\_Large       | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_large.html)      |
| DenseNet121           | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html)         |
| DenseNet169           | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet169.html)         |
| DenseNet201           | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html)         |
| GoogLeNet             | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html)           |
| Inception\_V3         | [Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html)          |
| MNASNet1\_0           | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_0.html)         |
| MNASNet1\_3           | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mnasnet1_3.html)         |
| MobileNet\_V2         | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.mobilenet_v2.html)       |
| RegNet\_X\_400MF      | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_400mf.html)     |
| RegNet\_X\_800MF      | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_800mf.html)     |
| RegNet\_X\_1\_6GF     | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_1_6gf.html)     |
| RegNet\_X\_3\_2GF     | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_x_3_2gf.html)        |
| RegNet\_X\_8GF        | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_8gf.html)       |
| RegNet\_X\_16GF       | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_16gf.html)      |
| RegNet\_X\_32GF       | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_x_32gf.html)      |
| RegNet\_Y\_400MF      | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_400mf.html)      |
| RegNet\_Y\_800MF      | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_800mf.html)      |
| RegNet\_Y\_1\_6GF     | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_1_6gf.html)     |
| RegNet\_Y\_3\_2GF     | [Link](https://docs.pytorch.org//vision/2.0/models/generated/torchvision.models.regnet_y_3_2gf.html)        |
| RegNet\_Y\_8GF        | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_8gf.html)       |
| RegNet\_Y\_16GF       | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_16gf.html)      |
| RegNet\_Y\_32GF       | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.regnet_y_32gf.html)      |
| ResNet18              | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet18.html)           |
| ResNet34              | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet34.html)           |
| ResNet50              | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet50.html)           |
| ResNet101             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet101.html)          |
| ResNet152             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnet152.html)          |
| ResNeXt50\_32X4D      | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext50_32x4d.html)    |
| ResNeXt101\_32X8D     | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_32x8d.html)   |
| ResNeXt101\_64X4D     | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.resnext101_64x4d.html)   |
| ShuffleNet\_V2\_X1\_0 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) |
| ShuffleNet\_V2\_X1\_5 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_5.html) |
| ShuffleNet\_V2\_X2\_0 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.shufflenet_v2_x2_0.html) |
| VGG11                 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11.html)              |
| VGG11\_BN             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg11_bn.html)           |
| VGG13                 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13.html)              |
| VGG13\_BN             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg13_bn.html)           |
| VGG16                 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16.html)              |
| VGG16\_BN             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg16_bn.html)           |
| VGG19                 | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19.html)              |
| VGG19\_BN             | [Link](https://docs.pytorch.org//vision/stable/models/generated/torchvision.models.vgg19_bn.html)           |
| Wide\_ResNet50\_2     | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet50_2.html)     |
| Wide\_ResNet101\_2    | [Link](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.wide_resnet101_2.html)    |



#### Object Detection (COCO)
| Model       | Original Source                                              |
| ----------- | --------------------------------------------------- |
| yolov3u     | [Link](https://docs.ultralytics.com/models/yolov3/) |
| yolov3-spp  | [Link](https://github.com/ultralytics/yolov3/)      |
| yolov3-sppu | [Link](https://docs.ultralytics.com/models/yolov3/) |
| yolov5su    | [Link](https://docs.ultralytics.com/models/yolov5/) |
| yolov5mu    | [Link](https://docs.ultralytics.com/models/yolov5/) |
| yolov5lu    | [Link](https://docs.ultralytics.com/models/yolov5/) |
| yolov5l6    | [Link](https://github.com/ultralytics/yolov5/)      |
| yolov5xu    | [Link](https://docs.ultralytics.com/models/yolov5/) |
| yolov5x6    | [Link](https://github.com/ultralytics/yolov5/)      |
| yolov7      | [Link](https://github.com/WongKinYiu/yolov7/)       |
| yolov7x     | [Link](https://github.com/WongKinYiu/yolov7/)       |
| yolov8s     | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8m     | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8l     | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8x     | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov9m     | [Link](https://github.com/WongKinYiu/yolov9/)       |
| yolov9c     | [Link](https://github.com/WongKinYiu/yolov9/)       |
| yolo11s     | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11m     | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11l     | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11x     | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo12s     | [Link](https://docs.ultralytics.com/models/yolo12/) |
| yolo12m     | [Link](https://docs.ultralytics.com/models/yolo12/) |


#### Instance Segmentation (COCO)
| Model       | Original Source                                              |
| ----------- | --------------------------------------------------- |
| yolov5l-seg | [Link](https://github.com/ultralytics/yolov5/)      |
| yolov5x-seg | [Link](https://github.com/ultralytics/yolov5/)      |
| yolov8s-seg | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8m-seg | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8l-seg | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov8x-seg | [Link](https://docs.ultralytics.com/models/yolov8/) |
| yolov9c-seg | [Link](https://docs.ultralytics.com/models/yolov9/) |
| yolo11s-seg | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11m-seg | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11l-seg | [Link](https://docs.ultralytics.com/models/yolo11/) |
| yolo11x-seg | [Link](https://docs.ultralytics.com/models/yolo11/) |
