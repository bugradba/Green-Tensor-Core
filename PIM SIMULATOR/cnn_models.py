"""
CNN Model Kütüphanesi

Popüler CNN modellerinin tam katman tanımları.
Bu modeller gerçek PIM/GPU simülasyonunda test edilir.
"""

class CNNModels:
    """Popüler CNN modellerinin katman tanımları"""
    
    # AlexNet (2012) - 8 öğrenilen katman
    ALEXNET = [
        # Convolutional Layers
        {
            'name': 'conv1',
            'type': 'Conv2D',
            'input': (3, 227, 227),
            'kernel': (96, 3, 11, 11),
            'stride': 4,
            'padding': 0
        },
        {
            'name': 'relu1',
            'type': 'ReLU',
            'input': (96, 55, 55)
        },
        {
            'name': 'pool1',
            'type': 'MaxPool',
            'input': (96, 55, 55),
            'kernel_size': 3,
            'stride': 2
        },
        {
            'name': 'conv2',
            'type': 'Conv2D',
            'input': (96, 27, 27),
            'kernel': (256, 96, 5, 5),
            'stride': 1,
            'padding': 2
        },
        {
            'name': 'relu2',
            'type': 'ReLU',
            'input': (256, 27, 27)
        },
        {
            'name': 'pool2',
            'type': 'MaxPool',
            'input': (256, 27, 27),
            'kernel_size': 3,
            'stride': 2
        },
        {
            'name': 'conv3',
            'type': 'Conv2D',
            'input': (256, 13, 13),
            'kernel': (384, 256, 3, 3),
            'stride': 1,
            'padding': 1
        },
        {
            'name': 'relu3',
            'type': 'ReLU',
            'input': (384, 13, 13)
        },
        {
            'name': 'conv4',
            'type': 'Conv2D',
            'input': (384, 13, 13),
            'kernel': (384, 384, 3, 3),
            'stride': 1,
            'padding': 1
        },
        {
            'name': 'relu4',
            'type': 'ReLU',
            'input': (384, 13, 13)
        },
        {
            'name': 'conv5',
            'type': 'Conv2D',
            'input': (384, 13, 13),
            'kernel': (256, 384, 3, 3),
            'stride': 1,
            'padding': 1
        },
        {
            'name': 'relu5',
            'type': 'ReLU',
            'input': (256, 13, 13)
        },
        {
            'name': 'pool5',
            'type': 'MaxPool',
            'input': (256, 13, 13),
            'kernel_size': 3,
            'stride': 2
        },
        
        # Fully Connected Layers
        {
            'name': 'fc1',
            'type': 'Linear',
            'in_features': 9216,   # 256 * 6 * 6
            'out_features': 4096
        },
        {
            'name': 'relu6',
            'type': 'ReLU'
        },
        {
            'name': 'dropout1',
            'type': 'Dropout',
            'p': 0.5
        },
        {
            'name': 'fc2',
            'type': 'Linear',
            'in_features': 4096,
            'out_features': 4096
        },
        {
            'name': 'relu7',
            'type': 'ReLU'
        },
        {
            'name': 'dropout2',
            'type': 'Dropout',
            'p': 0.5
        },
        {
            'name': 'fc3',
            'type': 'Linear',
            'in_features': 4096,
            'out_features': 1000  # ImageNet classes
        }
    ]
    
    # VGG16 (2014) - 16 öğrenilen katman
    VGG16 = [
        # Block 1
        {'name': 'conv1_1', 'type': 'Conv2D', 'input': (3, 224, 224), 'kernel': (64, 3, 3, 3), 'padding': 1},
        {'name': 'relu1_1', 'type': 'ReLU'},
        {'name': 'conv1_2', 'type': 'Conv2D', 'input': (64, 224, 224), 'kernel': (64, 64, 3, 3), 'padding': 1},
        {'name': 'relu1_2', 'type': 'ReLU'},
        {'name': 'pool1', 'type': 'MaxPool', 'kernel_size': 2, 'stride': 2},
        
        # Block 2
        {'name': 'conv2_1', 'type': 'Conv2D', 'input': (64, 112, 112), 'kernel': (128, 64, 3, 3), 'padding': 1},
        {'name': 'relu2_1', 'type': 'ReLU'},
        {'name': 'conv2_2', 'type': 'Conv2D', 'input': (128, 112, 112), 'kernel': (128, 128, 3, 3), 'padding': 1},
        {'name': 'relu2_2', 'type': 'ReLU'},
        {'name': 'pool2', 'type': 'MaxPool', 'kernel_size': 2, 'stride': 2},
        
        # Block 3
        {'name': 'conv3_1', 'type': 'Conv2D', 'input': (128, 56, 56), 'kernel': (256, 128, 3, 3), 'padding': 1},
        {'name': 'relu3_1', 'type': 'ReLU'},
        {'name': 'conv3_2', 'type': 'Conv2D', 'input': (256, 56, 56), 'kernel': (256, 256, 3, 3), 'padding': 1},
        {'name': 'relu3_2', 'type': 'ReLU'},
        {'name': 'conv3_3', 'type': 'Conv2D', 'input': (256, 56, 56), 'kernel': (256, 256, 3, 3), 'padding': 1},
        {'name': 'relu3_3', 'type': 'ReLU'},
        {'name': 'pool3', 'type': 'MaxPool', 'kernel_size': 2, 'stride': 2},
        
        # Block 4
        {'name': 'conv4_1', 'type': 'Conv2D', 'input': (256, 28, 28), 'kernel': (512, 256, 3, 3), 'padding': 1},
        {'name': 'relu4_1', 'type': 'ReLU'},
        {'name': 'conv4_2', 'type': 'Conv2D', 'input': (512, 28, 28), 'kernel': (512, 512, 3, 3), 'padding': 1},
        {'name': 'relu4_2', 'type': 'ReLU'},
        {'name': 'conv4_3', 'type': 'Conv2D', 'input': (512, 28, 28), 'kernel': (512, 512, 3, 3), 'padding': 1},
        {'name': 'relu4_3', 'type': 'ReLU'},
        {'name': 'pool4', 'type': 'MaxPool', 'kernel_size': 2, 'stride': 2},
        
        # Block 5
        {'name': 'conv5_1', 'type': 'Conv2D', 'input': (512, 14, 14), 'kernel': (512, 512, 3, 3), 'padding': 1},
        {'name': 'relu5_1', 'type': 'ReLU'},
        {'name': 'conv5_2', 'type': 'Conv2D', 'input': (512, 14, 14), 'kernel': (512, 512, 3, 3), 'padding': 1},
        {'name': 'relu5_2', 'type': 'ReLU'},
        {'name': 'conv5_3', 'type': 'Conv2D', 'input': (512, 14, 14), 'kernel': (512, 512, 3, 3), 'padding': 1},
        {'name': 'relu5_3', 'type': 'ReLU'},
        {'name': 'pool5', 'type': 'MaxPool', 'kernel_size': 2, 'stride': 2},
        
        # Classifier
        {'name': 'fc1', 'type': 'Linear', 'in_features': 25088, 'out_features': 4096},
        {'name': 'relu6', 'type': 'ReLU'},
        {'name': 'dropout1', 'type': 'Dropout', 'p': 0.5},
        {'name': 'fc2', 'type': 'Linear', 'in_features': 4096, 'out_features': 4096},
        {'name': 'relu7', 'type': 'ReLU'},
        {'name': 'dropout2', 'type': 'Dropout', 'p': 0.5},
        {'name': 'fc3', 'type': 'Linear', 'in_features': 4096, 'out_features': 1000}
    ]
    
    # ResNet-50 (Basitleştirilmiş - ilk birkaç katman)
    RESNET50_SIMPLIFIED = [
        # Initial Conv
        {'name': 'conv1', 'type': 'Conv2D', 'input': (3, 224, 224), 'kernel': (64, 3, 7, 7), 'stride': 2, 'padding': 3},
        {'name': 'bn1', 'type': 'BatchNorm'},
        {'name': 'relu1', 'type': 'ReLU'},
        {'name': 'pool1', 'type': 'MaxPool', 'kernel_size': 3, 'stride': 2, 'padding': 1},
        
        # Layer1 (3 blocks) - Basitleştirilmiş
        {'name': 'layer1_conv1', 'type': 'Conv2D', 'input': (64, 56, 56), 'kernel': (256, 64, 1, 1)},
        {'name': 'layer1_relu', 'type': 'ReLU'},
        
        # Layer2 (4 blocks) - Basitleştirilmiş
        {'name': 'layer2_conv1', 'type': 'Conv2D', 'input': (256, 56, 56), 'kernel': (512, 256, 3, 3), 'stride': 2},
        {'name': 'layer2_relu', 'type': 'ReLU'},
        
        # Layer3 (6 blocks) - Basitleştirilmiş
        {'name': 'layer3_conv1', 'type': 'Conv2D', 'input': (512, 28, 28), 'kernel': (1024, 512, 3, 3), 'stride': 2},
        {'name': 'layer3_relu', 'type': 'ReLU'},
        
        # Layer4 (3 blocks) - Basitleştirilmiş
        {'name': 'layer4_conv1', 'type': 'Conv2D', 'input': (1024, 14, 14), 'kernel': (2048, 1024, 3, 3), 'stride': 2},
        {'name': 'layer4_relu', 'type': 'ReLU'},
        
        # Average Pool + FC
        {'name': 'avgpool', 'type': 'AvgPool'},
        {'name': 'fc', 'type': 'Linear', 'in_features': 2048, 'out_features': 1000}
    ]
    
    # MobileNetV2 (Basitleştirilmiş)
    MOBILENETV2_SIMPLIFIED = [
        # Initial Conv
        {'name': 'conv1', 'type': 'Conv2D', 'input': (3, 224, 224), 'kernel': (32, 3, 3, 3), 'stride': 2},
        {'name': 'bn1', 'type': 'BatchNorm'},
        {'name': 'relu1', 'type': 'ReLU'},
        
        # Inverted Residual Blocks (sadece birkaç örnek)
        {'name': 'block1', 'type': 'Conv2D', 'input': (32, 112, 112), 'kernel': (16, 32, 1, 1)},
        {'name': 'block2', 'type': 'Conv2D', 'input': (16, 112, 112), 'kernel': (96, 16, 1, 1)},
        {'name': 'block3', 'type': 'Conv2D', 'input': (96, 56, 56), 'kernel': (24, 96, 1, 1)},
        
        # Final layers
        {'name': 'conv_last', 'type': 'Conv2D', 'input': (320, 7, 7), 'kernel': (1280, 320, 1, 1)},
        {'name': 'avgpool', 'type': 'AvgPool'},
        {'name': 'fc', 'type': 'Linear', 'in_features': 1280, 'out_features': 1000}
    ]
    
    @classmethod
    def get_model(cls, model_name):
        """
        Model adına göre katman listesini döndür
        
        Args:
            model_name: 'alexnet', 'vgg16', 'resnet50', 'mobilenet'
        
        Returns:
            layers: Katman listesi
        """
        model_map = {
            'alexnet': cls.ALEXNET,
            'vgg16': cls.VGG16,
            'resnet50': cls.RESNET50_SIMPLIFIED,
            'mobilenet': cls.MOBILENETV2_SIMPLIFIED
        }
        
        model_name_lower = model_name.lower()
        if model_name_lower not in model_map:
            raise ValueError(f"Model '{model_name}' bulunamadı. Geçerli modeller: {list(model_map.keys())}")
        
        return model_map[model_name_lower]
    
    @classmethod
    def get_model_info(cls, model_name):
        """Model hakkında bilgi"""
        layers = cls.get_model(model_name)
        
        total_layers = len(layers)
        conv_layers = len([l for l in layers if l['type'] == 'Conv2D'])
        fc_layers = len([l for l in layers if l['type'] == 'Linear'])
        
        return {
            'name': model_name,
            'total_layers': total_layers,
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'other_layers': total_layers - conv_layers - fc_layers
        }
    
    @classmethod
    def list_models(cls):
        """Mevcut tüm modelleri listele"""
        models = ['alexnet', 'vgg16', 'resnet50', 'mobilenet']
        
        print("\n" + "="*70)
        print("MEVCUT CNN MODELLERİ")
        print("="*70)
        print(f"{'Model':<15} {'Toplam':<10} {'Conv':<10} {'FC':<10} {'Diğer':<10}")
        print("-"*70)
        
        for model in models:
            info = cls.get_model_info(model)
            print(f"{info['name']:<15} {info['total_layers']:<10} "
                  f"{info['conv_layers']:<10} {info['fc_layers']:<10} "
                  f"{info['other_layers']:<10}")
        
        print("="*70)


# Kullanım örneği
if __name__ == "__main__":
    # Mevcut modelleri listele
    CNNModels.list_models()
    
    # AlexNet'i al
    print("\n AlexNet Katmanları:")
    alexnet = CNNModels.get_model('alexnet')
    
    for i, layer in enumerate(alexnet, 1):
        print(f"{i}. {layer['name']:<12} ({layer['type']})")
    
    # Model bilgisi
    print("\nModel İstatistikleri:")
    for model_name in ['alexnet', 'vgg16', 'resnet50', 'mobilenet']:
        info = CNNModels.get_model_info(model_name)
        print(f"\n{model_name.upper()}:")
        print(f"  Toplam Katman: {info['total_layers']}")
        print(f"  Conv Katmanları: {info['conv_layers']}")
        print(f"  FC Katmanları: {info['fc_layers']}")