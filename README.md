## Аугментация
Для аугементации были использованы [RandomHorizontalFlip](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html), [ ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html), [ RandAugment](https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html),[ CoarseDropout](https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html#coarsedropout)  
Сила аугементации регулируется гиперпарметрами. Aугементации изображений происходит во время обучения. Код с описанием аугементации -  [get_transform.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/get_transform.py) .

Датасет после аугементации можно скачать по ссылке - [google disk](https://drive.google.com/file/d/1Dq8azgqLBEEYUVE9Xx5k5hZaZ3viZSwv/view?usp=sharing) 

## Подготовка данных
Изображения переводятся в размер 224 на 224, аугементируются, переводятся в черное-белый формат (нужно, что бы сеть не переобучилась на цвете), нормализуются. Код с описанием обработки -  [get_transform.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/get_transform.py).

Все изображения делятся на две части train и valid. В train идет 70% изображений. Для загрузки изображений используется pytorch Dataset и pytorch DataLoader. Код для pytorch DataLoader - [get_dataloaders.py]( https://github.com/VladSmirN/classification_of_car_models/blob/master/get_dataloaders.py). Код для pytorch Dataset - [CarDataset.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/CarDataset.py)

## Обучение модели 
Для создание модели использовался pytorch и библиотека [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable//index.html).
Код с моделью PyTorch Lightning  [CarModel.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/CarModel.py).

Код для обучения - [train.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/train.py) 

Для backbone была выбрана сеть convnext_small. 

Оптимизатор - Adam. 

Learning Rate Scheduling - [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)

Для логирования и оценки качества модели использовался  wandb.  

Изменение ошибки во время обучения :
![alt text](https://github.com/VladSmirN/classification_of_car_models/blob/master/images/W%26B%20Chart%209_15_2024%2C%208_28_05%20PM.png) 

Точность на валидационном наборе: 1.0

Все метрики обученных моделей можно посмотреть на проекте в wandb [ссылка на проект](https://wandb.ai/vladsmirn/classification_of_car_models?nw=nwuservladsmirn)

## Тестирование модели 
Модель тестировалась на данных собранных из интернета. Тестовый датасет можно скачать по ссылке - 
[google disk](https://drive.google.com/file/d/1l9hhI_xdL-E8_e0HK56P-QGWhqdLH7AE/view?usp=sharing) 

Качество работы можно оценить по Confusion matrix
![alt text](https://github.com/VladSmirN/classification_of_car_models/blob/master/images/confusion_matrix_best.png)

точность на тестовом наборе: 0.72

## Запуск модели
Для запуска модели onnx нужно воспользоваться скриптом -  [run_onnx.py](https://github.com/VladSmirN/classification_of_car_models/blob/master/run_onnx.py).

Скачать onnx модель можно по ссылке [google disk](https://drive.google.com/file/d/1d9GSgXS3q3i3nai1WWmCxPyuHUYXN-gO/view?usp=sharing)

