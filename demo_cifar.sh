python feature_collect.py.py --in-dataset CIFAR-100  --out-datasets SVHN places365 LSUN iSUN dtd --name resnet34-supcon  --model-arch resnet34-supcon --ckp "/pretrained/CL/MODE-F/Finetune_epoch_500.pth"
python CSD.py   --in-dataset CIFAR-100  --out-datasets SVHN places365 LSUN iSUN dtd --name resnet34-supcon  --model-arch resnet34-supcon
