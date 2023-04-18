import torch
from models.attention import AttentionSimilarity
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_model(args, num_classes, load_ckpt=True):

    model, attention = None, None
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50-supcon':
            from models.resnet_supcon import SupConResNet
            model = SupConResNet(num_classes=num_classes)
            if load_ckpt:
                checkpoint = torch.load("./checkpoints/{in_dataset}/pytorch_{model_arch}_imagenet/supcon.pth".format(
                    in_dataset=args.in_dataset, model_arch=args.model_arch))
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
                model.load_state_dict(state_dict, strict=False)
    else:
        # create model
        if args.model_arch == 'densenet':
            model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
                                 dropRate=args.droprate, normalizer=None, method=args.method, p=args.p)
        elif args.model_arch == 'densenet-supcon':
            from models.densenet_ss import DenseNet3
            model = DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
                                     dropRate=args.droprate, normalizer=None, method=args.method, p=args.p)
        elif args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet18-supcon':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet18-supce':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet34':
            from models.resnet import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet34-supcon':
            from models.resnet_ss import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet34-supce':
            from models.resnet_ss import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        if load_ckpt:
            if args.in_dataset == "CIFAR-100":
                path = "/home/zhangji/OOD-project/knn-ood-master/pretrained/SupCon/cifar100_models/Finetune_SupCon_cifar100_resnet34_lr_0.1_bsz_128_trial_0_cosine/Finetune_epoch_500.pth"
            else:
                path = "/home/zhangji/OOD-project/knn-ood-master/pretrained/SupCon/cifar10_models/Finetune_SupCon_cifar10_resnet18_lr_0.1_bsz_128_trial_ce_cosine/Finetune_epoch_500.pth"

            checkpoint = torch.load(path,  map_location=device)
            state_dict = {str.replace(k, 'encoder.', ''): v for k, v in checkpoint['model'].items()}
            state_dict = {str.replace(k, 'shortcut.', 'downsample.'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    return model
