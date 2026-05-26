import torch
from pytorch_lightning import Trainer
from json import load
import sys
sys.path.append("../../")
import os
import argparse
import joblib
import csv
from ctf_datasets.morphomnist.dataset import MorphoMNISTLike
from ctf_datasets.celeba.dataset import Celeba
from ctf_datasets.adni.dataset import ADNI
from ctf_datasets.pendulum.dataset import PendulumLike
from ctf_datasets.celeba_hq.dataset import Celebahq
from ctf_datasets.transforms import get_attribute_ids
from models.classifiers.classifier import Classifier
from models.classifiers.celeba_classifier import CelebaClassifier,Celeba_anticausal_Classifier
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier
from models.classifiers.pendulum_classifier import PendClassifier
from models.classifiers.adni_classifier import ADNIClassifier
from models.utils import generate_checkpoint_callback, generate_early_stopping_callback, generate_ema_callback
from torchvision.transforms import RandomHorizontalFlip


dataclass_mapping = {
    "morphomnist": MorphoMNISTLike,
    "celeba": Celeba,
    "adni": ADNI,
    "pendulum": PendulumLike,
    "celebahq": Celebahq

}

classifier_mapping = {
    "morphomnist": Classifier,
    "celeba": CelebaClassifier,
    "adni": ADNIClassifier,
    "pendulum": PendClassifier,
    "celebahq": Celeba_anticausal_Classifier
    #"celebahq": CelebaClassifier
}


def train_classifier(classifier, attr, train_set, val_set, config, default_root_dir, weights=None,dataset_name = None):
    #mode = 'min' if attr in ["age", "brain_vol", "vent_vol", "thickness", "intensity"] else 'max'
    mode = 'min' if attr in ["age", "brain_vol", "vent_vol", "thickness", "intensity","pendulum","light","shadow_length","shadow_position"] else 'max'
    callbacks = [
        generate_checkpoint_callback(attr + "_classifier", config["ckpt_path"], monitor="val_metric", mode=mode),
        generate_early_stopping_callback(patience=config["patience"], monitor="val_metric", mode=mode, min_delta=1e-5)
    ]

    if config["ema"] == "True":
        callbacks.append(generate_ema_callback(decay=0.999))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                      callbacks=callbacks,
                      default_root_dir=default_root_dir, max_epochs=config["max_epochs"])

    if weights != None:
        
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set), replacement=True)
        print("Using sampler!")
        train_data_loader = torch.utils.data.DataLoader(train_set, sampler=sampler, batch_size=config["batch_size_train"], drop_last=False, num_workers=16)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, drop_last=False, num_workers=16)


    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=16)
    trainer.fit(classifier, train_data_loader, val_data_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-config", '-clf', type=str, help="Classifier config file."
                        , default="./configs/adni/classifier.json")

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(42)

    args = parse_arguments()

    assert os.path.isfile(args.classifier_config), f"{args.classifier_config} is not a file"

    with open(args.classifier_config, 'r') as f:
        config_cls = load(f)

    dataset = config_cls["dataset"]
    attribute_size = config_cls["attribute_size"]

    attribute_ids = get_attribute_ids(attribute_size)

    if dataset == 'morphomnist':
        data = dataclass_mapping[dataset](attribute_size=attribute_size, split="train", normalize_=True)

        train_set, val_set = torch.utils.data.random_split(data, [config_cls["train_val_split"],
                                                              1-config_cls["train_val_split"]])
    elif dataset == 'pendulum':
        data = dataclass_mapping[dataset](attribute_size=attribute_size, split="train", normalize_=True)

        train_set, val_set = torch.utils.data.random_split(data, [config_cls["train_val_split"],1-config_cls["train_val_split"]])
    elif dataset == "celebahq":
        tr_transforms = RandomHorizontalFlip(0.5)
        attr_size = {
            "Smiling": 1,
            "Eyeglasses": 1,
            'Mouth_Slightly_Open':1
        }
        train_set = dataclass_mapping[dataset](attribute_size=attr_size,
                                             split="train", transform_cls=tr_transforms)

        val_set = dataclass_mapping[dataset](attribute_size=attr_size, split="test")
    
    
    else:
        # celeba, adni
        tr_transforms = RandomHorizontalFlip(0.5)

        train_set = dataclass_mapping[dataset](attribute_size=attribute_size,
                                             split="train", transform_cls=tr_transforms)

        #val_set = dataclass_mapping[dataset](attribute_size=attribute_size, split="valid")
        
        # try to combine train and test set
        # test_set = dataclass_mapping[dataset](attribute_size=attribute_size,
        #                                      split="valid", transform_cls=tr_transforms)
        # from torch.utils.data import ConcatDataset
        # train_set = ConcatDataset([train_set, test_set])
        val_set = dataclass_mapping[dataset](attribute_size=attribute_size, split="valid")

    
        

    for attribute in attribute_size.keys():
        print("Train "+ attribute +" classfier!!")
        if dataset == "adni":
            classifier = classifier_mapping[dataset](attr=attribute, num_outputs=config_cls["attribute_size"][attribute],
                                    lr=config_cls["lr"], children=config_cls["anticausal_graph"][attribute], num_slices=config_cls["attribute_size"][attribute],
                                    attribute_ids=attribute_ids, arch=config_cls["arch"])
            weights = None
        else:
            if dataset == "celeba" or dataset == "celebahq":
                if dataset == "celeba":
                    if sum(attribute_size.values()) == 4:
                        classifier = CelebaComplexClassifier(attr=attribute, context_dim=len(list(config_cls["anticausal_graph"][attribute])),
                                                        num_outputs=config_cls["attribute_size"][attribute],
                                                        lr=config_cls["lr"], version=config_cls["version"])
                    else:
                        classifier = CelebaClassifier(attr=attribute, num_outputs=config_cls["attribute_size"][attribute],
                                                lr=config_cls["lr"])
                elif dataset == "celebahq":
                    # classifier = CelebaClassifier(attr=attribute, num_outputs=config_cls["attribute_size"][attribute],
                    #                             lr=config_cls["lr"])
                    classifier = Celeba_anticausal_Classifier(attr=attribute, num_outputs=config_cls["attribute_size"][attribute],
                                                        lr=config_cls["lr"])

                if attribute == "Smiling":

                    # imglabel = train_set.attrs
                    # smile_label = imglabel[:, 0].long()   # choose one label only
                    # mouth_label = imglabel[:, 2].long()
                    # combined_classes = smile_label*2 + mouth_label
                    # class_counts = torch.bincount(combined_classes.long())
                    
                    # alpha = 0.5   # <-- your "rate": 0 = uniform, 1 = full inverse-freq
                    # class_weights = (1.0 / class_counts.float()) ** alpha
                    # # Assign weight to each sample based on its class
                    # weights = class_weights[combined_classes.long()]
                    imglabel = train_set.attrs
                    smile_label = imglabel[:, 0].long()   # choose one label only
                    class_counts = torch.bincount(smile_label)
                    alpha = 1.0   # <-- your "rate": 0 = uniform, 1 = full inverse-freq
                    class_weights = (1.0 / class_counts.float()) ** alpha
                    weights = class_weights[smile_label]
                    #weights = torch.tensor(joblib.load("../../ctf_datasets/celeba/weights/weights_smiling.pkl")).double()
                    #weights = None
                elif attribute == "Eyeglasses":
                    imglabel = train_set.attrs
                    eye_label = imglabel[:, 1].long()   # choose one label only
                    class_counts = torch.bincount(eye_label)
                    alpha = 1.0   # <-- your "rate": 0 = uniform, 1 = full inverse-freq
                    class_weights = (1.0 / class_counts.float()) ** alpha
                    weights = class_weights[eye_label]
                    #weights = torch.tensor(joblib.load("../../ctf_datasets/celeba/weights/weights_eyes.pkl")).double()

                elif attribute in {"No_Beard", "Bald"}:
                    labels = train_set.attrs[: , classifier.variables[attribute]].long()
                    print((labels == 1).sum(), (labels==0).sum())
                    class_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
                    print(class_count)
                    class_weights = 1. / class_count.float()

                    weights = class_weights[labels]
                    print(weights)

                else:
                    weights = None

            elif dataset == 'pendulum':
                #morphomnist
                classifier = PendClassifier(attr=attribute, width=8, num_outputs=config_cls["attribute_size"][attribute],
                                        context_dim=len(list(config_cls["anticausal_graph"][attribute])), lr=config_cls["lr"])
                weights = None

            else:
                #morphomnist
                classifier = Classifier(attr=attribute, width=8, num_outputs=config_cls["attribute_size"][attribute],
                                        context_dim=len(list(config_cls["anticausal_graph"][attribute])), lr=config_cls["lr"])
                weights = None


        train_classifier(classifier, attribute, train_set, val_set, config_cls, default_root_dir=config_cls["ckpt_path"], weights=weights,dataset_name=dataset)
