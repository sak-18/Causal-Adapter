import torch

def get_attribute_ids(attribute_size):
    attribute_indices = {}
    idx = 0
    for attr, size in attribute_size.items():
        attribute_indices[attr] = list(range(idx, idx + size))
        idx += size
    return attribute_indices

class SelectParentAttributesTransform:
    def __init__(self, name, attribute_size, graph_structure):
        self.name = name
        attribute_indices = get_attribute_ids(attribute_size)

        self.attr_ids = attribute_indices[name] if self.name != 'image' else None
        self.pa_ids = sum([attribute_indices[attr] for attr in graph_structure[name]], [])

    def __call__(self, img, attrs):
        if self.name == 'image':
            # only return the parent attr during vae training as condition, for example, for adni, the parent attr is "brain_vol", "vent_vol", "slice"
            return img, torch.Tensor([attrs[idx] for idx in self.pa_ids])
        else:
            # to train flow, each time return son attr and parent node, for celeba [0,0,1,1], son attr is [1] for no_beard flow, 1 for bald, [0,0] for parent
            return torch.Tensor([attrs[idx] for idx in self.attr_ids]), torch.Tensor([attrs[idx] for idx in self.pa_ids])

class ReturnDictTransform:
    def __init__(self, attribute_size):
        self.attribute_ids = get_attribute_ids(attribute_size)

    def __call__(self, img, attrs):
        # Morphomnist example{'image':..., 'thickness': tensor([-0.6687]), 'intensity': tensor([-0.6007]), 'digit': tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])}
        return {**{"image": img}, **{attr: attrs[ids] for attr, ids in self.attribute_ids.items()}}
