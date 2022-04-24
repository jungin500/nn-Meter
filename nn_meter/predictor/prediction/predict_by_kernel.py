# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .utils import get_kernel_name
from .extract_feature import get_predict_features


def merge_conv_kernels(kernelname):
    """
    to speed up, we merge conv and dwconv related kernels into one kernel by their name
    """
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname


def predict_model(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    py = 0
    dicts = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        if rkernel not in dicts:
            dicts[rkernel] = []
        dicts[rkernel].append(features)

    for kernel in dicts:
        kernelname = get_kernel_name(kernel)
        if kernelname in predictors:
            pred = predictors[kernelname]
            
            # Workaround for newly created predictor - duplicate input exclusion required
            assert 'n_features_in_' in dir(pred), "scikit-learn RandomForestRegressor doesn't have n_features_in_; try downgrading library."
            if pred.n_features_in_ == len(dicts[kernel][0]) - 1:
                if kernelname in ['maxpool', 'avgpool', 'concat']:
                    continue
                # sanity check
                for feature in dicts[kernel]:
                    assert feature[1] == feature[2]  # Will remove 3rd argument
                for feature in dicts[kernel]:
                    feature.pop(2)
                    
            pys = pred.predict(dicts[kernel]) # in unit of ms
            if len(pys) != 0:
                py += sum(pys)

    return py


def nn_predict(predictors, kernel_units):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """

    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors)
    return py
