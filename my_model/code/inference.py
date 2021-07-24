"""This script consists of required functions for AWS SageMaker.

`model_fn`:
`input_fn`:
`predict_fn`:
`output_fn`:
"""
import os
import json

import numpy as np
from PIL import Image
from detecto import core


def clean_low_threshold_boxes(predictions):
    """
    """
    clean_predictions = []
    for prediction,threshold in zip(predictions[1],predictions[2]):
        if threshold.item()<0.5:
            continue
        clean_predictions.append(list(prediction.data.numpy()))
    return clean_predictions


def clean_unusual_sizes(predictions):
    """"""
    lengths = []
    widths  = []
    for prediction in predictions:
        lengths.append(prediction[2]-prediction[0])
        widths.append(prediction[3]-prediction[1])
    lengths = np.array(lengths)
    widths  = np.array(widths)
    ignore_indices = []
    for ind in range(len(lengths)):
        if lengths[ind]>np.mean(lengths)+2*np.std(lengths):
            ignore_indices.append(ind)
            continue
        if lengths[ind]<np.mean(lengths)-2*np.std(lengths):
            ignore_indices.append(ind)
            continue
        if widths[ind]>np.mean(widths)+2*np.std(widths):
            ignore_indices.append(ind)
            continue
        if widths[ind]<np.mean(widths)-2*np.std(widths):
            ignore_indices.append(ind)
            continue
    clean_prediction = []
    for ind,prediction in enumerate(predictions):
        if ind in ignore_indices:
            continue
        clean_prediction.append(prediction)

    return clean_prediction


def clean_overlapping_sizes(predictions):
    """"""
    clean_predictions = []
    predictions = [x for x in predictions]
    predictions = sorted(predictions)
    #for i range(len(predictions)):
    #
    ignore_indices = []
    for i in range(len(predictions)):
        i_area = (predictions[i][2]-predictions[i][0]) * \
                (predictions[i][3]-predictions[i][1])
        for j in range(i+1,len(predictions)):
            j_area = (predictions[j][2]-predictions[j][0]) * \
                    (predictions[j][3]-predictions[j][1])
            x_low = max(predictions[i][0],predictions[j][0])
            x_max = min(predictions[i][2],predictions[j][2])
            y_low = max(predictions[i][1],predictions[j][1])
            y_max = min(predictions[i][3],predictions[j][3])
            if x_low > x_max or y_low > y_max:
                break
            overlap_area = (x_max-x_low) * (y_max-y_low)
            if overlap_area>0.5*i_area or overlap_area>0.5*j_area:
                ignore_indices.append(i)
                break
    clean_predictions = []
    for i in range(len(predictions)):
        if i not in ignore_indices:
            clean_predictions.append(predictions[i])

    predictions = list(clean_predictions)
    predictions = [[x[1],x[0],x[3],x[2]] for x in predictions]
    predictions = sorted(predictions)
    ignore_indices = []
    for i in range(len(predictions)):
        i_area = (predictions[i][2]-predictions[i][0]) * \
                (predictions[i][3]-predictions[i][1])
        for j in range(i+1,len(predictions)):
            j_area = (predictions[j][2]-predictions[j][0]) * \
                    (predictions[j][3]-predictions[j][1])
            x_low = max(predictions[i][0],predictions[j][0])
            x_max = min(predictions[i][2],predictions[j][2])
            y_low = max(predictions[i][1],predictions[j][1])
            y_max = min(predictions[i][3],predictions[j][3])
            if x_low > x_max or y_low > y_max:
                break
            overlap_area = (x_max-x_low) * (y_max-y_low)
            if overlap_area>0.5*i_area or overlap_area>0.5*j_area:
                ignore_indices.append(i)
                break
    clean_predictions = []
    for i in range(len(predictions)):
        if i not in ignore_indices:
            clean_predictions.append(predictions[i])
    clean_predictions = [[x[1],x[0],x[3],x[2]] for x in clean_predictions]
    return clean_predictions


def clean_unaligned_images(predictions):
    """"""
    clean_predictions = []
    ignore_indices = []
    x_means = []
    y_means = []
    for prediction in predictions:
        x_means.append((prediction[0]+prediction[2])/2)
        y_means.append((prediction[1]+prediction[3])/2)
    x_means = np.array(x_means)
    y_means = np.array(y_means)
    if np.std(x_means)*2 > np.std(y_means) and np.std(y_means)*2 > np.std(x_means):
        return predictions

    for i in range(len(predictions)):
        if np.std(x_means)<np.std(y_means):
            if x_means[i]>np.mean(x_means)+2*np.std(x_means):
                ignore_indices.append(i)
            if x_means[i]<np.mean(x_means)-2*np.std(x_means):
                ignore_indices.append(i)
        else:
            if y_means[i]>np.mean(y_means)+2*np.std(y_means):
                ignore_indices.append(i)
            if y_means[i]<np.mean(y_means)-2*np.std(y_means):
                ignore_indices.append(i)

    for i in range(len(predictions)):
        if i not in ignore_indices:
            clean_predictions.append(predictions[i])
    
    return clean_predictions


def get_missing_boxes(predictions):
    """
    """
    predictions = sorted(predictions)
    x_means     = []
    y_means     = []
    for prediction in predictions:
        x_means.append((prediction[0]+prediction[2])/2)
        y_means.append((prediction[1]+prediction[3])/2)
    x_means = np.array(x_means)
    y_means = np.array(y_means)
    if np.std(x_means)>np.std(y_means):
        differences = [x_means[ind+1]-x_means[ind] for ind in range(len(x_means)-1)]
        differences = np.array(differences)
        missing_spans = [[predictions[0][0]-2*np.mean(differences),predictions[0][1],\
                predictions[0][0],predictions[0][3]]]
        for ind in range(len(differences)):
            if differences[ind]>np.mean(differences)+np.std(differences):
                missing_spans.append([predictions[ind][2],\
                        (predictions[ind][1]+predictions[ind+1][1])/2,\
                        predictions[ind+1][0],\
                        (predictions[ind][3]+predictions[ind+1][3])/2])
        missing_spans.append([predictions[-1][2],predictions[-1][1],\
                predictions[-1][2]+2*np.mean(differences),predictions[-1][3]])
    else:
        c_predictions = [[x[1],x[0],x[3],x[2]] for x in list(predictions)]
        c_predictions = sorted(c_predictions)
        y_means     = []
        for prediction in c_predictions:
            y_means.append((prediction[0]+prediction[2])/2)
        y_means     = np.array(y_means)
        differences = [y_means[ind+1]-y_means[ind] for ind in range(len(y_means)-1)]
        differences = np.array(differences)
        missing_spans = []
        missing_spans.append([c_predictions[0][1],c_predictions[0][0]-2*np.mean(differences),\
                c_predictions[0][3],c_predictions[0][0]])
        for ind in range(len(differences)):
            if differences[ind]>np.mean(differences)+np.std(differences):
                missing_spans.append([(c_predictions[ind][1]+c_predictions[ind+1][1])/2,\
                        c_predictions[ind][2],\
                        (c_predictions[ind][3]+c_predictions[ind+1][3])/2,\
                        c_predictions[ind+1][0]])
        missing_spans.append([c_predictions[-1][1],c_predictions[-1][2],\
                c_predictions[-1][3],c_predictions[-1][2]+2*np.mean(differences)])
    return missing_spans


def post_process(predictions):
    #c_predictions = predictions[1]
    c_predictions = clean_low_threshold_boxes(predictions)
    if len(c_predictions) == 0:
        return
    
    c_predictions = clean_unusual_sizes(c_predictions)
    if len(c_predictions) == 0:
        return

    c_predictions = clean_overlapping_sizes(c_predictions)
    if len(c_predictions) == 0:
        return

    c_predictions = clean_unaligned_images(c_predictions)
    if len(c_predictions) == 0:
        return

    return c_predictions


def model_fn(model_dir):
    model = core.Model(['stitch'])
    model = core.Model.load(open(os.path.join(model_dir, "model.pth"),"rb"), ['stitch'])
    return model


def input_fn(request_body, content_type='application/python-pickle'):
    if content_type == 'application/python-pickle':
        image_data = np.array(Image.open(request_body).convert('RGB'))
        return image_data
    else:
        raise Exception(f'Unsupported content type: {content_type}')


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(predictions, accept='application/json'):
    if accept == 'application/json':
        results = post_process(predictions)
        return json.dumps(results), accept
    
    raise Exception(f'Requested unsupported content type in Accept: {accept}')