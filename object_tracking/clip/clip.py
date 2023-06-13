import sys
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from math_utils import softmax  # noqa: E402C
# logger
from logging import getLogger  # noqa: E402

from simple_tokenizer import SimpleTokenizer as _Tokenizer

logger = getLogger(__name__)

_tokenizer = _Tokenizer()

# ======================
# Parameters
# ======================

WEIGHT_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx'
MODEL_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx.prototxt'
WEIGHT_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx'
MODEL_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx.prototxt'
WEIGHT_RN50_IMAGE_PATH = 'RN50-encode_image.onnx'
MODEL_RN50_IMAGE_PATH = 'RN50-encode_image.onnx.prototxt'
WEIGHT_RN50_TEXT_PATH = 'RN50-encode_text.onnx'
MODEL_RN50_TEXT_PATH = 'RN50-encode_text.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

IMAGE_SIZE = 224

model_type = 'ViTB32'
#'RN50'

# ======================
# Main functions
# ======================

def tokenize(texts, context_length=77, truncate=False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = np.array(tokens)

    result = result.astype(np.int64)

    return result


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = round(im_w * scale), round(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = img[:, :, ::-1]  # BGR -> RBG
    img = img / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(net, img, text_feature):
    img = preprocess(img)

    # feedforward
    output = net.predict([img])

    image_feature = output[0]

    image_feature = image_feature / np.linalg.norm(image_feature, ord=2, axis=-1, keepdims=True)

    pred_list = []
    logit_scale = 100
    for feature in text_feature:
        logits_per_image = (image_feature * logit_scale).dot(feature.T)
        pred = softmax(logits_per_image, axis=1)
        pred = pred[0]
        pred = np.expand_dims(pred,axis=0)
        pred_list.append(pred)

    return pred_list


def predict_text_feature(net, text):
    text = tokenize(text)

    # feedforward
    output = net.predict([text])

    text_feature = output[0]

    text_feature = text_feature / np.linalg.norm(text_feature, ord=2, axis=-1, keepdims=True)

    return text_feature


def recognize_clip(net_clip, img):
    # img is bgr format
    net_image = net_clip["net_image"]
    net_text = net_clip["net_text"]
    text_feature = net_clip["text_feature"]

    # inference
    pred_list = predict(net_image, img, text_feature)

    # show results
    return pred_list


def create_clip(text_inputs, env_id):
    dic_model = {
        'ViTB32': (
            (WEIGHT_VITB32_IMAGE_PATH, MODEL_VITB32_IMAGE_PATH),
            (WEIGHT_VITB32_TEXT_PATH, MODEL_VITB32_TEXT_PATH)),
        'RN50': (
            (WEIGHT_RN50_IMAGE_PATH, MODEL_RN50_IMAGE_PATH),
            (WEIGHT_RN50_TEXT_PATH, MODEL_RN50_TEXT_PATH)),
    }
    (WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH), (WEIGHT_TEXT_PATH, MODEL_TEXT_PATH) = dic_model[model_type]

    # model files check and download
    logger.info('Checking encode_image model...')
    check_and_download_models(WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH, REMOTE_PATH)
    logger.info('Checking encode_text model...')
    check_and_download_models(WEIGHT_TEXT_PATH, MODEL_TEXT_PATH, REMOTE_PATH)

    memory_mode = ailia.get_memory_mode(
        reduce_constant=True, ignore_input_with_initializer=True,
        reduce_interstage=False, reuse_interstage=False)
    net_image = ailia.Net(MODEL_IMAGE_PATH, WEIGHT_IMAGE_PATH, env_id=env_id, memory_mode=memory_mode)
    net_text = ailia.Net(MODEL_TEXT_PATH, WEIGHT_TEXT_PATH, env_id=env_id, memory_mode=memory_mode)
    text_feature = []
    for category in text_inputs:
        text_feature.append(predict_text_feature(net_text, category))

    return {"net_image":net_image, "net_text":net_text, "text_feature":text_feature}
