from paddleocr import PaddleOCR
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, ToPILImage
import os
import paddle
cfg={
    "crop_size": 256,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "size": 256
}

def make_ocr_prompt(ocr_result):
        if ocr_result is None or all(text_inform[1][1] < 0.85 for text_inform in ocr_result):
            return "The image includes text descriptions: N/A."
        ocr_texts = [text_inform[1][0] for text_inform in ocr_result if text_inform[1][1] >= 0.85]
        
        # Verbalization
        if len(ocr_texts)!=0:
            verbalization_ocr = 'The image includes text descriptions: '
            for i, txt in enumerate(ocr_texts):
                verbalization_ocr += txt
                if i!=len(ocr_texts)-1:
                    verbalization_ocr += ', and '
            verbalization_ocr += '.'
        else:
            verbalization_ocr = ''
        return verbalization_ocr
def make_human_string(*args):
    out = ''
    for ind, arg in enumerate(args):
        out += arg
        if len(args)-1 != ind: out += ' '
    return out

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# text_ocr_model = BertModel.from_pretrained("bert-base-uncased")
image_path = '/data/luogen_code/check_image_0.png'
save_dir='/data/luogen_code/'
image = Image.open(image_path)
image_np = np.array(image)
# paddle.set_device('gpu:0')

ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
ocr_model.to("cuda:0")
with torch.inference_mode():
    ocr_results = ocr_model.ocr(image_np)
    verbalization_result = make_ocr_prompt(ocr_results[0])
print(verbalization_result)


# print(ocr_results[0])

