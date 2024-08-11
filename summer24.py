import cv2
import os
import pupil_apriltags
import PIL
import transformers
import sys
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from slatedata import *


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

print("hello world")

#st = extractSceneAndTake(extractSlateImg('SlateTest1.jpg'))
#print(st)

#TODO: make this iterate through a file
for filename in sys.argv[1:]:
    proccessAndRenameVid(filename, True)

