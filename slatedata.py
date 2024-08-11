import cv2
import os
import pupil_apriltags
import PIL
import transformers
import numpy as np
from PIL import Image
from pupil_apriltags import Detector
import numpy as np
import re
from collections import Counter
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#print("hello world")
processer = None
model = None


def extractSlateImg(filename, debug = False):
    image = cv2.imread(filename)
    #Convert the image to grayscale 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    d = at_detector.detect(image)
    if len(d) == 0:
        if debug:
            print("no tag")
        return None
    
    aprCent = list(d[0].center)
    #print(aprCent)
    #cv2.circle(image, (int(aprCent[0]), int(aprCent[1])), 50, (255, 0, 0), 5) 
    #draws circle over april tag, when the slate is small in frame this can cover the writing causing bugs
    #cv2_imshow(image)

    homoGr = d[0].homography #list?
    #print("homoGr")
    #print(homoGr)

    homography, status = cv2.findHomography(d[0].corners, np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    #print("status")
    #print(status)

    #print("homography") # mult x y 1
    #print(homography)
    warp_img = cv2.warpPerspective(image, homography, (470, 360)) #demensions of the slate transformed into
    #im_dest = cv2.warpPerspective(image, homoGr, (100000, 100000))
    #resized_image_dest = cv2.resize(im_dest, (100, 100)) 
    flipWarp_img = cv2.flip(warp_img, 1)
    #cv2_imshow(flipWarp_img)
    return flipWarp_img

def ocr_image(src_img):
    global processer
    global model
    if not processer:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
    if not model:
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  
def ocrTextInRectangle(img, rect, debug = False, showImg = False):
    #rect is: [0, 100, 50, 200] for [x start, x len, y start, y len]
    y,height,x,width = rect
    crop_img = img[x:x+width, y:y+height]
    #if showImg:
        #cv2.imshow(crop_img)
    img = Image.fromarray(crop_img)
    color_img = img.convert('RGB')

    r = ocr_image(color_img)
    if debug:
        cv2.imwrite('slateCrop_{}.jpg'.format(f"recognized-as-{r}"), crop_img)
    return r

def extractSceneAndTake(exractedSlate, debug = False):
    #These numbers determine the box the take and scene will be read from in the image
    scene = ocrTextInRectangle(exractedSlate, [140, 130, 100, 130], debug) #showImg only for colab
    take = ocrTextInRectangle(exractedSlate, [280, 170, 110, 120], debug)
    return (scene, take)

def extractVidFrames(filename, debug = False):
    cap = cv2.VideoCapture(filename)

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(f"fps: {fps}!")
    # Calculate the interval between frames
    #interval = 1 / fps

    # Create a counter to keep track of the frame number
    frame_count = 0
    #print(cap.isOpened())
    cap.isOpened() == True
    results = []
    # Loop through the frames
    while cap.isOpened():

        # Capture the next frame
        ret, frame = cap.read()

        # If the frame is empty, break out of the loop
        if not ret:
            break

        # Increment the frame counter
        frame_count += 1

        # Save the frame as an image
        if frame_count%10 == 0:
            if debug:
                print(f"frame num: {frame_count}!")
                cv2.imwrite('frame_{}.jpg'.format(frame_count), frame)
                frameFile = ('frame_{}.jpg'.format(frame_count))
                esi = extractSlateImg(frameFile)#matrix representing the image itself an ndarray
            #print(esi)
            #print(type(esi))
            if isinstance(esi, np.ndarray):
                st = extractSceneAndTake(esi, debug)
                if debug:
                    cv2.imwrite('slate_{}.jpg'.format(frame_count), esi)
                    print(f"Found scene/take: {st}")
                results.append(st)
            elif len(results) > 6:
                if debug:
                    print("found at least 6")
                    print(f"results: {results}")
                cap.release()
                return results
            #elif esi == 0:


    # Release the video capture object
    if debug:
        print("went through entire video")
        print(f"results: {results}")
    cap.release()
    return results


def cleanScene(s, debug = False):
    match = re.search(r"[A-Za-z]\d+", s)
    #catches error when theres no capital Letter w/ num
    if not match:
        if debug:
            print("scene is not parsable returning none")
        return None
    return match[0]

def cleanTake(s, debug = False):
    match = re.search(r"\d+", s)
    #catches error when theres no num
    if not match:
        if debug:
            print("take is not parsable returning none")
        return None
    return match[0]
    #TODO: make this return list of all digets separated by spaces


def findScene(results):
    cleanS = [cleanScene(x[0]) for x in results]
    #finds most common one (use Count dict) FUTURE: maybe combine w/ findTake
    sceneCountDict = Counter(cleanS)
    modeScene = sceneCountDict.most_common(1)
    #print(cleanS)
    #print(modeScene)
    sceneRes = modeScene[0][0]
    return(sceneRes)

def findTake(results):
    cleanT = [cleanTake(x[1]) for x in results]
    #finds most common one
    takeCountDict = Counter(cleanT)
    modeTake = takeCountDict.most_common(1)
    #TODO: edge case when mode is a tie and or low mode
    #print(cleanT)
    #print(modeTake)
    takeRes = modeTake[0][0]
    return(takeRes)

def proccessAndRenameVid(filename, debug = False):
    scene,take = proccessVideo(filename, debug)
    #TODO: rename here
    dir, f = os.path.split(filename)
    newName = os.path.join(dir, f"{scene}.{take}.{f}")
    print(f"renaming {filename} to {newName}")
    os.rename(filename, newName)

def proccessVideo(filename, debug = False):
    results = extractVidFrames(filename, debug)
    if len(results) <= 2:
        s = "-scene-"
        t = "-take-"
    else:
        s = findScene(results)
        t = findTake(results)
    if debug:
        print(f"----processing results for {filename}----")
        print(results)
        print()
        print(s)
        print(t)
    return s, t
    