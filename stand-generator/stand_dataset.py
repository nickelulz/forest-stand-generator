import os, re, sys
from glob import glob
import numpy as np

ROOT_DIR = os.path.abspath("../")
DATA_DIR = os.path.abspath('./data/')

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils

class StandImage:
    IMG_SIZE = 640
    DIMENSIONS = (IMG_SIZE, IMG_SIZE)

    def __init__(self, root_path, index):
        self.root_path = root_path
        self.index = index
        self.label_path = os.path.join(root_path, f'labels/stand_{index}.txt')
        self.image_path = os.path.join(root_path, f'images/stand_{index}.png')
        self.mask_path  = os.path.join(root_path, f'masks/stand_{index}') 
        
        self.stands = []
        with open(self.label_path, 'r') as labels_file:
            line_index = 0
            for line in labels_file.readlines():
                # ignore the first 0
                raw = line.strip().split()[1:]
                if len(raw) % 2 != 0:
                    raise ValueError(f'Malformed stand labels at index {line_index} for stand {index}: length is {len(raw)}')

                normalized_coords = np.array(raw, dtype=np.float32).reshape(-1, 2)
                self.stands.append(normalized_coords)

    def load(self):
        return cv2.imread(self.image_path)

    def masks(self):
        mask_files = [f for f in os.listdir(self.mask_path) if f.endswith(".png")]
        mask_files.sort()  # Ensure consistent ordering

        masks = []
        class_ids = []

        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Ensure mask is binary
            mask = (mask > 0).astype(np.uint8)

            masks.append(mask)
            class_ids.append(1)  # Since all are "forest stands"

        if not masks:
            return np.zeros((640, 640, 0), dtype=np.uint8), np.array([])

        # Stack masks along depth dimension
        masks = np.stack(masks, axis=-1)
        return masks, np.array(class_ids, dtype=np.int32) 

    def show():
        """
        Draws the bounding boxes onto the stand
        """
        raise NotImplementedError()

class StandDataset:
    def __init__(self, path_name):
        self.name = path_name
        self.root_path   = os.path.join(DATA_DIR, path_name)
        self.images_path = os.path.join(self.root_path, 'images')
        self.labels_path = os.path.join(self.root_path, 'labels') 
        self.masks_path = os.path.join(self.root_path, 'masks')

        self.pool = []
        label_files = glob(os.path.join(self.labels_path, "*.txt"))
        for label_path in label_files:
            # get 2 directories up, i.e. /root/labels/stand_XXX.txt -> /root/
            root_path = os.path.abspath(os.path.join(label_path, "..", ".."))
            label_file_name = os.path.basename(label_path)
            image_file_name = label_file_name.replace('.txt', '.png')
            image_index = int(re.search(r'\d+', label_file_name).group())
            image = StandImage(root_path, image_index)

            self.pool.append(image)

            # compatibility with Mask RCNN
            self.add_image(source="stands",
                           image_id=image.index,
                           path=image.image_path,
                           mask_dir=image.mask_path)

class StandConfig(Config):
    NAME = "stands"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + forest stand
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10
