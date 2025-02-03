import os, cv2
import numpy as np
from stand_dataset import * 

def generate_masks(dataset):
    mask_dir = os.path.join(dataset.root_path, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    for image in dataset.pool:
        instance_mask_dir = os.path.join(mask_dir, f'stand_{image.index}')
        os.makedirs(instance_mask_dir, exist_ok=True)

        for stand_index, stand in enumerate(image.stands):
            # each stand will get its own distinct mask

            instance_mask = np.zeros(StandImage.DIMENSIONS, dtype=np.uint8)
            polygon = np.int32(stand * StandImage.IMG_SIZE).reshape(-1, 1, 2)
            cv2.fillPoly(instance_mask, [polygon], color=255)

            instance_mask_path = os.path.join(instance_mask_dir, f'stand-{stand_index}.png')
            cv2.imwrite(instance_mask_path, instance_mask)

    print(f'Masks for dataset {dataset.name} saved in {mask_dir}')
