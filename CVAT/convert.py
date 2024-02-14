from sklearn.model_selection import train_test_split
import os
import shutil
import yaml

def create_dataset(name, src_annotations):
    """
    Creates a dataset from the CVAT annotations and images.
    name = the name of the dataset.
    src_annotations = the folder containing the annotations in the CVAT folder.
    """
    files = os.listdir(f"CVAT/{src_annotations}")
    train_files, val_files = train_test_split(files, test_size=0.2)

    image_train = f'datasets/{name}/images/train'
    image_val = f'datasets/{name}/images/val'
    label_train = f'datasets/{name}/labels/train'
    label_val = f'datasets/{name}/labels/val'
    os.makedirs(label_train, exist_ok=True)
    os.makedirs(label_val, exist_ok=True)
    os.makedirs(image_train, exist_ok=True)
    os.makedirs(image_val, exist_ok=True)  

    # Copy the image and label files to the train folders
    for file in train_files:
        file_name = os.path.splitext(file)[0]

        # Get the path to the image and label files
        src_image = os.path.join('CVAT\images', file_name + '.jpg')
        if not os.path.exists(src_image):
            src_image = os.path.join('CVAT\images', file_name + '.png')
        src_label = os.path.join(f'CVAT\{src_annotations}', file_name + '.txt')

        dst_image = os.path.join(image_train, file_name + '.jpg')
        if not os.path.exists(dst_image):
            dst_image = os.path.join(image_train, file_name + '.png')
        dst_label = os.path.join(label_train, file_name + '.txt')

        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)
    
    # copy the image and label files to the validation folders
    for file in val_files:
        file_name = os.path.splitext(file)[0]

        # Get the path to the image and label files
        src_image = os.path.join('CVAT\images', file_name + '.jpg')
        if not os.path.exists(src_image):
            src_image = os.path.join('CVAT\images', file_name + '.png')
        src_label = os.path.join(f'CVAT\{src_annotations}', file_name + '.txt')

        dst_image = os.path.join(image_val, file_name + '.jpg')
        if not os.path.exists(dst_image):
            dst_image = os.path.join(image_val, file_name + '.png')
        dst_label = os.path.join(label_val, file_name + '.txt')

        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)

def create_yaml(name):
    yaml_data = {
        'path': f'C:\\GitHub\\mt2024-YOLOv8-ensemble\\datasets\\{name}',
        'train': 'images\\train',
        'val': 'images\\val',
        'names': {
            0: 'clustered other',
            1: 'clear',
            2: 'discrete crystal',
            3: 'precipitate',
            4: 'clustered crystals',
            5: 'discrete other'
        }
    }
    # Write the YAML data to the file
    with open(name + '.yaml', 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

# run functions
create_dataset(name="crystals400labels", src_annotations="annotations")
create_yaml("crystals400labels")










