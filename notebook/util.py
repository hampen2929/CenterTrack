import os
import json
from glob import glob


category_id_dict = {
#     'court_edge': 1,
#     'doubles_lower_left': 2,
#     'doubles_lower_right': 3,
#     'doubles_upper_left': 4,
#     'doubles_upper_right': 5,
    'person': 1,
    'ball': 37,
    '5': 5,
    '6': 5,
    '7': 5,
    '8': 5,
}


def load_json(data_path):
    f = open(data_path, 'r')
    jsonData = json.load(f)
    f.close()
    return jsonData

def save_json(label, save_path):
    f = open(save_path, "w")
    json.dump(label, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

def get_json_file_paths(data_dir):
    data_path = os.path.join(data_dir, '*.json')
    label_file_paths = glob(data_path)
    label_file_paths.sort()
    return label_file_paths

def get_info():
    file_dict = {
        'description': 'dummy',
         'url': 'dummy',
         'version': 'dummy',
         'year': 2045,
         'contributor': 'dummy',
         'date_created': '1993/01/24'
    }
    return file_dict

def get_licenses():
    file_dict = {
        'url': 'dummy',
        'id': 1,
        'name': 'dummy'}
    return file_dict

def get_images(label_file, image_id, file_name=None):
    file_dict = {
        'license': 1,
         'file_name': None,
         'coco_url': 'dummy',
         'height': None,
         'width': None,
         'date_captured': 'dummy',
         'flickr_url': 'dummy',
         'id': 99999
    }
    
    if file_name is None:
        file_name = label_file['imagePath']

    height = label_file['imageHeight']
    width = label_file['imageWidth']
    
    file_dict['file_name'] = file_name
    file_dict['height'] = height
    file_dict['width'] = width
    
    file_dict['id'] = image_id

    return file_dict

def get_annotations(shape, image_id, annotation_id, length=30, target_labels=None):
    file_dict = {
        'segmentation': [[]],
        'area': None,
        'iscrowd': 0,
        'image_id': 99999,
        'bbox': [],
        'category_id': None,
        'id': 999
    }
    point = shape['points'][0]
    
    x, y = point
    xmin = x - length / 2
    ymin = y - length / 2
    xmax = x + length / 2
    ymax = y + length / 2
    
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    
    bbox = [xmin, ymin, width, height]
    
    label = shape['label']
    
    if isinstance(label, str):
        if label.isdecimal():
            category_id = int(label)
        else:
            category_id = category_id_dict[label]
    else:
        msg = 'label must be str not {}'.format(type(label))
        raise ValueError(msg)
    
    file_dict['area'] = area
    file_dict['bbox'] = bbox
    file_dict['category_id'] = category_id
    
    file_dict['image_id'] = image_id
    file_dict['id'] = annotation_id

    return file_dict

def get_categories():
    categories = [
        {'supercategory': 'court_edge', 'id': 1, 'name': 'court_edge'},
        {'supercategory': 'court_edge', 'id': 2, 'name': 'doubles_lower_left'},
        {'supercategory': 'court_edge', 'id': 3, 'name': 'doubles_lower_right'},
        {'supercategory': 'court_edge', 'id': 4, 'name': 'doubles_upper_left'},
        {'supercategory': 'court_edge', 'id': 5, 'name': 'doubles_upper_right'}
    ]
    return categories