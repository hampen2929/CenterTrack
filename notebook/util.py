import os
import json
from glob import glob


# court_label_dict = {
#     '0': 'center',
#     '1': 'singles_upper_right',
#     '2': 'singles_upper_left',
#     '3': 'singles_lower_left',
#     '4': 'singles_lower_right',
#     '5': 'doubles_upper_right',
#     '6': 'doubles_upper_left',
#     '7': 'doubles_lower_left',
#     '8': 'doubles_lower_right',
#     '9': 'service_upper_right',
#     '10': 'service_upper_center',
#     '11': 'service_upper_left',
#     '12': 'service_lower_left',
#     '13': 'service_lower_center',
#     '14': 'service_lower_right',   
# }

category_id_dict = {
    'person': 1,
    'ball': 2,
    
    'doubles_upper_right': 3,
    'doubles_upper_left': 3,
    'doubles_lower_left': 3,
    'doubles_lower_right': 3,
    
    'singles_upper_right': 4,
    'singles_upper_left': 4,
    'singles_lower_left': 4,
    'singles_lower_right': 4,
    
    'service_upper_right': 5,
    'service_upper_center': 5,
    'service_upper_left': 5,
    'service_lower_left': 5,
    'service_lower_center': 5,
    'service_lower_right': 5,
    
    'center': 0,
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

def get_annotations(shape, image_id, annotation_id, target_labels=None):
    file_dict = {
        'segmentation': [[]],
        'area': None,
        'iscrowd': 0,
        'image_id': 99999,
        'bbox': [],
        'category_id': None,
        'id': 999
    }
    point = shape['points']
    xmin, ymin = point[0]
    xmax, ymax = point[1]
    
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
    # categories = [
    #     {'supercategory': 'court_edge', 'id': 1, 'name': 'court_edge'},
    #     {'supercategory': 'court_edge', 'id': 2, 'name': 'doubles_lower_left'},
    #     {'supercategory': 'court_edge', 'id': 3, 'name': 'doubles_lower_right'},
    #     {'supercategory': 'court_edge', 'id': 4, 'name': 'doubles_upper_left'},
    #     {'supercategory': 'court_edge', 'id': 5, 'name': 'doubles_upper_right'}
    # ]

    categories = [
        {'supercategory': 'person', 'id': 1, 'name': 'person'},
        {'supercategory': 'ball', 'id': 2, 'name': 'ball'},
        {'supercategory': 'court', 'id': 3, 'name': 'court_edge'},
    ]

    return categories
