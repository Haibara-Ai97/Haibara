import base64
import json
import unittest
from main import app
import numpy as np
import cv2
import random
from utils import save_image
import os


class FlaskTestCase(unittest.TestCase):
    def setUp(self):
         app.testing = True
         self.client = app.test_client()

    # def test_2(self):
    #
    #     # crop
    #     img_path = 'image/test2.bmp'
    #     pts = np.float32([[980, 212], [1477, 155], [1514, 604], [1086, 640]])
    #     save_path = 'res/2'
    #
    #     with open(img_path, 'rb') as image_file:
    #         encoded_str = base64.b64encode(image_file.read()).decode('utf-8')
    #     pts_list = pts.tolist()
    #
    #     data = {
    #         'image': encoded_str,
    #         'fixedParam': {
    #             'position': pts_list,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/crop', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res0.png'))
    #
    #     # shift
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'position': [[0, 57], [497, 0], [534, 449], [106, 485]],
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/perspective_shift', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res1.png'))
    #
    #     # HSV
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/HSV_Image', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     original = response_data['dynamicParam']['original']
    #     save_image(response_image, os.path.join(save_path, 'res2.png'))
    #
    #     # bluemask
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'low': [110, 50, 50],
    #             'up': [130, 255, 255],
    #         },
    #         'dynamicParam':{
    #             "original": original,
    #         },
    #     }
    #     response = self.client.post('/identify/bluemask', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res3.png'))
    #
    #     # countour
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': {
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/findcontour', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     contours = response_data['dynamicParam']['contours']
    #     save_image(response_image, os.path.join(save_path, 'res4.png'))
    #
    #     # centers
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': {
    #             'contours': contours,
    #         },
    #     }
    #     response = self.client.post('/identify/countcenters', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     centers = response_data['dynamicParam']['centers']
    #     save_image(response_image, os.path.join(save_path, 'res5.png'))
    #
    #     # count
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'brick': 97.5,
    #             'weight': 0.049677143871919,
    #         },
    #         'dynamicParam': {
    #             'centers': centers,
    #         },
    #     }
    #     response = self.client.post('/identify/countdistance', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res6.png'))

    # def test_2(self):
    #
    #     img_path = 'image/test10.bmp'
    #     pts = np.float32([[685,606],[1668, 703],[1645, 1770],[511, 1840]])
    #     save_path = 'res/10'
    #
    #     with open(img_path, 'rb') as image_file:
    #         encoded_str = base64.b64encode(image_file.read()).decode('utf-8')
    #     pts_list = pts.tolist()
    #
    #     # remap
    #     data = {
    #         'image': encoded_str,
    #         'fixedParam': {
    #             "mapx":"map/10/mapx.npy",
    #             "mapy":"map/10/mapy.npy",
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/remap', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res0.png'))
    #
    #     # crop
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'position': pts_list,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/crop', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res1.png'))
    #
    #     # shift
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'position': [[174, 0], [1157, 97], [1134, 1164], [0, 1234]],
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/perspective_shift', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res2.png'))
    #
    #     # HSV
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/HSV_Image', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     original = response_data['dynamicParam']['original']
    #     save_image(response_image, os.path.join(save_path, 'res3.png'))
    #
    #     # bluemask
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'low': [100, 35, 35],
    #             'up': [130, 255, 255],
    #         },
    #         'dynamicParam':{
    #             "original": original,
    #         },
    #     }
    #     response = self.client.post('/identify/bluemask', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res4.png'))
    #
    #     # countour
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': {
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/findcontour', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     contours = response_data['dynamicParam']['contours']
    #     save_image(response_image, os.path.join(save_path, 'res5.png'))
    #
    #     # centers
    #     data = {
    #         'image': response_image,
    #         'fixedParam': '123',
    #         'dynamicParam': {
    #             'contours': contours,
    #         },
    #     }
    #     response = self.client.post('/identify/countcenters', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     centers = response_data['dynamicParam']['centers']
    #     save_image(response_image, os.path.join(save_path, 'res6.png'))
    #
    #     # count
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'brick': 301.0,
    #             'weight': 0.0180170228333514,
    #         },
    #         'dynamicParam': {
    #             'centers': centers,
    #         },
    #     }
    #     response = self.client.post('/identify/countdistance', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res7.png'))


    def test_14_canny(self):

        img_path = 'image/test14.bmp'
        pts = np.float32([[1250,173],[1673,160],[1710,1346],[1320,1427]])
        save_path = 'res/14_canny'

        with open(img_path, 'rb') as image_file:
            encoded_str = base64.b64encode(image_file.read()).decode('utf-8')
        pts_list = pts.tolist()

        # crop
        data = {
            'image': encoded_str,
            'fixedParam': {
                'position': pts_list,
            },
            'dynamicParam': '123',
        }
        response = self.client.post('/preprocess/crop', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        save_image(response_image, os.path.join(save_path, 'res0.png'))

        # shift
        data = {
            'image': response_image,
            'fixedParam': {
                'position': [[0, 13], [423, 0], [460, 1186], [70, 1267]],
            },
            'dynamicParam': '123',
        }
        response = self.client.post('/preprocess/perspective_shift', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        save_image(response_image, os.path.join(save_path, 'res1.png'))

        # gray
        data = {
            'image': response_image,
            'fixedParam': '123',
            'dynamicParam': '123',
        }
        response = self.client.post('/preprocess/gray', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        original = response_data['dynamicParam']['original']
        save_image(response_image, os.path.join(save_path, 'res2.png'))

        # blur
        data = {
            'image': response_image,
            'fixedParam': '123',
            'dynamicParam': {
                "original": original,
            },
        }
        response = self.client.post('/preprocess/blur', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        save_image(response_image, os.path.join(save_path, 'res3.png'))

        # canny
        data = {
            'image': response_image,
            'fixedParam': {
                'cannya': 40,
                'cannyb': 80,
            },
            'dynamicParam': {
                'original': original,
            },
        }
        response = self.client.post('/identify/canny', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        save_image(response_image, os.path.join(save_path, 'res4.png'))

        # houghlines
        data = {
            'image': response_image,
            'fixedParam': {
                'hougha': 40,
                'houghb': 20,
                'houghc': 20,
            },
            'dynamicParam': {
                'original': original,
            },
        }
        response = self.client.post('/identify/houghlines', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        lines = response_data['dynamicParam']['lines']
        save_image(response_image, os.path.join(save_path, 'res5.png'))

        # filter
        data = {
            'image': response_image,
            'fixedParam': {
                'angledown': 0,
                'angleup': 30,
                'threshold': 0,
            },
            'dynamicParam': {
                'lines': lines,
                'original': original,
            },
        }
        response = self.client.post('/identify/filterlines', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        lines = response_data['dynamicParam']['lines']
        save_image(response_image, os.path.join(save_path, 'res6.png'))

        # count
        data = {
            'image': response_image,
            'fixedParam': {
                'level': [414, 1, 95226],
                'map': [-0.0024, 7.43957],
            },
            'dynamicParam': {
                'lines': lines,
                'original': original,
            },
        }
        response = self.client.post('/count/countlevelin', json=data)
        response_data = response.get_json()
        response_image = response_data['image']
        save_image(response_image, os.path.join(save_path, 'res6.png'))


    # def test_14(self):
    #
    #     # crop
    #     img_path = 'image/test14.bmp'
    #     pts = np.float32([[1250,173],[1673,160],[1710,1346],[1320,1427]])
    #     save_path = 'res/14'
    #
    #     with open(img_path, 'rb') as image_file:
    #         encoded_str = base64.b64encode(image_file.read()).decode('utf-8')
    #     pts_list = pts.tolist()
    #
    #     data = {
    #         'image': encoded_str,
    #         'fixedParam': {
    #             'position': pts_list,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/crop', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res0.png'))
    #
    #     # shift
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'position': [[0, 13], [423, 0], [460, 1186], [70, 1267]],
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/perspective_shift', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res1.png'))
    #
    #     # resize
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'image_size': 352,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/resize', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     original_size = response_data['dynamicParam']['original_size']
    #     original = response_data['dynamicParam']['original']
    #     save_image(response_image, os.path.join(save_path, 'res2.png'))
    #
    #     # mean
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'mean_bgr': [103.939, 116.779, 123.68, 137.86],
    #         },
    #         'dynamicParam':{
    #             'original_size': original_size,
    #             "original": original,
    #         },
    #     }
    #     response = self.client.post('/preprocess/mean', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res3.png'))
    #
    #     # DexiNed
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'threshold': 150,
    #         },
    #         'dynamicParam': {
    #             'original_size': original_size,
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/dexiNed', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res4.png'))
    #
    #     # Houghlines
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'hougha': 15,
    #             'houghb': 40,
    #             'houghc': 15,
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/houghlines', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     lines = response_data['dynamicParam']['lines']
    #     save_image(response_image, os.path.join(save_path, 'res5.png'))
    #
    #     # filterlines
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'angledown': 0,
    #             'angleup': 20,
    #             'threshold': 250,
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #             'lines': lines,
    #         },
    #     }
    #     response = self.client.post('/identify/filterlines', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     lines = response_data['dynamicParam']['lines']
    #     save_image(response_image, os.path.join(save_path, 'res6.png'))
    #
    #     # count
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'level': [414, 1, 95226],
    #             'map': [-0.0024, 7.43957],
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #             'lines': lines,
    #         },
    #     }
    #     response = self.client.post('/count/countlevelin', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res7.png'))


    # def test_13(self):
    #
    #     # crop
    #     img_path = 'image/test13.bmp'
    #     pts = np.float32([[1298, 558], [1689, 558], [1689, 1450], [1298, 1450]])
    #     save_path = 'res/13'
    #
    #     with open(img_path, 'rb') as image_file:
    #         encoded_str = base64.b64encode(image_file.read()).decode('utf-8')
    #     pts_list = pts.tolist()
    #
    #     data = {
    #         'image': encoded_str,
    #         'fixedParam': {
    #             'position': pts_list,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/crop', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res0.png'))
    #
    #     # resize
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'image_size': 352,
    #         },
    #         'dynamicParam': '123',
    #     }
    #     response = self.client.post('/preprocess/resize', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     original_size = response_data['dynamicParam']['original_size']
    #     original = response_data['dynamicParam']['original']
    #     save_image(response_image, os.path.join(save_path, 'res1.png'))
    #
    #     # mean
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'mean_bgr': [103.939, 116.779, 123.68, 137.86],
    #         },
    #         'dynamicParam':{
    #             'original_size': original_size,
    #             "original": original,
    #         },
    #     }
    #     response = self.client.post('/preprocess/mean', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res2.png'))
    #
    #     # DexiNed
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'threshold': 100,
    #         },
    #         'dynamicParam': {
    #             'original_size': original_size,
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/dexiNed', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res4.png'))
    #
    #     # Houghlines
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'hougha': 40,
    #             'houghb': 20,
    #             'houghc': 30,
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #         },
    #     }
    #     response = self.client.post('/identify/houghlines', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     lines = response_data['dynamicParam']['lines']
    #     save_image(response_image, os.path.join(save_path, 'res5.png'))
    #
    #     # filterlines
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'angledown': 30,
    #             'angleup': 50,
    #             'threshold': 200,
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #             'lines': lines,
    #         },
    #     }
    #     response = self.client.post('/identify/filterlines', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     lines = response_data['dynamicParam']['lines']
    #     save_image(response_image, os.path.join(save_path, 'res6.png'))
    #
    #     # count
    #     data = {
    #         'image': response_image,
    #         'fixedParam': {
    #             'level': [16.66518, 1, 3595.05209],
    #             'map': [-0.01875, 1002.28813],
    #         },
    #         'dynamicParam': {
    #             'original': original,
    #             'lines': lines,
    #         },
    #     }
    #     response = self.client.post('/count/countlevelout', json=data)
    #     response_data = response.get_json()
    #     response_image = response_data['image']
    #     save_image(response_image, os.path.join(save_path, 'res7.png'))



if __name__ == '__main__':
    unittest.main()


























