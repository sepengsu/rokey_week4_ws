import numpy as np
import json
'''
point는 각각 [x,y]로 이루어진 list
'''
from myutils import rad2deg, dict_to_array

class PointSort:
    def __init__(self,usb, chipset):
        '''
        usb: usb point (detect object)
        chipset: chipset point (detect object)
        '''
        self.usb = np.array(usb)
        self.chipset = np.array(chipset)
        
    def __call__(self,points):
        '''
        points: hole points
        returns: sorted hole points or -1 
        -1: points가 정렬이 불가능한 경우
        '''
        self.points = points
        self.vector()
        self.types_make()
        if self.check():
            self.sorting()
            return self.sorted_points
        else:
            return -1

    def vector(self):
        '''
        usb: usb point
        chipset: chipset point
        1. usb와 chipset의 vector를 구함
        2. chipset - usb
        3. vector를 구함
        4. vector에 -pi/2를 더한 vector를 구함
        5. vector에 pi/2를 더한 vector를 구함
        6. vector에 pi를 더한 vector를 구함
        7. 각각에 대한 angle을 구함
        '''
        usb = self.usb
        chipset = self.chipset
        vector_up = usb - chipset
        vector_down = chipset - usb
        vector_right = np.array([vector_up[1],-vector_up[0]]) # -pi/2
        vector_left = np.array([-vector_up[1],vector_up[0]]) # pi/2
        self.angle_up = rad2deg(np.arctan2(vector_up[1],vector_up[0])) # 0
    
    def direction(self,angle):
        '''
        rigtht< angle < up: 1
        up< angle < left: 2
        left< angle < down: 3
        down< angle < right: 4
        '''
        diff_angle = angle - self.angle_up
        if diff_angle < 0:
            diff_angle += 360
        if 0 <= diff_angle < 90:
            return 1
        elif 90 <= diff_angle < 180:
            return 2
        elif 180 <= diff_angle < 270:
            return 3
        elif 270 <= diff_angle < 360:
            return 4
    
    def types_make(self):
        point1,point2,point3,point4 = self.points
        vector1 = point1 -self.chipset
        vector2 = point2 -self.chipset
        vector3 = point3 -self.chipset
        vector4 = point4 -self.chipset
        angle1 = rad2deg(np.arctan2(vector1[1],vector1[0]))
        angle2 = rad2deg(np.arctan2(vector2[1],vector2[0]))
        angle3 = rad2deg(np.arctan2(vector3[1],vector3[0]))
        angle4 = rad2deg(np.arctan2(vector4[1],vector4[0]))
        self.types = [self.direction(angle1),self.direction(angle2),self.direction(angle3),self.direction(angle4)]

    def check(self):   
        if set(self.types) != {1,2,3,4}:
            print(set(self.types))
            return False
        else:
            return True
    def sorting(self):
        self.sorted_points = []
        for i in range(1,5):
            index = self.types.index(i)
            self.sorted_points.append(self.points[index])
    


        

class PointRotate:
    def __init__(self,threshold):
        '''
        threshold: threshold value
        '''
        self.threshold = threshold
        self.point_standard = dict_to_array(ST_POINT)

    def __call__(self, points_predicted):
        '''
        point 순서 
        0:"BOOTSEL"  
        1: "CHIPSET"  
        2: "HOLE1"
        3: "HOLE2"
        4: "HOLE3"
        5: "HOLE4"
        6: "OSCILLATOR"
        7: "RASPBERRY PICO"
        8: "USB"
        points_predicted: Predicted points
        return : True or False
        '''
        points_predicted = dict_to_array(points_predicted)
        rmse = self.point_compare(self.point_standard, points_predicted)
        result = self.threshold_check(rmse, self.threshold)
        return result
    
    def align_coordinates(self,coords1, coords2):
        '''
        coords1: Original coordinates
        coords2: Predicted coordinates
        coords1을 2에 맞추기 위해 회전 및 스케일링
        '''

        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        coords1_centered = coords1 - center1
        coords2_centered = coords2 - center2

        # 스케일 계산 및 정규화
        scale1 = np.sqrt(np.sum(coords1_centered**2) / coords1.shape[0])
        scale2 = np.sqrt(np.sum(coords2_centered**2) / coords2.shape[0])
        coords1_normalized = coords1_centered / scale1
        coords2_normalized = coords2_centered / scale2

        # 회전 계산 (SVD)
        H = coords1_normalized.T @ coords2_normalized
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt

        # 변환 적용
        coords1_transformed = scale2 * (coords1_normalized @ R.T) + center2
        return coords1_transformed

    # Test
    def point_compare(self,points_standard, points_predicted):
        '''
        box_point_o: Original box points (8,2)
        box_point_p: Predicted box points (8,2)
        point_o는 8개 좌표로 생각
        '''
        points_standard = np.array(points_standard)
        points_predicted = np.array(points_predicted)
        # 좌표 정렬
        points_predicted_aligned = self.align_coordinates(points_predicted, points_standard)
        
        # RMSE 계산
        rmse = np.sqrt(np.mean((points_standard - points_predicted_aligned)**2))
        return rmse

    def threshold_check(self,rmse, threshold):
        '''
        rmse: RMSE value
        threshold: Threshold value
        '''
        if rmse < threshold:
            return True
        else:
            print(rmse)
            return False
        

def _test():
    # path = r'C:\Users\na062\Desktop\week4\project\data\st.json'
    path =r="/home/rokey/rokey_week4_ws/project/data/st.json"
    OBJECT_LIST =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY PICO","HOLE"]
    with open(path, 'r') as f:
        data = json.load(f)
    objs = data['annotation_result']['objects']
    box_dict = dict()
    # print(objs)
    hole_num = 0
    for obj in objs:
        name = obj['class_name']
        if name == "HOLE":
            hole_num += 1
            name = f"HOLE{hole_num}"
        x,y = obj['annotation']['coord']['x'],obj['annotation']['coord']['y']
        box_dict[name] = [x,y]
    ps = PointSort(box_dict['USB'],box_dict['CHIPSET'])

    sortings = ps([box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4']]) 
    if sortings == -1:
        raise Exception("sort error")
    box_dict['HOLE1'],box_dict['HOLE2'],box_dict['HOLE3'],box_dict['HOLE4'] = sortings
    return box_dict

ST_POINT = _test()