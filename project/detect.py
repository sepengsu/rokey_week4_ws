import cv2
def show(image,result):
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    object_list =["BOOTSEL","USB","CHIPSET","OSCILLATOR","RASPBERRY PICO","HOLE"]
    color_dict = {obj: color for obj, color in zip(object_list, color_list)}
    class_count = {class_name: 0 for class_name in object_list}
    objs = result
    for obj in objs:
        name = obj["class"]
        color = color_dict[name]
        class_count[name] += 1
        x1, y1, x2, y2 = obj["box"]
        start_point = (x1, y1)
        end_point = (x2, y2)
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        text = obj["class"]
        position = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        
    # cv2.rectangle(image, (0, 0), (200, 20*(len(class_count)+1)), (255, 255, 255))
    for index, (class_name, count) in enumerate(class_count.items()):
        text = f"{class_name}: {count}"
        position = (20, 20*(index+1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = color_dict[class_name]
        thickness = 2
        # cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()