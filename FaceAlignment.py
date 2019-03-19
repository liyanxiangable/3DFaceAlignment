import cv2
import numpy as np
import os
import path
import face_recognition
import getopt, sys

def getOriginalData(file):
    count_vertices = 0
    count_faces = 0
    original_coordinates = []
    faces_indices = []
    texture_coordinates = []
    texture_indices = []
    oc_file = open("Original_Vertices.txt", "w")
    fi_file = open("Face_Indices.txt", "w")
    tc_file = open("Texture_Coordinates.txt", "w")
    ti_file = open("Texture_Indices.txt", "w")
    for line in file.readlines():
        content = line.split(" ")
        # 顶点数据
        if content[0] == "v":
            count_vertices += 1
            coordinate = []
            for i in range(1, 4):
                num = float(content[i].replace("\n", ""))
                coordinate.append(num)
            original_coordinates.append(coordinate)
            oc_file.write(str(coordinate) + "\n")
        # 三角面片数据
        if content[0] == "f":
            count_faces += 1
            vertex_indices = []
            face_texture = []
            for i in range(1, 4):
                a = int(content[i].split("/")[0])
                b = int(content[i].split("/")[1])
                vertex_indices.append(a)
                face_texture.append(b)
            faces_indices.append(vertex_indices)
            texture_indices.append(face_texture)
            fi_file.write(str(vertex_indices) + "\n")
            ti_file.write(str(face_texture) + "\n")
        # 纹理数据
        if content[0] == "vt":
            coordinate = [float(content[1]), float(content[2].replace("\n", ""))]
            tc_file.write(str(coordinate) + "\n")
            texture_coordinates.append(coordinate)
    print("共有三角网格顶点 " + str(count_vertices) + " 个")
    print("共有三角网格面片 " + str(count_faces) + " 个")
    oc_file.close()
    fi_file.close()
    tc_file.close()
    ti_file.close()
    return np.array(original_coordinates, dtype=np.float32),\
           np.array(faces_indices, dtype=np.int32), \
           np.array(texture_indices, dtype=np.int32), \
           np.array(texture_coordinates, dtype=np.float32)

def getRoundingCoordinates(coordinates):
    rc_file = open("Rounding_Vertices.txt", "w")
    rounding_coordinates = np.zeros(coordinates.shape, dtype=np.int32)
    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            rounding_coordinates[i][j] = int(round(coordinates[i][j], 4) * 10000)
    for coordinate in rounding_coordinates:
        rc_file.write(str(coordinate) + "\n")
    rc_file.close()
    return rounding_coordinates

def getAdjustedCoordinates(coordinates, x_min, y_min):
    ac_file = open("Adjusted_Vertices.txt", "w")
    adjusted_coordinates = np.zeros(coordinates.shape, dtype=np.int32)
    print("偏移量 x : " + str(x_min) + "\ty : " + str(y_min))
    for i in range(coordinates.shape[0]):
        adjusted_coordinates[i][0] = coordinates[i][0] - x_min - 1
        adjusted_coordinates[i][1] = coordinates[i][1] - y_min - 1
        adjusted_coordinates[i][2] = coordinates[i][2]
    for coordinate in adjusted_coordinates:
        ac_file.write(str(coordinate) + "\n")
    ac_file.close()
    return adjusted_coordinates

def renderTexture(texture_coordinates, vertices_coordinates, vertices_indices,
                  texture_indices, texture_file, image):
    '''
    对图像进行着色，遍历每个三角面片，获取当前三角面片的顶点索引与贴图索引
    通过顶点索引与贴图索引获得顶点坐标与贴图坐标
    将三角形按照重心与中线分为三个小四边形并进行着色
    :param texture_coordinates:     纹理贴图坐标
    :param vertices_coordinates:    顶点坐标
    :param vertices_indices:        三角面片顶点索引
    :param texture_indices:         三角面片贴图索引
    :param texture_file:            贴图文件
    :return:
    '''
    texture = cv2.imread(texture_file, cv2.IMREAD_COLOR)
    # 获取纹理图像大小
    height, width, channels = texture.shape
    print("纹理贴图尺寸: " + str(height) + " , " + str(width) + " , " + str(channels))
    # 遍历各面
    for i in range(vertices_indices.shape[0] - 1, 0, -1):
        # 获取当前三角面片顶点索引
        index_va = vertices_indices[i][0] - 1
        index_vb = vertices_indices[i][1] - 1
        index_vc = vertices_indices[i][2] - 1
        # 获取当前三角面片顶点的贴图索引
        index_ta = texture_indices[i][0] - 1
        index_tb = texture_indices[i][1] - 1
        index_tc = texture_indices[i][2] - 1
        # 获取当前三角面片顶点坐标
        va = vertices_coordinates[index_va]
        vb = vertices_coordinates[index_vb]
        vc = vertices_coordinates[index_vc]
        # 获取当前三角面片顶点的贴图坐标
        ta = texture_coordinates[index_ta]
        tb = texture_coordinates[index_tb]
        tc = texture_coordinates[index_tc]
        # 获取贴图 BGR 值，注意贴图首先是高（行），其次是宽（列）
        ca = texture[getTexturePosition(height, 1 - ta[1]), getTexturePosition(width, ta[0])]
        cb = texture[getTexturePosition(height, 1 - tb[1]), getTexturePosition(width, tb[0])]
        cc = texture[getTexturePosition(height, 1 - tc[1]), getTexturePosition(width, tc[0])]
        # 求三角形重心坐标
        gravity_centre = []
        for j in range(3):
            gravity_centre.append(int((va[j] + vb[j] + vc[j]) / 3))
        # 求各顶点对边中点，注意此时应当舍弃 z 坐标
        ab = [int((va[0] + vb[0]) / 2), int((va[1] + vb[1]) / 2)]
        ac = [int((va[0] + vc[0]) / 2), int((va[1] + vc[1]) / 2)]
        bc = [int((vc[0] + vb[0]) / 2), int((vc[1] + vb[1]) / 2)]

        cv2.fillConvexPoly(image, np.array([[va[0], va[1]], ab, [gravity_centre[0],
                            gravity_centre[1]], ac], dtype=np.int32), ca.tolist())
        cv2.fillConvexPoly(image, np.array([[vb[0], vb[1]], ab, [gravity_centre[0],
                            gravity_centre[1]], bc], dtype=np.int32), cb.tolist())
        cv2.fillConvexPoly(image, np.array([[vc[0], vc[1]], bc, [gravity_centre[0],
                            gravity_centre[1]], ac], dtype=np.int32), cc.tolist())
    cv2.imwrite("Textured.jpg", color_image)
    return

def renderDepth(vertices_coordinates, vertices_indices, depth_image, min_depth, max_depth):
    # 生成深度图像
    # 遍历各面
    temp_depth_a = np.zeros(depth_image.shape, np.uint8)
    temp_depth_b = np.zeros(depth_image.shape, np.uint8)
    for i in range(vertices_indices.shape[0] - 1, 0, -1):
        # 获取当前三角面片顶点索引
        index_va = vertices_indices[i][0] - 1
        index_vb = vertices_indices[i][1] - 1
        index_vc = vertices_indices[i][2] - 1
        # 获取当前三角面片顶点坐标
        va = vertices_coordinates[index_va]
        vb = vertices_coordinates[index_vb]
        vc = vertices_coordinates[index_vc]
        # 计算三角面片平均深度
        mean_depth = (va[2] + vb[2] + vc[2]) / 3
        # 归一化深度为 0-255 的灰度
        scale = int((mean_depth - min_depth) / (max_depth - min_depth) * 255)
        grey_scale = [scale, scale, scale]
        cv2.fillConvexPoly(temp_depth_a, np.array([[va[0], va[1]], [vb[0], vb[1]], [vc[0], vc[1]]], dtype=np.int32), grey_scale)
    for i in range(vertices_indices.shape[0]):
        # 获取当前三角面片顶点索引
        index_va = vertices_indices[i][0] - 1
        index_vb = vertices_indices[i][1] - 1
        index_vc = vertices_indices[i][2] - 1
        # 获取当前三角面片顶点坐标
        va = vertices_coordinates[index_va]
        vb = vertices_coordinates[index_vb]
        vc = vertices_coordinates[index_vc]
        # 计算三角面片平均深度
        mean_depth = (va[2] + vb[2] + vc[2]) / 3
        # 归一化深度为 0-255 的灰度
        scale = int((mean_depth - min_depth) / (max_depth - min_depth) * 255)
        grey_scale = [scale, scale, scale]
        cv2.fillConvexPoly(temp_depth_b, np.array([[va[0], va[1]], [vb[0], vb[1]], [vc[0], vc[1]]], dtype=np.int32), grey_scale)
    for row in range(depth_image.shape[0]):
        for col in range(depth_image.shape[1]):
            front = 0
            grey_a = temp_depth_a[row][col][0]
            grey_b = temp_depth_b[row][col][0]
            if grey_a <= grey_b:
                front = grey_b
            else:
                front = grey_a
            depth_image[row][col] = front
    cv2.imwrite("Depth.jpg", depth_image)
    return

def drawTriangularMesh(vertices_indices, coordinates, image, color):
    for faces_index in vertices_indices:
        # 由索引获取坐标
        # print("三角网格索引 " + str(faces_index))
        vertex_a = coordinates[faces_index[0] - 1]
        vertex_b = coordinates[faces_index[1] - 1]
        vertex_c = coordinates[faces_index[2] - 1]
        # print("三角面片顶点坐标为 " + str(vertex_a) + "\t" + str(vertex_b) + "\t" + str(vertex_c))
        cv2.line(image, (vertex_a[0], vertex_a[1]), (vertex_b[0], vertex_b[1]), color)
        cv2.line(image, (vertex_c[0], vertex_c[1]), (vertex_b[0], vertex_b[1]), color)
        cv2.line(image, (vertex_a[0], vertex_a[1]), (vertex_c[0], vertex_c[1]), color)
    for coordinate in coordinates:
        # 注意图片的坐标是 height, width
        image[int(coordinate[1]), int(coordinate[0])] = black
    return

def getTexturePosition(length, ratio):
    p = int(np.floor(length * ratio))
    if p >= length:
        p = length - 1
    return p

def getFaceFeatures():
    #利用 FaceAlignment 获取特征坐标点
    image = face_recognition.load_image_file("./Textured.jpg")
    face_landmarks_list = face_recognition.face_landmarks(image)
    alignment_image = cv2.imread("./Textured.jpg", cv2.IMREAD_COLOR)
    alignment_coordinates = open("Alignment.txt", "w")
    if len(face_landmarks_list) >= 1:
        print("成功检测面部")
    else:
        print("未检测到面部，请核查输入文件！")
        return

    for face_landmarks in face_landmarks_list:
        # 打印此图像中每个面部特征的位置
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]
        for facial_feature in facial_features:
            print("The {} in this face has the following points: {}"
                  .format(facial_feature, face_landmarks[facial_feature]))
        for facial_feature in facial_features:
            #alignment_image[][] = face_landmarks[facial_feature]
            alignment_coordinates.write(facial_feature + ": " + str(face_landmarks[facial_feature]) + "\n")
    alignment_coordinates.close()
    return face_landmarks_list[0]

def drawLandmarks(face_landmarks, image):
    red = (0, 0, 255)
    for face_landmark in face_landmarks.values():
        for coordinate in face_landmark:
            cv2.circle(image, coordinate, 5, red, -1)
    cv2.imwrite("FaceLandMarked.jpg", image)

def landmarksDictToList(face_landmarks):
    all_coordinates = []
    for face_landmark in face_landmarks.values():
        for coordinate in face_landmark:
            all_coordinates.append(coordinate)
    return all_coordinates

def getSurroundFaces(adjusted_coordinates, vertices_indices, all_coordinates):
    landmarks_dict = {}
    faces_dict = {}
    landmark_triangles = open("Landmark_Triangles.txt", "w")
    for coordinate in all_coordinates:
        landmarks_dict.update({})
        faces_dict.update({str(coordinate): []})
    for vertices_index in vertices_indices:
        index_a = vertices_index[0] - 1
        index_b = vertices_index[1] - 1
        index_c = vertices_index[2] - 1
        va = adjusted_coordinates[index_a]
        vb = adjusted_coordinates[index_b]
        vc = adjusted_coordinates[index_c]
        for coordinate in all_coordinates:
            if cv2.pointPolygonTest(np.array([[va[0], va[1]], [vb[0], vb[1]], [vc[0], vc[1]]], dtype=np.int32),
                                    coordinate, False) >= 0:
                faces_dict[str(coordinate)].append([va, vb, vc])
    for landmark, triangle in faces_dict.items():
        landmark_triangles.write(str(landmark) + ":\t" + str(triangle) + "\n")
    landmark_triangles.close()
    refined = refineTriangleFaces(faces_dict)
    return refined

def refineTriangleFaces(faces_dict):
    refined = {}
    for landmark, triangles in faces_dict.items():
        if len(triangles) == 1:
            refined.update({str(landmark): triangles[0]})
        elif len(triangles) == 0:
            refined.update({str(landmark): []})
        else:
            depth = []
            for triangle in triangles:
                z = triangle[0][2] + triangle[1][2] + triangle[2][2]
                depth.append(z)
            index = np.argmin(depth)
            refined.update({str(landmark): triangles[index]})
    refined_file = open("refined.txt", "w")
    for k, v in refined.items():
        refined_file.write(str(k) + ":\t" + str(v) + "\n")
    refined_file.close()
    return refined

def getDistance(feature_a, index_a, feature_b, index_b, landmark_triangles, feature_landmarks, xy = True):
    # to be continue
    distance = 0
    return distance

def getGlassesDistanceInformation(face_landmarks):
    information_file = open("manInformation.txt", "w")
    distances = []
    a = (face_landmarks['chin'][16][0] - face_landmarks['chin'][0][0])/10
    b = (face_landmarks['right_eye'][3][0] - face_landmarks['left_eye'][0][0])/10
    c = (face_landmarks['right_eye'][1][0] + face_landmarks['right_eye'][2][0] - face_landmarks['left_eye'][1][0] - face_landmarks['left_eye'][2][0]) / 20
    d = (face_landmarks['right_eye'][1][0] - face_landmarks['left_eye'][3][0])/10
    h = (face_landmarks['right_eye'][2][1] - face_landmarks['right_eyebrow'][2][1])/10
    f = (face_landmarks['nose_bridge'][3][1] - face_landmarks['nose_bridge'][0][1])/10
    g = round(0.7 * h, 1)
    e = round(2.2 * f, 1)
    distances.append(a)
    distances.append(b)
    distances.append(c)
    distances.append(d)
    distances.append(e)
    distances.append(f)
    distances.append(g)
    distances.append(h)
    print("配镜所需参数依次为...")
    for distance in distances:
        information_file.write(str(distance) + "\n")
        print(str(distance) + " (mm)")
    information_file.close()
    return distances

def cutPointCloud(old_obj, threshold):
    # 对 z 轴设定阈值，以消除 z 坐标重复
    # obj 文件包含 顶点 v, 纹理 vt, 三角面片 f（f中存放顶点与纹理的索引） 信息
    # 通过 z 坐标过滤顶点 v，此时包含顶点 v 的三角面片也应当去除
    # 若去除三角面片，则其含有的纹理信息也应当去除
    # 首先遍历原始 obj 文件，获取顶点列表、纹理列表、三角面片索引信息列表
    # 遍历顶点列表，获取要去除的顶点的索引
    # 遍历三角面片索引信息列表，将含有要去除的索引的三角面片信息去除
    # 记录所去除的面的纹理坐标，遍历纹理列表并去除
    # 最终形成新的文件
    cut_obj = open("cut_obj.obj", "w")
    original_vertices = []
    original_texture = []
    original_faces = []
    remove_vertices_indices = set()
    remove_face_indices = set()
    remove_texture_indices = set()
    for line in old_obj.readlines():
        content = line.split(" ")
        # 顶点数据
        if content[0] == "v":
            original_vertices.append(line)
        # 三角面片数据
        if content[0] == "f":
            original_faces.append(line)
        # 纹理数据
        if content[0] == "vt":
            original_texture.append(line)
    old_obj.close()
    print("未裁剪文件 顶点:\t" + str(len(original_vertices)) +
          "\t纹理:\t" + str(len(original_texture)) + "\t三角面:\t" + str(len(original_faces)))
    for index, line in enumerate(original_vertices):
        content = line.split(" ")
        if float(content[3]) > threshold:
            remove_vertices_indices.add(index)
        else:
            continue
    for index, line in enumerate(original_faces):
        content = line.split(" ")
        for i in range(1, 4):
            v = int(content[i].split("/")[0])
            vt = int(content[i].split("/")[1])
            if v - 1 in remove_vertices_indices:
                remove_face_indices.add(index)
                remove_texture_indices.add(vt - 1)
    print("需去除点:\t" + str(len(remove_vertices_indices)) + "\t去除纹理:\t" +
          str(len(remove_texture_indices)) + "\t去除网格:\t" + str(len(remove_face_indices)))
    # 注意，f 信息中储存的索引是从 1 开始的
    # 所以上述代码索引都是从 0 开始的
    for index, line in enumerate(original_vertices):
        if index not in remove_vertices_indices:
            cut_obj.write(line)
        else:
            # 本行仅是行号补位作用，无意义
            cut_obj.write("v 0.042966 -0.094774 0.43439\n")
    for index, line in enumerate(original_texture):
        if index not in remove_texture_indices:
            cut_obj.write(line)
        else:
            cut_obj.write("vt 0.14193 0.20604\n")
    for index, line in enumerate(original_faces):
        if index not in remove_face_indices:
            cut_obj.write(line)
    cut_obj.close()


opts, args = getopt.getopt(sys.argv[1:], "o:t:")
obj_file_path = r"C:\Users\liyanxiang\Desktop\head\resUnWarpMesh.obj"
texture_file_path = r"C:\Users\liyanxiang\Desktop\head\clonedBlur.png"
for opt, value in opts:
    print("输入文件 : " + value)
    if opt == "-o":
        print("obj 文件路径为 : " + value)
        obj_file_path = value
    if opt == "-t":
        print("texture 文件路径为 : " + value)
        texture_file_path = value
obj_file = open(obj_file_path, "r")
threshold = 0.45
cutPointCloud(obj_file, threshold)
cut_obj = open("cut_obj.obj", "r")
# original_coordinates 原始顶点坐标
# vertices_indices 三角面片顶点索引
# texture_indices 三角面片贴图索引
# texture_coordinates 贴图坐标
original_coordinates, vertices_indices, texture_indices, texture_coordinates = getOriginalData(cut_obj)
cut_obj.close()
rounding_coordinates = getRoundingCoordinates(original_coordinates)
x_max = np.max(rounding_coordinates[:, 0])
x_min = np.min(rounding_coordinates[:, 0])
y_max = np.max(rounding_coordinates[:, 1])
y_min = np.min(rounding_coordinates[:, 1])
z_max = np.max(rounding_coordinates[:, 2])
z_min = np.min(rounding_coordinates[:, 2])
print("X max: " + str(x_max) + "\t\tX min: " + str(x_min) + "\nY max: " + str(y_max) +
      "\t\tY min: " + str(y_min) + "\nZ max: " + str(z_max) + "\t\tZ min: " + str(z_min))

height = int(y_max - y_min)
width = int(x_max - x_min)
depth = int(z_max - z_min)
print("图片高度为: " + str(height))
print("图片宽度为: " + str(width))
adjusted_coordinates = getAdjustedCoordinates(rounding_coordinates, x_min, y_min)
color_image = np.zeros((height, width, 3), np.uint8)
depth_image = np.zeros((height, width, 3), np.uint8)
white = (255, 255, 255)
green = (0, 255, 0)
black = (0, 0, 0)
color_image[:, :] = white
depth_image[:, :] = white

faces_coordinates_file = open("Faces_Coordinates.txt", "w")
'''
for coordinate in rounding_coordinates:
    blank_image[int(coordinate[1] - y_min - 1)][int(coordinate[0] - x_min - 1)] = black
'''

drawTriangularMesh(vertices_indices, adjusted_coordinates, color_image, green)
cv2.imwrite("Triangular.jpg", color_image)

renderTexture(texture_coordinates, adjusted_coordinates, vertices_indices, texture_indices, texture_file_path, color_image)

drawTriangularMesh(vertices_indices, adjusted_coordinates, color_image, green)
cv2.imwrite("TextureCombineTriangle.jpg", color_image)

renderDepth(adjusted_coordinates, vertices_indices, depth_image, z_max, z_min)

face_landmarks = getFaceFeatures()
drawLandmarks(face_landmarks, color_image)

all_coordinates = landmarksDictToList(face_landmarks)
landmarks_faces = getSurroundFaces(adjusted_coordinates, vertices_indices, all_coordinates)

distances = getGlassesDistanceInformation(face_landmarks)

faces_coordinates_file.close()
cv2.imshow("Created", color_image)
cv2.imwrite("Created.jpg", color_image)
print("image saved!")
#cv2.waitKey(0)
#cv2.destroyWindow()
