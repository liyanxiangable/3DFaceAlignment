# 3DFaceAlignment

三维面部三角网格模型的特征点定位

Feature point location of 3d facial triangular mesh model

3D Face Alignment

原材料：面部 obj 格式模型以及对应纹理贴图

将三维面部模型降维平面并定位特征点，之后映射到三维，以初步提取面部参数。

### 依赖
``` shell
python 3.X
numpy
opencv-python
dlib
facce_recognition (或自定义部特征点定位模型)
```

### 使用方法

``` shell
python FaceAlignment.py -o ./resources/yourObjModel.obj -t ./resources/yourTextureImage.png
```

### 效果

三角网格模型：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/SideFace.jpg)

纹理贴图：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/clonedBlur.png)

三角网格映射：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/Triangular.jpg)

贴图映射：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/Textured.jpg)

贴图与三角网格映射：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/TextureCombineTriangle.jpg)

特征点存在于对应三角网格内：
图中红色点即面部特征点定位

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/FaceLandMarked.jpg)
