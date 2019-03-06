# 3DFaceAlignment
三维面部三角网格模型的特征点定位



将三维面部模型降维平面并定位特征点，之后映射到三维，以提取面部信息。

### 使用方法

``` shell
python FaceAlignment.py -o ./resources/objModel.obj -t ./resources/texture.png
```

### 效果

三角网格模型：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/SideFace.jpg)

纹理贴图：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/clonedBlur.jpg)

三角网格映射：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/Triangular.jpg)

贴图映射：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/Textured.jpg)

贴图特征点定位：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/TextureCombineTriangle.jpg)

特征点存在于对应三角网格内：

![](https://github.com/liyanxiangable/3DFaceAlignment/blob/master/resources/FaceLandMarked.jpg)
