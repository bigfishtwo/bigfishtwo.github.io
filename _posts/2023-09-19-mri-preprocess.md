---
title: MRI 预处理
date: 2023-06-28 +0800
categories: [医学相关, 医学图像]
tags: [MRI]
---

## 1. MRI 图像是什么

如果抛开MRI成像的物理原理，用计算机的角度去理解MRI图像，那么MRI图像是真实世界内的某样物体通过某种方式被采样到了计算机上。

假设真实的空间内有一个确定的原点和坐标轴，一个物理尺寸（140，180）（单位未知）的物体，我们要对它进行成像（采样），采样的原点在真实空间内的坐标为（60，70），采样方向为 [1,0] 和 [0,1]，采样间隔为20和30（真实空间单位）。采样位置的值为物体在这个位置的某种物理属性，采样之后得到数字矩阵即图像。

MRI 图像通常为三维图像，一个二维图像上的像素对应到三维图像上被称为 **体素**。体素空间由三个体素轴定义，其中（0，0，0）是阵列中第一个体素的中心，轴上的单位是体素。因此，体素坐标被定义在一个叫做体素空间的参考空间中。

![SimpleITK Images and Resampling](https://upload-images.jianshu.io/upload_images/16576979-1ce8163248f495d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/512)

 RAS 坐标系，通常指三个体素轴的方向分别是从左到右（Right），从后向前（Anterior），从低到高（Superior）。 RAS 也被写作 RAS+，意思是右、前、上为正方向。

根据上面的介绍，如果我们想要把一个真实世界的物体采样到计算机里，并记录下它在真实世界的位置、方向、物理属性等，需要：
- 原点：在真实世界的位置
- 坐标轴的方向：与真实世界坐标轴的夹角
- 图像大小：每个维度上像素的个数
- Spacing：采样的物理间隔
- 数据类型

## 2. MRI 图像的种类

让我们暂时用“种类”这个略显外行的词来指不同的成像序列得到的 MRI 图像，包括 T1WI， T2WI，DWI（dMRI），fMRI等，根据成像序列不同，它们在不同位置会有不同的明暗变化。例如，脂肪在T1加权图像上显得很亮，而在T2加权图像上显示为暗色。

## 3. MRI 图像的存储格式


DICOM（digital imaging and communications in medicine） 不只是一种数据储存格式，还是一种通信标准。DICOM提供了一种封装方式，将信息对象定义的一个服务对象对（SOP）实例以数据集的形式封装在一个文件中。DICOM标准文件由DICOM文件头和DICOM数据集两部分组成。

NIFTI（神经影像信息技术）的扩展名是（.nii），包含了头文件及图像资料。同时NIfTI也可使用独立的图像文件（.img）和头文件（.hdr）。使用压缩格式（.nii.gz）可以节省空间，但是会降低读取速度。

NRRD（近原始栅格数据），和 NIFTI相似，有包含头文件和使用单独头文件两种形式。与DICOM 相比，它的主要优点是 NRRD 文件是匿名的并且不包含敏感的患者信息。此外，NRRD 文件可以将一次医学扫描的所有数据存储在单个文件中，便于传输。

### 读写速度对比
NIFTI 具有固定的 348 字节二进制标头，这使让NIFTI 读起来非常简单，但是难写，因为它规定了图像数据维度的顺序，前三个维度必须是空间维度，第四个维度是时间维度，维度 5,6 ,7 由用户决定。
NRRD 是基于文本的，容易编写，因为可以轻松地描述想要的维度顺序。代价是创建阅读器更难，因为需要兼顾图像尺寸。 NRRD 的优雅之处在于标头的灵活性，允许创建一个小的标头来描述不同格式的复杂图像，例如，可以编写一个 NRRD 格式的 nhdr 文件来描述大多数未压缩的 DICOM 图像，从而轻松支持DICOM。
就速度而言，两者的解析速度都非常快，而且头文件的小成本被大图像淹没了。

NIFTI 格式包含许多特定于神经成像的细节，并且非常严格（固定的二进制标头；不灵活的维度）。这使得 NIFTI 成为一种很好的神经成像文件格式，但不太适合作为通用图像文件格式。

从nrrd或nifti加载一个三维阵列和从npy/npz文件加载一样快。所有这些格式都可以选择使用压缩，这通常可以将所需的磁盘空间减少一半，并将加载时间增加5-10倍。从DICOM文件中加载（每个文件单帧）比从nrrd/nifty/npy中加载相同体积的文件（不考虑压缩）要多花大约100倍的时间。

通常会使用的所有三种数据表现形式： DICOM、研究文件格式（nrrd、nifti）和张量文件（numpy npy/npz 阵列或其他专门的超立方体存储）。每个文件格式都有其非常重要的作用：
DICOM：长期档案格式，保留所有元数据和UID；可跨项目和机构使用；非常缓慢和复杂。
研究文件格式（nrrd, nifti）：保留了基本的元数据（即图像的几何形状）；适合于代表一个队列，在一个项目中可跨实验使用，由一小组合作者使用；快速和简单，只适合图像。
Numpy数组（npy, npz）：不保留元数据，因此只适用于单个实验，由一个或几个紧密合作的研究人员使用；快速且非常简单，支持任意尺寸，可以表示非图像矢量数据。

一些处理 MRI 的 python 库：[nilearn](https://nilearn.github.io/dev/auto_examples/00_tutorials/index.html), [nibabel](https://nipy.org/nibabel/coordinate_systems.html), [dipy](https://dipy.org/documentation/1.7.0/examples_built/#preprocessing), [SimpleITK](https://simpleitk.org/TUTORIAL/)


### DICOM 
read

```python
# SimpleITK
# Read a DICOM series
data_directory = os.path.dirname(fdata("CIRS057A_MR_CT_DICOM/readme.txt"))
series_ID = "1.2.840.113619.2.290.3.3233817346.783.1399004564.515"

# Use the functional interface to read the image series.
original_image = sitk.ReadImage(
    sitk.ImageSeriesReader_GetGDCMSeriesFileNames(data_directory, series_ID)
)

# Select a specific DICOM series from a directory and only then load user selection.
series_IDs = sitk.ImageSeriesReader_GetGDCMSeriesIDs(data_directory)
...
for series in series_IDs:
    series_file_names[series] = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
        data_directory, series)
img = sitk.ReadImage(series_file_names[selected_series])
```

write
```python
# SimpleITK
# write as mha file
sitk.WriteImage(original_image, os.path.join(OUTPUT_DIR, "3DImage.mha")) 

# write as a series of JPEG
# 将卷写成一系列 JPEG。 WriteImage 函数接收一个卷和一个图像名称列表，并根据 z 轴写入该卷。
# 对于可显示的结果，我们需要重新调整图像强度（默认为 [0,255]），
# 因为 JPEG 格式需要转换为 UInt8 像素类型。

sitk.WriteImage(
    sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8),
    [
        os.path.join(OUTPUT_DIR, "slice{0:03d}.jpg".format(i))
        for i in range(img.GetSize()[2])
    ],
)
```


### NIFITI
```python
# SimpleITK
# read 
image_file = sitk.ReadImage('temp_image.nii', sitk.sitkFloat32)

#write 
mask_file.SetOrigin(image_file.GetOrigin())
mask_file.SetSpacing(image_file.GetSpacing())
mask_file.SetDirection(image_file.GetDirection())
sitk.WriteImage(mask_file, 'temp_mask.nii')
```

```python
# dipy
from dipy.io.image import load_nifti, save_nifti
data, affine = load_nifti(image_path)
save_nifti('temp_image.nii', den, affine)

# nibabel
import nibabel as nib
temp_file = nib.load('temp_image.nii')
nib.save(temp_file, 'temp_image.nii')
```

### 与numpy array的转换：

**SimpleITK** 和 numpy 索引访问顺序相反

- **SimpleITK2Numpy：**
    - GetArrayFromImage()：返回图像数据的副本。然后可以自由修改数据，因为它对原始 SimpleITK 图像没有影响
    - GetArrayViewFromImage()：返回图像数据的视图，这对于以内存高效方式显示很有用。如果删除原始 SimpleITK 图像，将无法修改数据并且视图将无效。
- **Numpy2SimpleITK**
GetImageFromArray()：返回一个 SimpleITK 图像，原点设置为零，所有维度的间距设置为 1，方向余弦矩阵设置为恒等。强度数据是从 numpy 数组中复制的。在大多数情况下，需要**设置适当的元数据值。**
    
    ```python
    # example
    
     # in case you are using 3D data or doing the registration
        if image_file.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            image_file.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        
        # without this, might cause errors in some files
        mask_file.SetOrigin(image_file.GetOrigin())
        mask_file.SetSpacing(image_file.GetSpacing())
        mask_file.SetDirection(image_file.GetDirection())
    ```
    

nibabel

```python
import nibabel as nib
epi_img = nib.load('downloads/someones_epi.nii.gz')
epi_img_data = epi_img.get_fdata()
```

更快的速度
如果想在训练过程中读取数据更快，可以考虑把数据处理好之后存成hdf5格式，需要到 h5py ，[dataloader读入hdf5格式数据](https://www.jianshu.com/p/ee4b76b32779)。


## 4. Resampling

重采样作为动词的含义是对图像进行取样的行为，而图像本身就是对原始连续信号的取样。

一般来说，SimpleITK 中的重采样涉及四个部分:

- Image - 我们重采样的图像，在坐标系中给出
- 重采样网格 - 坐标系中给定的规则点网格f, 将被映射到坐标系m
- Transform-从坐标系映射的变换
- Interpolator-获取坐标系m中任意点强度值的方法,来自图像定义的点的值。

虽然 SimpleITK 提供了大量的插值方法，但最常用的两种是 sitkLinear 和 sitkNearestNeighbor。前者用于大多数插值任务，是精度和计算效率之间的折衷。后者用于对表示分割的标记图像进行插值，它是唯一不会在结果中引入新标签的插值方法。
SimpleITK 的程序 API 提供了三种执行重采样的方法，不同之处在于指定重采样网格的方式：

```python
Resample(const Image &image1, Transform transform, InterpolatorEnum interpolator, double defaultPixelValue, PixelIDValueEnum outputPixelType)
Resample(const Image &image1, const Image &referenceImage, Transform transform, InterpolatorEnum interpolator, double defaultPixelValue, PixelIDValueEnum outputPixelType)
Resample(const Image &image1, std::vector< uint32_t > size, Transform transform, InterpolatorEnum interpolator, std::vector< double > outputOrigin, std::vector< double > outputSpacing, std::vector< double > outputDirection, double defaultPixelValue, PixelIDValueEnum outputPixelType)
```

常见错误：

重新采样后以空（全黑）图像结束的情况并不少见。这是因为：

- 对重采样网格使用错误的设置。
- 使用变换的逆。这是一个相对常见的错误，可以通过调用转换 GetInverse 方法轻松解决。

## 5. 预处理流程

参考：
https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need
https://github.com/ZHAN-GAN/MRI-Preprocess/tree/main
https://carpentries-incubator.github.io/SDC-BIDS-sMRI/02-Image_Cleanup/index.html

## 6. Dataloader

这里放一个伪代码
```
class MRIDataset(torch.utils.data.Dataset):

    def __init__(self, args):
         # 一些初始化
    def open_hdf5(self):
        # 如果用 HDF5
        self.img_hdf5 = h5py.File('data.hdf5', 'r')
    def load_nii_file(self, nii_image):
        # 如果读入 NIFTI 
        image = sitk.ReadImage(nii_image)
        image_array = sitk.GetArrayFromImage(image)
        return image_array
    def __getitem__(self, index):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        img = self.img_hdf5[index][:]
        # 或者读入预处理好的 NIFTI
        # img = self.load_nii_file(files[index])
        if transformation:
            img = transformation(img)
        return (image, label)
```

## 参考
1. https://github.com/SimpleITK/TUTORIAL/blob/main/01_spatial_transformations.ipynb
2. https://www.embodi3d.com/blogs/entry/341-how-to-create-an-nrrd-file-from-a-dicom-medical-imaging-data-set/
3. https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
4. https://nipy.org/nibabel/dicom/dicom_intro.html#dicom-data-format