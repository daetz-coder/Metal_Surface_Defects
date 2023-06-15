说明：下面仅以[Pytorch CNN Transfer Learning: Image Classifier ](https://www.kaggle.com/code/gurpreetmeelu/pytorch-cnn-transfer-learning-image-classifier)

关于金属表面缺陷检测类的实践项目为例介绍如何创建虚拟环境以及在JuypterLab中使用对应的内核

# 一、创建环境，依赖导入

## 1、创建虚拟环境

创建虚拟环境可以帮助您在不同项目之间隔离Python包的安装和版本，以下是创建虚拟环境的一种常见方法：

安装虚拟环境工具：首先，您需要安装`virtualenv`或`conda`等虚拟环境管理工具。如果您使用的是`pip`，可以运行以下命令安装`virtualenv`：

```
pip install virtualenv
```

![image-20230615123149054](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151231459.png)

创建虚拟环境：进入您希望创建虚拟环境的目录，并运行以下命令来创建虚拟环境：

```bash
virtualenv env_ids
```



![image-20230615123240023](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151232080.png)

这将在当前目录下创建一个名为`env`的新虚拟环境文件夹。

激活虚拟环境：根据您所使用的操作系统，激活虚拟环境的命令略有不同：

- 在 Windows 系统上，运行以下命令：

```
env\Scripts\activate
```

![image-20230615123335974](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151233087.png)

在 macOS/Linux 系统上，运行以下命令：

```
source env_ids/bin/activate
```

激活虚拟环境后，您会注意到命令提示符发生了变化，显示出虚拟环境的名称。

在虚拟环境中安装依赖：激活虚拟环境后，您可以使用`pip`安装所需的Python包，例如：

```
pip install tensorflow
```

这将在虚拟环境中安装TensorFlow包。

使用虚拟环境：在激活虚拟环境的状态下，您可以运行和管理您的项目，并确保它们使用虚拟环境中的正确Python包和版本。

## 2、安装对应的依赖

![image-20230615123647406](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151236458.png)



![image-20230615123446709](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151234775.png)



## 3、设置jupyterLab的内核

要在JupyterLab中使用您创建的虚拟环境，您需要将虚拟环境添加为JupyterLab的内核。以下是一种常见的方法：

1. 激活虚拟环境：首先，在命令行中激活您的虚拟环境。根据您的操作系统，可以使用以下命令之一：

   在 Windows 上：

```
env\Scripts\activate
```

在 macOS/Linux 上：

```
source env/bin/activate
```

安装 ipykernel：确保在虚拟环境中安装了`ipykernel`包。可以使用以下命令安装：

```
pip install ipykernel
```

![image-20230615123920621](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151239794.png)

添加虚拟环境到 JupyterLab：将虚拟环境添加为JupyterLab的内核，使用以下命令：

```
python -m ipykernel install --user --name=env
```

这将在JupyterLab中创建一个名为`env`的内核。

```
这是一个用于在 Jupyter Notebook 中安装 IPython 内核的命令。让我为您解释每个参数的含义：

    -m ipykernel: 这告诉 Python 解释器运行 ipykernel 模块。ipykernel 是用于支持 Jupyter Notebook 内核的模块。

    install: 这是 ipykernel 模块的一个子命令，用于安装 IPython 内核。

    --user: 这个参数告诉安装程序将内核安装到当前用户的主目录下，而不是系统范围内安装。这样做可以避免对系统进行更改，仅限于当前用户。

    --name=env: 这个参数指定内核的名称为 "env"。您可以将其替换为您希望的任何其他名称。内核名称用于在 Jupyter Notebook 中识别和选择特定的内核。

综上所述，该命令的目的是在 Jupyter Notebook 中安装一个名为 "env" 的 IPython 内核，并将其安装到当前用户的主目录下。如果您的当前虚拟环境是 "env_ids"，那么您可以将 --name 参数设置为 "env_ids"，以与您的环境名称一致。例如：--name=env_ids。
python -m ipykernel install --user --name=env_ids
```

![image-20230615124114911](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151241959.png)

启动 JupyterLab：启动JupyterLab，可以在命令行中运行以下命令：

```
jupyter lab
```

![image-20230615124157243](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151241354.png)

JupyterLab将在默认浏览器中打开。

在 JupyterLab 中选择虚拟环境：在JupyterLab的界面中，点击右上角的"Kernel"选项，在下拉菜单中选择您创建的虚拟环境（即`env`）。

现在，您可以在JupyterLab中使用该虚拟环境作为内核来运行和编辑Notebooks。确保在运行Notebooks之前选择正确的内核。

请注意，如果您已经在JupyterLab中打开了一个Notebook，您需要重新启动Notebook并选择虚拟环境的内核，使其生效。

## 4、安装所需依赖

![image-20230615124324363](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151243416.png)

```
!pip install matplotlib
```

![image-20230615124419363](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151244423.png)

![image-20230615124445111](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151244157.png)

```
!pip install seaborn
```

![image-20230615124521425](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151245499.png)

```
!pip install scikit-learn pillow torch torchvision
```

![image-20230615124824368](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151248454.png)

![image-20230615124929830](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151249883.png)



# 二、项目复现

## 1、数据下载

官方数据集地址：https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data

![image-20230615125330033](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151253156.png)

![image-20230615125117807](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151251879.png)

![image-20230615125351628](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151253732.png)

解压

```
unzip archive.zip
```

![image-20230615125429850](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151254957.png)

![image-20230615125747780](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151257847.png)

## 2、训练结果

![image-20230615131118861](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151311944.png)

![image-20230615131139904](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202306151311969.png)