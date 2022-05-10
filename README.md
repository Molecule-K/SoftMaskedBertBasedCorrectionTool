# SoftMaskedBertBasedCorrectionTool

基于SoftMasked-BERT的文本自动纠错应用

## 数据准备
1. 从 [http://nlp.ee.ncu.edu.tw/resource/csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)下载SIGHAN数据集
2. 解压上述数据集并将文件夹中所有 ''.sgml'' 文件复制至 datasets/csc/ 目录
3. 复制 ''SIGHAN15_CSC_TestInput.txt'' 和 ''SIGHAN15_CSC_TestTruth.txt'' 至 datasets/csc/ 目录
4. 下载 [https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) 至 datasets/csc 目录
5. 请确保以下文件在 datasets/csc 中
    ```
    train.sgml
    B1_training.sgml
    C1_training.sgml  
    SIGHAN15_CSC_A2_Training.sgml  
    SIGHAN15_CSC_B2_Training.sgml  
    SIGHAN15_CSC_TestInput.txt
    SIGHAN15_CSC_TestTruth.txt
    ```


## 环境准备
1. python3.7
2. 终端在主目录下运行`pip install -r requirements.txt`
3. 请将工程路径添加为环境变量


## 使用方式

在终端下运行如下命令：
`python tools/run.py`

您将会看到如下界面：
```
欢迎使用自动文本纠错应用,下面我将对本工具的使用进行一个简单的介绍：
本工具主要分为训练、纠错（预测）和模型迁移三个部分
    训练部分：
        如果您有意向使用本工具的训练功能，请在下面的模式选择中输入1；
    纠错部分：
        如果您想要使用该模型进行预测请在模式选择中输入2；
    模型迁移部分：
        如果您仅仅需要将训练好的模型权重提取出来方便其他项目使用，请在模式选择中输入3；
    当您希望退出该工具时，请在模式选择中输入0；
祝您使用愉快，希望有帮到您。
```

在模式1下，如果您完成了数据准备，那么模型会自动开始进行训练。这一过程耗时较长，请耐心等待。
训练结束后您可以在checkpoints/SoftMaskedBert/下找到您训练的模型。
另外，配置文件的参数可以在configs/csc下找到，如有其他需求，可根据需要自行调整配置文件中的参数。

在模式2下，如果您希望读入txt文件中的语句进行纠错，请确保该文件保存为datasets/test/test.txt，然后按照引导完成选择，输出的纠错文本将会保存在result/result.txt中。
其中test.txt的格式为：
```
如果你缺钱的话尽管和我说，我会借给拟或者一起想别的办法。
在美国，大学生上班是很正常地是，因为美国地大学制度使得选课系统十分开饭。
社会上需要的人才既要有能力又要有经验，当然，董事听指挥也是很重要的。
你还年轻呢，有计算机专场，中英文都通，一定会有公司需要你这样的人才。
```
与此同时，如果您只需要实时输入文本进行纠错，也请按照引导进行选择。注意，这种情况下纠错的语句会被直接输出到终端的屏幕上，并不会保存到文件中。

在模式3下，提供了一种将模型权重导出的方式，导出的权重可以被pycorrector或者transformer引用，它将被存放在checkpoints/SoftMaskedBert下。
