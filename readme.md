由于摄像头的角度会影响到模型的使用，所以，推荐老师或助教花上20分钟左右，
设计自己的手势，亲自训练一下模型。

流程：
1. 先用get_samples.py文件自己收集数据集，推荐简单背景下收集图片。
2. 将收集好的图片文件夹放进network.py文件生成的文件夹下，输入种类及训练次数，进行训练
3. 将network.py运行成功后会生成一个ss_label.dat和ss_model.h5文件，在运行ges_test.py进行测试，看看测试结果怎么样
4. 测试如果精度高的话，则可以用来玩游戏了，具体游戏的操作可以根据你的手势来决定

需要库有opencv，keras。。。打不开就装相应的库吧
