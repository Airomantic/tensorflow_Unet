
#对输入的彩色图像需要做归一化
def normal(img,mask):
    img=tf.cast(img,tf.float32)/127.5-1 #转换数据 归一
    mask=tf.cast(mask,tf.int32)
    return img,mask
def load_image_train(img_path,mask_path):
    img=read_png(img_path)
    mask=read_png_label(mask_path)
    img,mask=crop_img(img,mask) #随机采检
    #随机常量
    if tf.random.uniform(())>0.5: #同时翻转
        img=tf.image.flip_left_right(img)
        mask=tf.image.flip_left_right(mask)
        #裁剪
        # img=tf.image.resize(img,(256,256))
        # mask = tf.image.resize(mask, (256, 256))

    img,mask=normal(img,mask) #归一化
    return img,mask

#test无需翻转
def load_image_val(img_path, mask_path):
    img = read_png(img_path)
    mask = read_png_label(mask_path)

    # 裁剪
    img = tf.image.resize(img, (256, 256))
    mask = tf.image.resize(mask, (256, 256))

    img, mask = normal(img, mask)  # 归一化
    return img, mask
BATCH_SIZE=32
BUFFER_SIZE=300
#每一个epoch训练多少步 才能将所有的图片给循环一遍
step_per_epoch=train_count//BATCH_SIZE
val_step=val_count//BATCH_SIZE
#计算机根据CPU选择读取图片的线程数
auto=tf.data.experimental.AUTOTUNE
#trian数据加载 多线程处理的线程数
#输入管道
dataset_train=dataset_train.map(load_image_train,num_parallel_calls=auto)
dataset_val=dataset_val.map(load_image_val, num_parallel_calls=auto)
for i,m in dataset_train.take(1):
    plt.subplot(1,2,1)
    plt.imshow((i.numpy()+1)/2) #归一化0-1之间
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(m.numpy()))
    plt.show()
dataset_train=dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val=dataset_val.batch(BATCH_SIZE)
#至此训练数据全部配置好
#定义Unet模型 自定义层封装
#下采样卷积卷积 每次都同样的卷积核
class Downsample(tf.keras.layers.Layer):
    def __init__(self,units): #一个卷积层有units个单元数
        super(Downsample,self).__init__() #继承父类所有属性
        # 定义卷积核 使用tensorflow内置的卷积层
        self.convolution_kernel_1=tf.keras.layers.Conv2D(units,kernel_size=3,
                                                         padding='same')
        self.convolution_kernel_2 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                                           padding='same')
        self.pool=tf.keras.layers.MaxPooling2D()
    #前向传播
    def call(self, x, is_pool=True): #设置is_pool=false时就是两个卷积层
        if is_pool:
            x=self.pool(x) #下采样
        #否则其它情况下直接调用卷积
        x=self.convolution_kernel_1(x)
        x=tf.nn.relu(x)#激活x
        x = self.convolution_kernel_2(x)
        x = tf.nn.relu(x)  # 激活x
        return x
#上采样
class Upsample(tf.keras.layers.Layer):
    def __init__(self,units): #一个卷积层有units个单元数
        super(Upsample,self).__init__() #继承父类所有属性
        #两个卷积层+一个反卷积层 输入的是单元数 反卷积为上一层单元数除以2
        self.convolution_kernel_1 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                                           padding='same')
        self.convolution_kernel_2 = tf.keras.layers.Conv2D(units, kernel_size=3,
                                                           padding='same')
        #反卷积
        self.de_convolution_kernel= tf.keras.layers.Conv2DTranspose(units//2,
                                                                    kernel_size=2,
                                                                    strides=2,#放大图像 跨度为2
                                                                    padding='same'
                                                                    )
    #前向传播运算
    def call(self, x, is_pool=True):
        x = self.convolution_kernel_1(x)
        x = tf.nn.relu(x)
        x = self.convolution_kernel_2(x)
        x = tf.nn.relu(x)
        x = self.de_convolution_kernel(x)
        x = tf.nn.relu(x)
        return x
#is_pool这个参数是对实例调用的时候，运行call()时调用的参数，而不是初始化时传入的参数
#定义模型的子类
class Unet_model(tf.keras.Model):
    def __init__(self):
        super(Unet_model, self).__init__()
        #定义Unet所需要的所有的层
        self.down1=Downsample(64) #unit=64
        self.down2 = Downsample(64*2) #128
        self.down3 = Downsample(64*4) #256
        self.down4 = Downsample(64*8) #512
        self.down5 = Downsample(64*16) #1024

        self.up=tf.keras.layers.Conv2DTranspose(512,
                                                kernel_size=2,
                                                strides=2, #放大，如果没有设置为2就没有上采样
                                                padding='same' #合并 填充
                                                )
        #初始化三个上采样层 卷积层的Unit
        self.up1=Upsample(512)
        self.up2=Upsample(512//2)
        self.up3=Upsample(512//4)
        #初始化
        self.convolution_kernel_last=Downsample(64)
        #张量长度为34 最后一层-输出层，综合利用前面得到的结果，将每一个像素分类
        self.last=tf.keras.layers.Conv2D(34,
                                         kernel_size=1,
                                         padding='same')
    #全向传播 下采样封装 调用前面定义好的逻辑方法
    def call(self, x):
        #这里只有两层卷积 没有pool
        x1 = self.down1(x, is_pool=False)  # 256*256*64
        x2 = self.down2(x1)  # 128*128*128
        x3 = self.down3(x2)  # 64*64*256
        x4 = self.down4(x3)  # 32*32*512
        x5 = self.down5(x4)  # 16*16*1024

        x5 = self.up(x5)  # 32*32*512
        #concat()保留了它的空间结构 用于合并
        x5 = tf.concat([x4, x5], axis=-1)  # 32*32*1024
        x5 = self.up1(x5)  # 64*64*256
        x5 = tf.concat([x3, x5], axis=-1)  # 64*64*512
        x5 = self.up2(x5)  # 128*128*128
        x5 = tf.concat([x2, x5], axis=-1)  # 128*128*256
        x5 = self.up3(x5)  # 256*256*64
        x5 = tf.concat([x1, x5], axis=-1)  # 256*256*128

        x5 = self.convolution_kernel_last(x5, is_pool=False)  # 256*256*64

        x5=self.last(x5) #输出 # 256*256*3
        return x5

model=Unet_model()
print(model)
opt=tf.keras.optimizers.Adam(0.0001)
#如果没有激活，需要添加参数from_logits: bool = True
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #labels 0.1.2.3
#交并集
#输出是长度为34的张量 而实际是0，1，2，3的位置/分类
class MeanIOU(tf.keras.metrics.MeanIoU):
    #y_ture 对应每一个像素是长度为34的张量 y_pred类别 3还是4还是33 所以要把y_ture变成y_pred实际的预测结果
    def __call__(self, y_ture,y_pred):
        y_pred=tf.argmax(y_pred) #这里要求y_ture，y_pred是一样的形状
        return super().__call__(y_ture,y_pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#计算指标
train_loss=tf.keras.metrics.Mean(name='train_loss') #解析成实际运行的结果
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_iou=MeanIOU(34,name='train_iou')

test_loss=tf.keras.metrics.Mean(name='test_loss') #解析成实际运行的结果
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_iou=MeanIOU(34,name='test_iou')


@tf.function #自动图（同）运算，因为这个函数流程是一样的（关键点），设置成静态图来运行
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions) #损失值
        #计算梯度 可训练参数
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #使用优化函数对函数内的参数变量进行优化

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_iou(labels, predictions)
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions) #test的损失

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_iou(labels, predictions)
###训练函数
"""
在这里可以把打印的epoch指标append到一个列表里面，有利于绘图看变化
"""
EPOCHS = 60

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标 #将所有的指标清零
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_iou.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_iou.reset_states()

    for images, labels in dataset_train:
        train_step(images, labels)

    for test_images, test_labels in dataset_val:
        test_step(test_images, test_labels)

    template = 'Epoch {:.3f}, Loss: {:.3f}, Accuracy: {:.3f}, \
                IOU: {:.3f}, Test Loss: {:.3f}, \
                Test Accuracy: {:.3f}, Test IOU: {:.3f}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           train_iou.result(),
                           test_loss.result(),
                           test_accuracy.result()*100,
                           test_iou.result()
                           ))