import torch
import matplotlib.pyplot as plt 
torch.manual_seed(99)

# -----------计算网络输出:前馈式计算---------------
def forward(w1,b1,w2,b2,x):                                   
    return w2@torch.tanh(w1@x+b1)+b2
# w1 @ x + b1：输入层 → 隐藏层（矩阵乘法 + 偏置）
# torch.tanh(...)：激活函数（引入非线性！否则 MLP 只能拟合线性函数）
# w2 @ ... + b2：隐藏层 → 输出层

# -----------计算损失函数: 使用均方差--------------
def loss(y,py):
    return ((y-py)**2).mean()
# 均方误差，适合回归任务
# 用来衡量：模型的预测结果和真实答案之间的“差距”有多大。
# 告诉模型“你错得有多离谱”，从而指导模型如何改进自己。


# ------训练数据----------------
x = torch.linspace(-5,5,20).reshape(1,20)                      # 在[-5,5]之间生成20个数作为x
y = torch.sin(x)                                               # 模型的输出值y

#-----------训练模型------------------------
in_num  = x.shape[0]                                            # 输入个数
out_num = y.shape[0]                                            # 输出个数
hn  = 4                                                         # 隐节点个数
w1  = torch.randn([hn,in_num],requires_grad=True)    # 输入 → 隐藏层的映射           # 初始化输入层到隐层的权重w1
b1  = torch.randn([hn,1],requires_grad=True) #                    # 初始化隐层的阈值b1
# 没有偏置，网络无法灵活调整每个神经元的“激活中心”。
# 隐藏神经元输出为：h=tanh(w⋅x)
# 当 x=0 时，无论 w 是多少，都有 h=tanh(0)=0
w2  = torch.randn([out_num,hn],requires_grad=True) #   隐藏层 → 输出的映射           # 初始化隐层到输出层的权重w2
b2  = torch.randn([out_num,1],requires_grad=True)  #              # 初始化输出层的阈值b2
# 输入 →（线性映射 + 非线性）→ 隐藏表示 →（线性映射）→ 输出

lr = 0.01                                                       # 学习率
for i in range(5000):                                           # 训练5000步
    py = forward(w1,b1,w2,b2,x)                                 # 计算网络的输出
    L = loss(y,py)                                              # 计算损失函数
    print('第',str(i),'轮：',L)                                 # 打印当前损失函数值
    L.backward()                                                # 用损失函数更新模型参数的梯度
    w1.data=w1.data-w1.grad*lr                                  # 更新模型系数w1
    b1.data=b1.data-b1.grad*lr                                  # 更新模型系数b1
    w2.data=w2.data-w2.grad*lr                                  # 更新模型系数w2
    b2.data=b2.data-b2.grad*lr                                  # 更新模型系数b2
    w1.grad.zero_()                                             # 清空w1梯度,以便下次backward
    b1.grad.zero_()                                             # 清空b1梯度,以便下次backward
    w2.grad.zero_()                                             # 清空w2梯度,以便下次backward
    b2.grad.zero_()                                             # 清空b2梯度,以便下次backward
px = torch.linspace(-5,5,100).reshape(1,100)                    # 测试数据,用于绘制网络的拟合曲线    
py = forward(w1,b1,w2,b2,px).detach().numpy()                   # 网络的预测值
plt.scatter(x, y)                                               # 绘制样本
plt.plot(px[0,:],py[0,:])                                       # 绘制拟合曲线  
print('w1:',w1)
print('b1:',b1)
print('w2:',w2)
print('b2:',b2)