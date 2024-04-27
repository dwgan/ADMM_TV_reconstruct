# ADMM_TV_reconstruct

# 1. Problem
---
信号压缩是是目前信息处理领域非常成熟的技术，其主要原理是利用信号的稀疏性。一个稀疏信号的特征是，信号中有且仅有少量的位置是有值的，其它位置都是零。对于一个稀疏的信号，在存储时只需要记录有值的位置，从而实现对原始信号的压缩。

对于原本不稀疏的信号，可以利用一种字典（正交变换基，例如傅里叶、小波）对信号进行线性表示，得到原始信号在这个稀疏域上的稀疏表示系数，只需要记录这些少量的系数就能够实现对信号的压缩存储。

在信号重建时，首先对信号补零得到原始维度的信号，由于采用的变换字典是正交的，可以通过正交的变换得到原始信号。

压缩感知与传统的信号压缩有着异曲同工之妙，而不同之处在于，压缩感知的信号压缩过程是将原始的模拟信号直接进行压缩（采样即压缩）。而传统信号压缩通常是先将信号采样后再进行压缩（采样后压缩），这种压缩方式的主要问题在于采用较高的采样资源将信号采集后，又在信号压缩过程将费尽心思采集到的信号丢弃了，从而造成资源的浪费[^1]。

以经典的图像采集为例，对于一副$m\times n$的图像信号$I$进行采集，根据正交变换的思想，至少要对其进行$N=m\times n$次采样才能够得到这副图像。

若该图像是稀疏的，根据压缩感知理论，可以至少进行$M$次采样就能够采集到该信号，其中$M<<N$，$M$的值有信号的稀疏性决定。

# 2. Formulation
---
一个压缩感知采样过程可以表示为

$Ax=b+e\tag{1}$

其中， $A\in \mathbb R^{M\times N}$ 表示观测矩阵， $x\in \mathbb R^{N\times 1}$ 表示 $I\in \mathbb R^{m\times n}$ 的向量形式， $b\in \mathbb R^{M\times 1}$ 表示测量向量， $e\in \mathbb R^{M\times 1}$ 表示测量噪声。可以看到，一个 $N$ 维的信号在采样过程中被压缩成 $M$ 维的信号，这里的M<N。

信号重构过程可以表示为一个优化问题，利用信号的梯度稀疏性质，可以构建目标函数：

$\min_x\ ||Dx||_1\ s.t. \ Ax=b+e\tag{2}$

其中， $x$ 为待求解信号， $D$ 为全变分算子[^2]。去除约束有


$\min_x \ \lambda||Dx||_1+\frac12 ||Ax-b||^2_2\tag{3}$


该问题是一个非凸、不光滑问题，无法直接采用梯度下降法求解。引入变量 $d$ ，将问题(3)转化为ADMM的一般形式[^3]

$\min_x \frac12 ||Ax-b||^2_2+\lambda||d||_1 \ \ s.t.\ \ Dx-d=0\tag{4}$

利用增广拉格朗日法引入凸松弛，同时去除约束条件，有

$L(x,d,\mu)=\frac12 ||Ax-b||^2_2+\lambda||d||_1+\mu^T(Dx-d)+\frac \delta 2||Dx-d||^2_2\tag{5}$

其中 $\mu$ 为拉格朗日乘子， $\delta>0$ 为拉格朗日惩罚项。为了使表达更简洁，可做如下替换：

$L(x,d,\mu)=\frac12 ||Ax-b||^2_2+\lambda||d||_1+\frac \delta 2||Dx-d+p||^2_2-\frac \delta 2||p||_2^2\tag{6}$

其中 $p=\mu / \delta$ 。利用ADMM，问题(6)的求解可通过交替求解以下三个问题进行实现：

$x_{n+1}=arg\,\min_x\ \frac12 ||Ax-b||^2_2+\frac \delta 2||Dx-d_n+p_n||^2_2\tag{7}$

$d_{n+1}=arg\,\min_u\ \lambda||d||_1+\frac \delta 2||Dx_n-d+p_n||^2_2\tag{8}$

$p_{n+1}=p_n+(Dx_{n+1}-d_{n+1})\tag{9}$

# 3. Simulation
---
测试图像采用的是house，分别测试在20%、40%、60%、80%和100%采样率时，压缩感知重构算法的图像恢复结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dc09aa43e7434ea488b3df0c69bad46f.jpeg#pic_center)


从仿真结果可以看到，在20%采样率时，信号的基本轮廓信息就被成功采集了，当采样率达到60%以上时，继续增加采样率并没有使得图像更加的清晰，也就是说，针对这副图像，若采用传统的正交采集的方式，有将近一半的采样资源是被浪费的。

# 4. Algorithm
---

```matlab
ratio=0.5;%采样率
x=double(imread('house.bmp'));
[m,n]=size(x);
N=m*n;
M=floor(ratio*N);
A=rand(M,N);%观测矩阵
e=50*randn(M,1);%测量噪声
% 信号采集过程，利用线性投影对信号x进行采集，同时考虑了测量噪声
b=A*reshape(x,N,1)+e;
% 信号重构过程，利用仅有的M个测量值恢复维度为N的信号
x_r=ADMM_TV_reconstruct(A,b,300,500,100);
figure;
subplot(121);
imshow(uint8(x));
title('原始图像');
subplot(122);
imshow(uint8(reshape(x_r,m,n)));
title(sprintf('重构图像（%d%%采样率）',ratio*100));
```

```matlab
function xp=ADMM_TV_reconstruct(A,b,delta,lambda,iteratMax)
    [~,N]=size(A);
    [Dh,Dv]=TVOperatorGen(sqrt(N));
    D=sparse([(Dh)',(Dv)']');
    d=D*ones(N,1);
    p=ones(2*N,1)/delta;
    invDD=inv(A'*A+delta*(D'*D));
    for ii=1:iteratMax
        x=invDD*(A'*b+delta*D'*(d-p));
        d=wthresh(D*x+p,'s',lambda/delta);
        p=p+D*x-d;
    end
    xp=x;
end
```

```matlab
function [Dh,Dv]=TVOperatorGen(n)
    Dh=-eye(n^2)+diag(ones(1,n^2-1),1);
    Dh(n:n:n^2,:)=0;
    Dv=-eye(n^2)+diag(ones(1,n^2-n),n);
    Dv(n*(n-1)+1:n^2,:)=0;
end
```
```matlab
Github link: https://github.com/dwgan/ADMM_TV_reconstruct
```

# 参考文献
[^1]:Baraniuk, Richard G. "Compressive sensing [lecture notes]." IEEE signal processing magazine 24.4 (2007): 118-121.
[^2]:Rudin, Leonid I., Stanley Osher, and Emad Fatemi. "Nonlinear total variation based noise removal algorithms." Physica D: nonlinear phenomena 60.1-4 (1992): 259-268.
[^3]:Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.
