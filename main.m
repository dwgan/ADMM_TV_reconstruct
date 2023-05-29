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