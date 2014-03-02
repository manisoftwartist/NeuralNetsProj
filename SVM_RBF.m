clear all;
use_spider;                    % Required to initialise toolbox
clc;

% Load the training vector set
load  trainvectors1.mat;
U = U';
% Load the testing vector set
load  testvectors1.mat;
X = X';

% Network Outputs pre-allocated
Y   =   zeros( [1 size(U,1)]);
Ylabel_class1   =   ones( [1 size(U,1)/2]);
Ylabel_class2  =   -1*( ones( [1 size(U,1)/2]) );
Ylabel = [ Ylabel_class1 Ylabel_class2];
Ylabel = Ylabel';
save Ylabel.mat Ylabel;
load Ylabel.mat;

traindata = data(U,Ylabel);
testdata = data( X,Ylabel);

% Effect of sigma on the classification quality of the radial machine
% Use sigma = 
% Given in question are 0.3,0.9,2.7

a = svm(kernel('gaussian',0.9));
[traindata,a] = train(a,traindata);
r = test(a,testdata);
loss(traindata);
loss(r);
plot(a);
title('RBF Kernel with Sigma = 2.7; Classification of Testing dataset-3');