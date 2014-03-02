clear all;
clc;
use_spider;                    % Required to initialise toolbox

% Load the training vector set
load  trainvectors3.mat;
U = U';
% Load the testing vector set
load  testvectors3.mat;
X = X';

% Network Outputs pre-allocated
Y   =   zeros( [1 size(U,1)]);
Ylabel_class1   =   ones( [1 size(U,1)/2]);
Ylabel_class2   =   -1*( ones( [1 size(U,1)/2]) );
Ylabel = [ Ylabel_class1 Ylabel_class2];
Ylabel = Ylabel';
save Ylabel.mat Ylabel;
load Ylabel.mat;

traindata = data(U,Ylabel);
testdata = data( X,Ylabel);  

% Effect of degree of polynomial on the classification quality of the polynomial kernel
%  Use degree 2,3 and 1 - linear
a = svm(kernel('poly',3)); 
[traindata,a] = train(a,traindata);
r = test(a,testdata);
loss(r)
plot(a);
title('Polynomial kernel of degree=3 classification of Testing Dataset-3');


