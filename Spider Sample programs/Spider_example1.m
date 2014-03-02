X = randn(5);
Y =[1 1 1 -1 -1]';
d = data(X,Y);
a  = svm;
[tr,a]=train(a,d);