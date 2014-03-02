use_spider;                    % Required to initialise toolbox

load colon.mat;
d1 = data(  X(1:31,:) , Y(1:31) ) ;
d2 = data(  X(32:end,:) , Y(32:end) ) ;
a = svm(kernel('rbf' , 2));
[tr,a] = train(a,d1);
r = test(a,d2);

loss('confusion_matrix',tr)
loss(r)
