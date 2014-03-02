
clear all;
 
% Load the training vector set
load  trainvectors1.mat;
data = U';

% Class labels for classes 1 and 2
Ydes1 =  ones(1,100);
Ydes2 =  -1 * ones(1,100);
Ydes = [ Ydes1 Ydes2];
target = Ydes';

nhidden = 10;
nout = 1;
alpha = 0.1; % weight decay

no_inputs = 2;
net = MLP(no_inputs,nhidden,nout,'logistic',alpha);
options = foptions; 
options(1) =1;         % Display the error
options(14) = 500;  % 500 iteratins
[net,options] = netopt(net,options,data,target,'quasinew');

yg = mlpfwd(net,[data(:,1) data(:,2)]);

yg = reshape( yg(:,1),size(data(:,1)) );
[cn hn] = contour(xrange,yrange,yg,[0.5 0.5],'r-');
