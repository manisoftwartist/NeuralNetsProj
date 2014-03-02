clear all;

load trainvectors1.mat;
load testvectors1.mat;

p = [U X];

Tdes1 = 0.8 * ones(1,100);
Tdes2 =  -0.8 * ones(1,100);
t = [ Tdes1 Tdes2 Tdes1 Tdes2];


 [p,ps] = mapminmax(p);
 [target,ts] = mapminmax(t);

[trainV,val,test] = dividevec(p,t,0.25,0.25);

% Feedforward network with Levenberg-Marquardt algorithm
net = newff(minmax(p),[10,1],{'logsig','logsig'},'traingd');
net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

a1 = sim(net,p);
net.trainParam.show = 50;
net.trainParam.epochs = 4000;
net.trainParam.goal = 0.05;

% Actual Training
% [net,tr]=train(net,trainV.P,trainV.T,[],[],val,test);
[net,tr]=train(net,p,t);
a2 = sim(net,p);
%[m b r] = postreg(a2,t);

