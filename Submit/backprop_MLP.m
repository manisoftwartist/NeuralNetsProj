% function  [W1,B1,W2,B2,estored,miscl] = backprop(Z,labZ,M,T,epsilon,eta)
%  Backpropagation  training  of  an  MLP with  a  single  hidden  layer  with  M  nodes  in  it
%  W1  and  W2  are  arrays  with  weights:  input-hidden and  hidden-output,  respectively


%% INITIALIZATION 
clear all;

load trainvectors1.mat;

Z = U';
% Class labels for classes 1 and 2
Ydes1 =  1 * ones(1,100);
Ydes2 =  2 * ones(1,100);
Ydes = [ Ydes1 Ydes2];
Ydes = Ydes';

M = 10; % No of hidden layer nodes
epsilon = 0;
T = 1000;
labZ = Ydes;
eta = 0.1;

[n,m] = size(Z); 
c = 2;  % Number of classes
bin_labZ = repmat(labZ,1,c) == repmat(1:c,n,1);  %  use  labels to  form  c-component  binary  target  vectors

W1 = rand(M,m);     %weights  input - hidden
B1 =  rand(M,1);       %biases  input - hidden
W2 = rand(c,M);      %weights  hidden - output
B2 = rand(c,1);         %biases  hidden - output
E = inf;                      %criterion  value
t = 1;                         %iteration  counter

%%  CALCULATION

estored = []; miscl = [];
while  (E>epsilon)  &&  (t<=T)
    
    temp1 = [W1  B1] * [Z  ones(n,1)]';
    oh = 1./(1+exp(-temp1));                     %  outputs  of  the  hidden  layer
    temp2 = [W2  B2] * [oh;  ones(1,n)];
    o = 1./(1+exp(-temp2));
        
    %  outputs  of  the  output  layer
    E = sum(sum((o'-bin_labZ).^2));
    delta_o = (o-bin_labZ').*o.*(1-o);
    delta_h = ( (delta_o'*W2).*oh'.*(1-oh') )';

    for  i = 1:c,  %  update  W2  and  B2
        for  j = 1:M,
            W2(i,j) = W2(i,j)-eta*delta_o(i,:)*(oh(j,:))';
       end;
       B2(i) = B2(i)-eta*delta_o(i,:)*ones(n,1);
    end;

for  i=1:M,  %  update  W1  and  B1
    for  j=1:m,
        W1(i,j) = W1(i,j)-eta*delta_h(i,:)*Z(:,j);
    end;
    B1(i) = B1(i)-eta*delta_h(i,:)*ones(n,1);
end;

t=t+1;
estored = [estored;E];  %  store  the  MLP  squared  error
[dummy,guessed_labels] = max(o);    
miscl = [miscl;1-mean(guessed_labels' == labZ)];
%  store  the  classification  error
end;


