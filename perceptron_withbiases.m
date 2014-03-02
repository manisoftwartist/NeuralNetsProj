%  Rosenblatt perceptron for classification
% Last Updated:  08/01/2009
% Update : Included the biases as an extra layer of weights and plot the separating line
%               Modified Regression to Classification

 clear all;
 max_iterations = 10000;
 
% Load the training vector set
load  trainvectors1.mat;

U3 = ones(1,200);               %  In extended notation,we have a row of all inputs as 1.
U = [U;U3] ;                        % Append this to U so that U becomes a (3X200)
W = rand(size(U,1),1);          % Assume random weights

% Class labels for classes 1 and 2
Ydes1 =  ones(1,100);
Ydes2 =  -1 * ones(1,100);
Ydes = [ Ydes1 Ydes2];

Y   =   zeros(1,size(U,2));

misclass_flg = 1;                %  Initialize  a  misclassification  indicator
pass = 0;                           %  1 pass is one iteration through all the input images contained in U

while (misclass_flg ==1)
    misclass_flg = 0;
    
    for j = 1 : size(U,2)
        Y(j) = sign( W' * U(:,j) ); % Perceptron Output for each input data point
        if ( Y(j) * Ydes(j) < 0)     % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
            misclass_flg = 1;
            W_update = U(:,j) * sign(Ydes(j) - Y(j));
            W = W + W_update;
        end
    end
    pass=pass+1;
    if  (pass==max_iterations) 
           misclass_flg = 0;
           disp(' The data points are NOT linearly separable; Training exited after 10,000 iterations');
            pass = 0;
    end
end

figure(1);
plot( U(1,1:100), U(2,1:100) , 'b*' );
hold on;
plot( U(1,101:200), U(2,101:end) , 'm+' );
hold on;
% To plot the separating line for the training data
% Like Y = a.X + b;
% c2 = -2.5 : 0.25 : 2.5;
% c1 = - ( ( (W(1,:)*c2) + W(3,:)) / W(2,:) );
% plot(c2,c1,'r');
hold off;
title('Perceptron Classification on Training Dataset');

% % Load the testing vector set
load testvectors3.mat; %X is the matrix of test data

X3 = ones(1,200);               %  In extended notation,we have a row of all inputs as 1.
X = [X;X3] ;                        % Append this to U so that U becomes a (3X200)

mistakes = 0;
for j = 1 : size(X,2)
        Y(j) = sign( W' * X(:,j) );     % Perceptron Output for each test data point
        if ( Y(j)*Ydes(j) < 0)           % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
             mistakes = mistakes+1;
        end
end
disp('No of mistakes');
disp(mistakes);
 
figure(2);
plot( X(1,1:100), X(2,1:100) , 'b*' );
hold on;
plot( X(1,101:200), X(2,101:end) , 'm+' );
hold on;
legend('Class1 data','Class2 data');
% To plot the separating line for the training data
% Like Y = a.X + b;
% c2t = -2.5 : 0.25 :2.5;
% c1t = - ( ( (W(1,:)*c2t) + W(3,:)) / W(2,:) );
% plot(c2t,c1t,'r');
title('Perceptron Classification on Testing Dataset');
hold off;
