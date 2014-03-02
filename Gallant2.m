%  Rosenblatt perceptron for classification
% Last Updated:  31/12/2008
% Update : Modified Classification to include plotting of weights and bias at regular intervals

%% TRAINING PART
%  Check if the proper mat files have been loaded

 clear all;
 max_iterations = 10000;
 
% Load the training vector set
load  trainvectors2.mat; 
U3 = ones(1,200);               %  In extended notation,we have a row of all inputs as 1.
U = [U;U3] ;                        % Append this to U so that U becomes a (3X200)

% Class labels for classes 1 and 2
Ydes1 =  ones(1,100);
Ydes2 =  -1 * ones(1,100);
Ydes = [ Ydes1 Ydes2];
Y   =   zeros(1,size(U,2));

W = rand(size(U,1),1);      % Assume random weights
Ws = W;                           % Stored weights for Gallant s algorithm
eta = 0.1;                          % Learning Rate
misclass_flg = 1;               %  Initialize  a  misclassification  indicator
pass = 0;                          %  1 pass is one iteration through all the input images contained in U
i =1;   imax = 100;            % To log weights
W_chkpts = zeros(size(U,1),imax);
 
while (misclass_flg ==1)
    misclass_flg = 0;
    h = 0;   hs =0;                      % No of correct classifications in an iteration
    for j = 1 : size(U,2)
        Y(j) = sign( W' * U(:,j) ); % Perceptron Output for each input data point
        if ( Y(j) * Ydes(j) < 0)     % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
             misclass_flg = 1;
             W_update = U(:,j) * sign(Ydes(j) - Y(j));
             W_old = W;
             W = W + W_update;
        else
             h = h+1;
        end
    end         % for lopp ends
    if (j==1)
        hs = h;
    end
    if (h > hs)         % If @ any moment, h>hs,substitute h with hs and W with Ws
        hs = h;
        Ws = W;
   end    
    pass=pass+1;    
    if (mod(pass,100) == 0)     % Log the weights every 1000th pass
        W_chkpts(1,i) = W_old(1,:);    
        W_chkpts(2,i) = W_old(2,:);
        W_chkpts(3,i) = W_old(3,:);
        i = i +1;
    end
%     if  (pass == 100*100) 
     if  (pass == max_iterations);
            misclass_flg = 0;
            disp(' The data points are NOT linearly separable...Program exited after 10,000 iterations');
            pass = 0;
    end
end


figure(1);
plot( U(1,1:100), U(2,1:100) , 'b+' );
hold on;
plot( U(1,101:200), U(2,101:end) , 'g.' );
legend('Class1 data','Class2 data');

figure(2);
t = 1 : 100;         % No of times weight is logged
plot( t , W_chkpts(1,:) ,'r');
hold on;
plot( t , W_chkpts(2,:),'b');
hold on;
plot( t , W_chkpts(3,:),'g');
title('bias');
xlabel('No. of iterations X 100');
legend('Weight w1','Weight w2');


%% TESTING PART

% Load the testing vector set
load testvectors2.mat; %X is the matrix of test data
X3 = ones(1,200);
X = [X;X3];

% Optimal W calculated from training samples
% Desired Outputs
Ydes1 =  ones(1,100);
Ydes2 =  -1 * ones(1,100);
Ydes = [ Ydes1 Ydes2];
Y   =   zeros(1,size(X,2));

mistakes = 200;          % Assume all are wrong
p=1;
 for j = 1 : size(X,2)
        Y(j) = sign( Ws' * X(:,j) ); % Perceptron Output for each test data point
        if ( Y(j)*Ydes(j) > 0)       % i.e., if Y = -1, but actual class Ydesired = +1,this function would be negative
            mistakes = mistakes-1;
        else
            X_misclass(1,p) = X(1,j);
            X_misclass(2,p) = X(2,j);
            p = p+1;
        end
end
disp('No of mistakes');
disp(mistakes);

figure(3);
plot( X(1,1:100), X(2,1:100) , 'b+' );
hold on;
plot( X(1,101:200), X(2,101:end) , 'g.' );
hold on;
plot( X_misclass(1,1:end),X_misclass(2,1:end), 'r*' );
legend('Class 1 data','Class 2 data',' Unclassified Data');


     
