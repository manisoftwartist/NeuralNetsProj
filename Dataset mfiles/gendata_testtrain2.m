% Generate training vectors 
clear all;

% For class S-
U1 = normrnd(-1.167,sqrt(0.3),[1 100]);
U2 = normrnd(-1.09,sqrt(0.3),[1 100]);

% For class S+
U3 = normrnd(1.167,sqrt(0.6),[1 100]);
U4 = normrnd(1.09,sqrt(0.6),[1 100]);

U_row1 = [U1 U3];
U_row2 = [U2 U4];
U = [U_row1;U_row2];

figure('Name','TRAINING SET');
xlabel('Input Vector Dimension1 u1');
ylabel('Input vector Dimension2 u2');
scatter(U(1,:), U(2,:),'b');

% For class S-
X1 = normrnd(-1.167,sqrt(0.3),[1 100]);
X2 = normrnd(-1.09,sqrt(0.3),[1 100]);

% For class S+
X3 = normrnd(1.167,sqrt(0.6),[1 100]);
X4 = normrnd(1.09,sqrt(0.6),[1 100]);

X_row1 = [X1 X3];
X_row2 = [X2 X4];
X = [X_row1;X_row2];

figure('Name','TESTING SET');
xlabel('Testing set Input Vector Dimension1 u1');
ylabel('Testing set Input vector Dimension2 u2');
scatter(X(1,:), X(2,:),'g');

save trainvectors2.mat U;
disp('Training Data generated and saved in matrix  trainvectors.mat');
save testvectors2.mat X;
disp('Test Data generated and saved in matrix  testvectors.mat');