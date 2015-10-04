%% Use Least Squares to Classify Handwritten Digits:

images = loadMNISTImages('train-images.idx3-ubyte');
images = images';
labels = loadMNISTLabels('train-labels.idx1-ubyte');
N = numel(labels); % get number of training data points
% --------------------------------------------------------------------------
%% Organize Training Data:

[classes,classIdx] = sort(labels); % sort by class
ONES_idx = find(classes==1,1);
TWOS_idx = find(classes==2,1); 
THREES_idx = find(classes==3,1);
FOURS_idx = find(classes==4,1); 
FIVES_idx = find(classes==5,1); 
SIXES_idx = find(classes==6,1); 
SEVENS_idx = find(classes==7,1);
EIGHTS_idx = find(classes==8,1);
NINES_idx = find(classes==9,1);


A = images(classIdx,:); % sort the data in numerical order of classes

A = [A,ones(N,1)]; % add in a column of ones for use with the constant term 
% in our linear model

% -------------------------------------------------------------------------
%% Using least squares to classify whether or not sample is a given digit:
% rather than trying to classify what digit it is right away, let's use
% least squares to decide whether or not it is any given digit 

% zeros: ------------------------------------------------------------------
b = -1*ones(N,1);
b(1:ONES_idx) = 1; 

coeff = A\b; % least squares
ZEROS_c1 = coeff(1:(end-1)); ZEROS_c2 = coeff(end); % yHat = c1*x + c2

% ones: -------------------------------------------------------------------
b = -1*ones(N,1);
b(ONES_idx:(TWOS_idx-1)) = 1; % make b an array which is -1 for all indexes 
% corresponding to non ones and 1 for all indexes corresponding to ones

coeff = A\b; % least squares
ONES_c1 = coeff(1:(end-1)); ONES_c2 = coeff(end); % yHat = c1*x + c2

% twos: -------------------------------------------------------------------
b = -1*ones(N,1);
b(TWOS_idx:(THREES_idx-1)) = 1; 

coeff = A\b; % least squares
TWOS_c1 = coeff(1:(end-1)); TWOS_c2 = coeff(end); % yHat = c1*x + c2

% threes: -----------------------------------------------------------------
b = -1*ones(N,1);
b(THREES_idx:(FOURS_idx-1)) = 1; 

coeff = A\b; % least squares
THREES_c1 = coeff(1:(end-1)); THREES_c2 = coeff(end); % yHat = c1*x + c2

% fours: ------------------------------------------------------------------
b = -1*ones(N,1);
b(FOURS_idx:(FIVES_idx-1)) = 1; 

coeff = A\b; % least squares
FOURS_c1 = coeff(1:(end-1)); FOURS_c2 = coeff(end); % yHat = c1*x + c2

% fives: ------------------------------------------------------------------
b = -1*ones(N,1);
b(FIVES_idx:(SIXES_idx-1)) = 1; 

coeff = A\b; % least squares
FIVES_c1 = coeff(1:(end-1)); FIVES_c2 = coeff(end); % yHat = c1*x + c2

% sixes: ------------------------------------------------------------------
b = -1*ones(N,1);
b(SIXES_idx:(SEVENS_idx-1)) = 1; 

coeff = A\b; % least squares
SIXES_c1 = coeff(1:(end-1)); SIXES_c2 = coeff(end); % yHat = c1*x + c2

% sevens: -----------------------------------------------------------------
b = -1*ones(N,1);
b(SEVENS_idx:(EIGHTS_idx-1)) = 1; 

coeff = A\b; % least squares
SEVENS_c1 = coeff(1:(end-1)); SEVENS_c2 = coeff(end); % yHat = c1*x + c2

% eights: -----------------------------------------------------------------
b = -1*ones(N,1);
b(EIGHTS_idx:(NINES_idx-1)) = 1; 

coeff = A\b; % least squares
EIGHTS_c1 = coeff(1:(end-1)); EIGHTS_c2 = coeff(end); % yHat = c1*x + c2

% nines: ------------------------------------------------------------------
b = -1*ones(N,1);
b(NINES_idx:end) = 1; 

coeff = A\b; % least squares
NINES_c1 = coeff(1:(end-1)); NINES_c2 = coeff(end); % yHat = c1*x + c2

% ------------------------------------------------------------------------
%% Check accuracy of each Binary Classifier:

% test data:
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_images = test_images';
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
n = numel(test_labels);

%%
err = zeros(10,1);
C1 = [ZEROS_c1, ONES_c1, TWOS_c1, THREES_c1, FOURS_c1, FIVES_c1, SIXES_c1,...
        SEVENS_c1, EIGHTS_c1, NINES_c1];
C2 = [ZEROS_c2; ONES_c2; TWOS_c2; THREES_c2; FOURS_c2; FIVES_c2; SIXES_c2;...
        SEVENS_c2; EIGHTS_c2; NINES_c2];

for k = 0:9
    test = -1*ones(n,1);
    test(test_labels==k) = 1;

    calc = test_images*C1(:,k+1) + C2(k+1)*ones(n,1);
    classifyBin = ones(size(calc)); classifyBin(calc<0) = -1;
    % choose to classify the as specified digit for value >0 and as not
    % that digit for value <0

    err(k+1) = sum(test - classifyBin)/n * 100; 
end

disp(err)

% note that the largest error is 8.48 percent :)

% ------------------------------------------------------------------------

%%
% okay, so can classify if a test sample is a specified digit or not,
% next can we identify what digit a test sample is?

% .... to be continued
%%