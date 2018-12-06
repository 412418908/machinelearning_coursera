
data = [ 1 2; 2 4; 3 6;4 8; 5 10]
m = size(data)(1)
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
y = data(:, 2)
theta = zeros(2, 1); % initialize fitting parameters
iterations = 10;
alpha = 0.03;


----------------
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

m = length(y); % number of training examples
theta = zeros(2, 1);
%iterations = 10; alpha = 0.03;
iterations = 1500;
alpha = 0.01;
[a,b]=gradientDescent(X, y, theta, alpha, iterations);


data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X mu sigma] = featureNormalize(X);
X
mu
sigma
