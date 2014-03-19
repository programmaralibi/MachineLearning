function performance = nn_model(inputs, targets)

inputs = inputs';
targets = targets';

% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 40/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 50/100;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
% errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
% view(net);
end