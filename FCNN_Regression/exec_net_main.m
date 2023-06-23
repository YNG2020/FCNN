%% Specify some related value.
% Specify the datafile and the related parameters.
[filename, pathname] = uigetfile('*.xlsx', 'Open the parameters setting file');
[Num, Txt] = xlsread(fullfile(pathname, filename),'Sheet1');
datafile = char(strcat(pathname, Txt(1,2)));
result_file = char(strcat(pathname, 'result.xlsx'));

% Specify the data's location in the excel file
sheet_name = char(Txt(2, 2:end));
sheet_range1 = char(Txt(3, 2));
sheet_range2 = char(Txt(4, 2));

% Specify some parameters for the data.
num_classification = Num(1);
spectrums_per_sample = Num(8);
train_num = Num(2, :) .* spectrums_per_sample;
test_num = Num(3, :) .* spectrums_per_sample;

% Shuffle the raw data?
shuffle_flag = Num(9);

% Read the data.
[tr_input, tr_output, te_input, te_output, va_input, va_output] ...
    = read_data(datafile, sheet_name, sheet_range1, sheet_range2, num_classification, ...
    train_num, test_num, shuffle_flag);

% Test or valid?    
test_or_valid = char(Txt(5, 2));

% The number of the waves.
num_input = size(tr_input, 1);

% The inner size of the network.
net_inner_sizes = Num(10, :);
net_inner_sizes(isnan(net_inner_sizes)) = []; 
if isnan(net_inner_sizes)
    net_inner_sizes = [];
end

% The optimization of SGD.
optimization = char(Txt(7, 2));

% The hyperparameters of the network.
epochs = Num(4);
mini_batch_size  = Num(5);
eta = Num(6);
lmbda = Num(7);
reg_fun = char(Txt(6, 2));
keep_prop = Num(16);
momentum = Num(17);
max_norm = Num(18);
beta_momentum = Num(19);
beta_rmsprop = Num(20);
epsilon_adam = 1e-8;

% Early stopping.
early_stopping_n = Num(11);
if early_stopping_n
    epochs = 10000;
end

% Monitor the training process.
monitor_training_cost = Num(12);
monitor_training_accuracy = Num(13);
monitor_evaluation_cost = Num(14);
monitor_evaluation_accuracy = Num(15);

%% Create and train the network.
% Preprocess the data.
[pro_tr_input,iPS] = mapminmax(tr_input);
[pro_test_input] = mapminmax('apply', te_input, iPS);
[pro_valid_input] = mapminmax('apply', va_input, iPS);

[pro_tr_output,oPS] = mapminmax(tr_output);
[pro_test_output] = mapminmax('apply', te_output, oPS);
[pro_valid_output] = mapminmax('apply', va_output, oPS);

% Create the network.
net_sizes = [num_input net_inner_sizes 1];
net = Network(net_sizes, QuadraticCost, keep_prop, momentum, max_norm, ...
    beta_momentum, beta_rmsprop, epsilon_adam, optimization);

% Train the net work.
[evaluation_cost, training_cost] = net.SGD(pro_tr_input, pro_tr_output, epochs, mini_batch_size, eta, ...
    pro_test_input, pro_test_output, lmbda, reg_fun, ...
    monitor_evaluation_cost, ...
    monitor_training_cost, ...
    early_stopping_n, oPS);

%% Show the result with figures and excel.
num_figure = 0;

% Show the detail of the training.
num_figure = plot_monitor(num_figure, monitor_training_cost,...
    monitor_evaluation_cost,...
    evaluation_cost, training_cost);

% Show the performance of the network.
tr_result = mapminmax('reverse', net.feedforward(pro_tr_input), oPS);
te_result = mapminmax('reverse', net.feedforward(pro_test_input), oPS);
all_result = [tr_result te_result];
all_output = [tr_output te_output];

plotregression(tr_output,tr_result,'Training',te_output,te_result,'Test', all_output,all_result,'All')
print(gcf, '-dpng', strcat(optimization, ' epochs=', num2str(epochs), ' eta=', num2str(eta), ' lambda=', num2str(lmbda), ' hidden=', num2str(net_inner_sizes), '.png'))

% Save the network.
save net
% Save the result.
xlswrite(result_file,"Target",'Sheet1','B3');
xlswrite(result_file,all_output,'Sheet1','C3');
xlswrite(result_file,"Result",'Sheet1','B4');
xlswrite(result_file,all_result,'Sheet1','C4');
tr_mse = sum(net.cost.fn(tr_result, tr_output)) / train_num;
te_mse = sum(net.cost.fn(te_result, te_output)) / test_num;
all_mse = sum(net.cost.fn(all_result, all_output)) / (train_num + test_num);

xlswrite(result_file,{'Train_MSE', 'Test_MSE', 'All_MSE'},'Sheet1','B6');
xlswrite(result_file,[tr_mse te_mse all_mse],'Sheet1','B7');