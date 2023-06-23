%% Specify some related value.
% Specify the datafile and the related parameters.
[filename, pathname] = uigetfile('*.xlsx', 'Open the parameters setting file');
[Num, Txt] = xlsread(fullfile(pathname, filename),'Sheet1');
datafile = char(strcat(pathname, Txt(1,2)));
result_file = char(strcat(pathname, 'result.xlsx'));

% Specify the data's location in the excel file
sheet_name = char(Txt(2, 2:end));
sheet_range = char(Txt(3, 2));

% Specify some parameters for the data.
num_classification = Num(1);
spectrums_per_sample = Num(8);
train_num = Num(2, :) .* spectrums_per_sample;
test_num = Num(3, :) .* spectrums_per_sample;

% Shuffle the raw data?
shuffle_flag = Num(9);

% Read the data.
[tr_input, tr_output, te_input, te_output, va_input, va_output] ...
    = read_data(datafile, sheet_name, sheet_range, num_classification, ...
    train_num, test_num, shuffle_flag);

% Test or valid?    
test_or_valid = char(Txt(4, 2));

% The number of the waves.
num_input = size(tr_input, 1);

% The inner size of the network.
net_inner_sizes = Num(10, :);
net_inner_sizes(isnan(net_inner_sizes)) = []; 
if isnan(net_inner_sizes)
    net_inner_sizes = [];
end

% The optimization of SGD.
optimization = char(Txt(6, 2));

% The hyperparameters of the network.
epochs = Num(4);
mini_batch_size  = Num(5);
eta = Num(6);
lmbda = Num(7);
reg_fun = char(Txt(5, 2));
keep_prop = Num(14);
momentum = Num(15);
max_norm = Num(16);
beta_momentum = Num(17);
beta_rmsprop = Num(18);
epsilon = 1e-8;

% Early stopping.
early_stopping_n = Num(11);
if early_stopping_n
    epochs = 10000;
end

% Monitor the training process.
monitor_cost = Num(12);
monitor_accuracy = Num(13);

%% Create and train the network.
% Preprocess the data.
[pro_tr_input,PS] = mapminmax(tr_input);
[pro_te_input] = mapminmax('apply', te_input, PS);
[pro_va_input] = mapminmax('apply', va_input, PS);

% Create the network.
net_sizes = [num_input net_inner_sizes num_classification];
net = Network(net_sizes, CrossEntropyCost, keep_prop, momentum, max_norm, ...
    beta_momentum, beta_rmsprop, epsilon, optimization);

% Train the net work.
[testing_cost, testing_accuracy, training_cost, training_accuracy, validation_cost, validation_accuracy] = ...
    net.SGD(pro_tr_input, tr_output, epochs, mini_batch_size, eta, ...
    pro_te_input, te_output, lmbda, reg_fun, ...
    monitor_cost, ...
    monitor_accuracy, ...
    early_stopping_n, ...
    pro_va_input, va_output);

%% Show the result with figures and write it into the excel.
num_figure = 0;

% Show the detail of the training.
num_figure = plot_monitor(num_figure, monitor_cost, monitor_accuracy,...
    testing_cost, testing_accuracy, training_cost, training_accuracy,...
    validation_cost, validation_accuracy);

% Show the final result

% Without the validation data.
if test_or_valid == 'T'
    num_figure = num_figure + 1;
    figure(num_figure)
    plot_confusion(net, ...
        tr_input, tr_output, ...
        te_input, te_output)
    
    % Print the figure with related hyperparameters.
    inner = [];
    for i = 1 : length(net_inner_sizes)
        inner = strcat(num2str(net_inner_sizes(i)), ' ');
    end
    print(gcf, '-dpng', strcat(optimization, ' ', reg_fun, ' epochs=', num2str(epochs), ' eta=', num2str(eta), ' lambda=', num2str(lmbda), ' hidden=', inner, '.png'))
% With the validation data.
elseif test_or_valid == 'V'
    num_figure = num_figure + 1;
    figure(num_figure)
    plot_confusion(net, ...
        tr_input, tr_output, ...
        te_input, te_output, ...
        va_input, va_output)
    
    % Print the figure with related hyperparameters.
    inner = [];
    for i = 1 : length(net_inner_sizes)
        inner = strcat(num2str(net_inner_sizes(i)), ' ');
    end
    print(gcf, '-dpng', strcat(optimization, ' ', reg_fun, ' epochs=', num2str(epochs), ' eta=', num2str(eta), ' lambda=', num2str(lmbda), ' hidden=', inner, '.png'))
else
    fprintf('Warning! You do not correctly specify the mode of test or validation!');
end

% Write the result into the excel
tr_predict = net_ouput(net, pro_tr_input, tr_output);
te_predict = net_ouput(net, pro_te_input, te_output);

write_result(tr_predict, te_predict,...
 tr_output, te_output, result_file, num_classification, train_num, test_num)