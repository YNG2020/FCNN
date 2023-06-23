function [tr_input, tr_output, te_input, te_output, va_input, va_output] ...
    = read_data(datafile, sheet_name, sheet_range1, sheet_range2, num_classification, ...
    train_num, test_num, shuffle_flag)

    % Read the data.
    input = cell(1, num_classification);
    for i = 1 : num_classification
        input{1, i} =  xlsread(datafile, sheet_name, sheet_range1);
        if shuffle_flag
            index = randperm(size(input{1, i}, 2));
            input{1, i} = input{1, i}(:, index);
        end
    end

    % Count the number of samples and waves.
    count_num = zeros(1, num_classification);
    labels = cell(1, num_classification);
    for i = 1 : num_classification
        count_num(i) = size(input{1, i}, 2);
        labels{1, i} = zeros(num_classification, count_num(i));
    end
    
    % Create the labels.
    for i = 1 : num_classification
        labels{1, i} = xlsread(datafile, sheet_name, sheet_range2);
    end

    tr_input = [];
    tr_output = [];
    te_input = [];
    te_output = [];
    va_input = [];
    va_output = [];
    for i = 1 : num_classification
        tr_input = [tr_input input{1, i}(:, 1:train_num(i))];
        tr_output = [tr_output labels{1, i}(:, 1:train_num(i))];
        te_input = [te_input input{1, i}(:, 1+train_num(i):train_num(i)+test_num(i))];
        te_output = [te_output labels{1, i}(:, 1+train_num(i):train_num(i)+test_num(i))];
        va_input = [va_input input{1, i}(:, 1+train_num(i)+test_num(i):end)];
        va_output = [va_output labels{1, i}(:, 1+train_num(i)+test_num(i):end)];
    end
end