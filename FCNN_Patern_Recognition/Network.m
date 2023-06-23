classdef Network < handle
    
    properties
        % Some property about the neural network.
        sizes;
        num_layers;
        biases;
        weights;
        cost;
        
        % Max-norm regularization.
        max_norm;
        
        % For dropout.
        keep_prop;
        
        % For SGD with momentum or Nesterov Accelerated Gradient Descent.
        momentum;
        weight_velocities;
        bias_velocities;
        
        % For AdaGrad or RMSprop or Adam.
        beta_momentum;
        beta_rmsprop;
        epsilon;
        weight_m;
        biases_m;
        weight_v;
        biases_v;
        t_num;
        
        % Point to the chosen SGD optimization algorithm.
        optimization;
    end
    
    methods
        
        function obj = Network(shapes, cost, keep_prop, momentum, max_norm, beta_momentum, ...
                beta_rmsprop, epsilon, optimization)
            
            % NEURAL_LAYER Construct an instance of this class.
            
            a = cell(1, length(shapes) - 1);
            b = cell(1, length(shapes) - 1);
            
            for j = 1 : length(shapes) - 1
                a{1, j} = randn(shapes(j+1), shapes(j)) ./ sqrt(shapes(j));
                b{1, j} = randn(shapes(j+1), 1);
            end
            
            c = cell(1, length(shapes) - 1);
            d = cell(1, length(shapes) - 1);
            
            for j = 1 : length(shapes) - 1
                c{1, j} = zeros(shapes(j+1), shapes(j));
                d{1, j} = zeros(shapes(j+1), 1);
            end
            
            obj.sizes = shapes;
            obj.num_layers = length(shapes);
            obj.weights = a;
            obj.biases = b;
            
            obj.weight_velocities = c;
            obj.bias_velocities = d;
            obj.weight_m = c;
            obj.biases_m = d;
            obj.weight_v = c;
            obj.biases_v = d;
            obj.t_num = 0;
            
            obj.cost = cost;
            obj.keep_prop = keep_prop;
            obj.momentum = momentum;
            obj.max_norm = max_norm;
            obj.beta_momentum = beta_momentum;
            obj.beta_rmsprop = beta_rmsprop;
            obj.epsilon = epsilon;
            obj.optimization = optimization;
        end
        
        function A = feedforward(obj, A)
            % Return the output of the network if ``a`` is input.
            for i = 1 : length(obj.sizes) - 1
                A = A*obj.keep_prop;
                A = sigmoid(obj.weights{i}*A + obj.biases{i});
            end
        end
        
        function [testing_cost, testing_accuracy, training_cost, training_accuracy, validation_cost, validation_accuracy] = ...
                        SGD(obj, tr_input, tr_output, epochs, ...
                        mini_batch_size, eta, te_input, te_output, lmbda, reg_fun, ...
                        monitor_cost, ...
                        monitor_accuracy, ...
                        early_stopping_n, ...
                        va_input, va_output)
            
            % Train the neural network using mini-batch stochastic
            % gradient descent.  The ``training_data`` is a list of tuples
            % ``(x, y)`` representing the training inputs and the desired
            % outputs.  The other non-optional parameters are
            % self-explanatory.  If ``test_data`` is provided then the
            % network will be evaluated against the test data after each
            % epoch, and partial progress printed out.  This is useful for
            % tracking progress, but slows things down substantially.
           
            n_test = size(te_input, 2);
            n = size(tr_input, 2);
            t1 = clock;     % begin timer
            
            % early stopping functionality:
            best_cost = 1000;
            no_cost_change = 0;

            testing_cost = zeros(epochs, 1);
            testing_accuracy = zeros(epochs, 1);
            training_cost = zeros(epochs, 1);
            training_accuracy = zeros(epochs, 1);
            validation_cost = zeros(epochs, 1);
            validation_accuracy = zeros(epochs, 1);
            
            for i = 1 : epochs
                index = randperm(n);
                tr_input = tr_input(:, index);
                tr_output = tr_output(:, index);

                for k = 1 : n / mini_batch_size
                    X = tr_input(:,mini_batch_size*(k-1)+1:mini_batch_size*k);
                    Y = tr_output(:,mini_batch_size*(k-1)+1:mini_batch_size*k);
                    obj.update_parameters(X, Y, eta, lmbda, n, reg_fun);
                end
         
                t2 = clock;
                ela = etime(t2, t1);
                te_cor_num = obj.evaluate(te_input, te_output);
                fprintf('Epoch %d: %d / %d, elapsed time: %5.2fs\n', i, te_cor_num, n_test, ela);
                
                if monitor_cost
                    train_cost = obj.total_cost(tr_input, tr_output);
                    test_cost = obj.total_cost(te_input, te_output);
                    valid_cost = obj.total_cost(va_input, va_output);
                    
                    training_cost(i) = train_cost;
                    testing_cost(i) = test_cost;
                    validation_cost(i) = valid_cost;
                end
                if monitor_accuracy
                    train_accuracy = obj.accuracy(tr_input, tr_output);
                    test_accuracy = obj.accuracy(te_input, te_output);
                    valid_accuracy = obj.accuracy(va_input, va_output);
                    
                    training_accuracy(i) = train_accuracy;
                    testing_accuracy(i) = test_accuracy;
                    validation_accuracy(i) = valid_accuracy;
                end
                
                % Early stopping:
                if early_stopping_n > 0
                    eva_cost = obj.total_cost(te_input, te_output);
                    testing_cost(i) = eva_cost;
                    if eva_cost < best_cost
                        best_cost = eva_cost;
                        no_cost_change = 0;
                    else
                        no_cost_change = no_cost_change + 1;
                    end
                    if (no_cost_change == early_stopping_n)
                        testing_cost = testing_cost(1:i);
                        testing_accuracy = testing_accuracy(1:i);
                        training_cost = training_cost(1:i);
                        training_accuracy = training_accuracy(1:i);
                        validation_cost = validation_cost(1:i);
                        validation_accuracy = validation_accuracy(1:i);
                        break
                    end
                end
            end
        end
        
        function [] = update_parameters(obj, X, Y, eta, lmbda, n, reg_fun)
            
            % Specify the SGD optimization.
            
            if obj.optimization == "SGD"
                obj.update_mini_batch_SGD(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "Momentum"
                obj.update_mini_batch_Momentum(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "NGD"
                obj.update_mini_batch_Nesterov(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "AdaGrad"
                obj.update_mini_batch_AdaGrad(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "RMSprop"
                obj.update_mini_batch_RMSprop(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "Adam"
                obj.update_mini_batch_Adam(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "AdaMax"
                obj.update_mini_batch_AdaMax(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "Nadam"
                obj.update_mini_batch_Nadam(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "AdamW"
                obj.update_mini_batch_AdamW(X, Y, eta, lmbda, n, reg_fun)
            elseif obj.optimization == "NadamW"
                obj.update_mini_batch_NadamW(X, Y, eta, lmbda, n, reg_fun)
            else
                msg = "Hey! You do not specify the optimization function correctly!";
                error(msg);
            end
        end
        
        function [] = update_mini_batch_SGD(obj, X, Y, eta, lmbda, n, reg_fun)
            
            % Updata the weights and the biases by SGD. 
            
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2);
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} - (eta/m)*nabla_w{1, j};
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j};
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) - (eta/m)*nabla_w{1, j};
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j};
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - (eta/m) * nabla_w{1, j};
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - (eta/m)*nabla_w{1, j};
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j};
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end
        
        function [] = update_mini_batch_Momentum(obj, X, Y, eta, lmbda, n, reg_fun)
            
            % Updata the weights and the biases by SGD with momentum. 

            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2);
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    obj.weight_velocities{1, j} = obj.momentum * obj.weight_velocities{1, j} - (1 - obj.momentum) * (eta/m) * nabla_w{1, j};
                    obj.bias_velocities{1, j} = obj.momentum * obj.bias_velocities{1, j} - (1 - obj.momentum) * (eta/m) * sum_nabla_b{1, j};
                    
                    weight_v_corrected = obj.weight_velocities{1, j} / (1 - power(obj.momentum, t));
                    biases_v_corrected = obj.bias_velocities{1, j} / (1 - power(obj.momentum, t));
                    
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} + weight_v_corrected;
                    obj.biases{1, j} = obj.biases{1, j} + biases_v_corrected;
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    obj.weight_velocities{1, j} = obj.momentum * obj.weight_velocities{1, j} - (1 - obj.momentum) * (eta/m) * nabla_w{1, j};
                    obj.bias_velocities{1, j} = obj.momentum * obj.bias_velocities{1, j} - (1 - obj.momentum) * (eta/m) * sum_nabla_b{1, j};
                    
                    weight_v_corrected = obj.weight_velocities{1, j} / (1 - power(obj.momentum, t));
                    biases_v_corrected = obj.bias_velocities{1, j} / (1 - power(obj.momentum, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) + weight_v_corrected;
                    obj.biases{1, j} = obj.biases{1, j} + biases_v_corrected;
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    obj.weight_velocities{1, j} = obj.momentum * obj.weight_velocities{1, j} - (1 - obj.momentum) * (eta/m) * nabla_w{1, j};
                    obj.bias_velocities{1, j} = obj.momentum * obj.bias_velocities{1, j} - (1 - obj.momentum) * (eta/m) * sum_nabla_b{1, j};
                    
                    weight_v_corrected = obj.weight_velocities{1, j} / (1 - power(obj.momentum, t));
                    biases_v_corrected = obj.bias_velocities{1, j} / (1 - power(obj.momentum, t));

                    obj.weights{1, j} = obj.weights{1, j} + weight_v_corrected;
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                    obj.biases{1, j} = obj.biases{1, j} + biases_v_corrected;
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    obj.weight_velocities{1, j} = obj.momentum * obj.weight_velocities{1, j} - (1 - obj.momentum) * (eta/m) * nabla_w{1, j};
                    obj.bias_velocities{1, j} = obj.momentum * obj.bias_velocities{1, j} - (1 - obj.momentum) * (eta/m) * sum_nabla_b{1, j};
                    
                    weight_v_corrected = obj.weight_velocities{1, j} / (1 - power(obj.momentum, t));
                    biases_v_corrected = obj.bias_velocities{1, j} / (1 - power(obj.momentum, t));

                    obj.weights{1, j} = obj.weights{1, j} + weight_v_corrected;
                    obj.biases{1, j} = obj.biases{1, j} + biases_v_corrected;
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end
        
        function [] = update_mini_batch_Nesterov(obj, X, Y, eta, lmbda, n, reg_fun)
            
            
            % Updata the weights and the biases by NAG(Nesterov accelerated gradient descent). 
            
            m = size(X, 2);
            
            % For the NAG Update Rule
            
            shapes = obj.sizes;
            % Initialize the weight_ahead and bia_ahead.
            weight_ahead = cell(1, length(shapes) - 1);
            bia_ahead = cell(1, length(shapes) - 1);
            
            for j = 1 : length(shapes) - 1
                weight_ahead{1, j} = zeros(shapes(j+1), shapes(j));
                bia_ahead{1, j} = zeros(shapes(j+1), 1);
            end
            
            for j = 1 : length(obj.sizes) - 1
                weight_ahead{1, j} = obj.weights{1, j} + obj.momentum * obj.weight_velocities{1, j};
                bia_ahead{1, j} = obj.biases{1, j} + obj.momentum * obj.bias_velocities{1, j};
            end
            
            % Calculate the updated velocities.
            
            sum_nabla_b_ahead = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b_ahead{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            [nabla_b_ahead, nabla_w_ahead] = obj.grad_cal(X, Y, weight_ahead, bia_ahead);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b_ahead{1, j} = sum(nabla_b_ahead{1, j}, 2);
            end
            
            for j = 1 : length(obj.sizes) - 1
                obj.weight_velocities{1, j} = obj.momentum * obj.weight_velocities{1, j} - (eta/m) * nabla_w_ahead{1, j};
                obj.bias_velocities{1, j} = obj.momentum * obj.bias_velocities{1, j} - (eta/m) * sum_nabla_b_ahead{1, j};
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} + obj.weight_velocities{1, j};
                    obj.biases{1, j} = obj.biases{1, j} + obj.bias_velocities{1, j};
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) + obj.weight_velocities{1, j};
                    obj.biases{1, j} = obj.biases{1, j} + obj.bias_velocities{1, j};
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} + obj.weight_velocities{1, j};
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                    obj.biases{1, j} = obj.biases{1, j} + obj.bias_velocities{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} + obj.weight_velocities{1, j};
                    obj.biases{1, j} = obj.biases{1, j} + obj.bias_velocities{1, j};
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
            
        end
        
        function [] = update_mini_batch_AdaGrad(obj, X, Y, eta, lmbda, n, reg_fun)
                        
            % Updata the weights and the biases by AdaGrad. 
            
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2);
            end
            
            for j = 1 : length(obj.sizes) - 1
                obj.weight_m{1, j} = obj.weight_m{1, j} + nabla_w{1, j}.^2;
                obj.biases_m{1, j} = obj.biases_m{1, j} + sum_nabla_b{1, j}.^2;
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} - (eta/m)*nabla_w{1, j}./sqrt(obj.weight_m{1, j} + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j}./sqrt(obj.biases_m{1, j} + obj.epsilon);
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) - (eta/m)*nabla_w{1, j}./sqrt(obj.weight_m{1, j} + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j}./sqrt(obj.biases_m{1, j} + obj.epsilon);
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - (eta/m) * nabla_w{1, j}./sqrt(obj.weight_m{1, j} + obj.epsilon);
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j}./sqrt(obj.biases_m{1, j} + obj.epsilon);
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    obj.weights{1, j} = obj.weights{1, j} - (eta/m)*nabla_w{1, j}./sqrt(obj.weight_m{1, j} + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - (eta/m)*sum_nabla_b{1, j}./sqrt(obj.biases_m{1, j} + obj.epsilon);
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end
            
        function [] = update_mini_batch_RMSprop(obj, X, Y, eta, lmbda, n, reg_fun)
            
            % Updata the weights and the biases by RMSprop. 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} - eta * nabla_w{1, j} ./ sqrt(weight_v_corrected + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * sum_nabla_b{1, j} ./ sqrt(biases_v_corrected + obj.epsilon);
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) - eta * nabla_w{1, j}./ sqrt(weight_v_corrected + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * sum_nabla_b{1, j} ./ sqrt(biases_v_corrected + obj.epsilon);
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * nabla_w{1, j} ./ sqrt(weight_v_corrected + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * sum_nabla_b{1, j} ./ sqrt(biases_v_corrected + obj.epsilon);
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * nabla_w{1, j} ./ sqrt(weight_v_corrected + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * sum_nabla_b{1, j} ./ sqrt(biases_v_corrected + obj.epsilon);
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end
        
        function [] = update_mini_batch_Adam(obj, X, Y, eta, ~, ~, reg_fun)
            
            % Updata the weights and the biases by Adam(Adaptive moment estimation). 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            else
                msg1 = "Hey! The ";
                msg = strcat(msg1, reg_fun);
                msg = strcat(msg, " regularization is not included in this algorithm!");
                error(msg);
            end
        end
        
        function [] = update_mini_batch_AdaMax(obj, X, Y, eta, lmbda, n, reg_fun)
            
            % Updata the weights and the biases by AdaMax. 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                                        
                    % AdaMax
                    obj.weight_v{1, j} = max(obj.beta_rmsprop * obj.weight_v{1, j}, abs(nabla_w{1, j}));
                    obj.biases_v{1, j} = max(obj.beta_rmsprop * obj.biases_v{1, j}, abs(sum_nabla_b{1, j}));
                    
                    obj.weights{1, j} = (1-eta*(lmbda/n))*obj.weights{1, j} - eta * weight_m_corrected ./ sqrt(obj.weight_v{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ sqrt(obj.biases_v{1, j});
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % AdaMax
                    obj.weight_v{1, j} = max(obj.beta_rmsprop * obj.weight_v{1, j}, abs(nabla_w{1, j}));
                    obj.biases_v{1, j} = max(obj.beta_rmsprop * obj.biases_v{1, j}, abs(sum_nabla_b{1, j}));
                    
                    obj.weights{1, j} = obj.weights{1, j} - sign(obj.weights{1, j})*eta*(lmbda/n) - eta * weight_m_corrected ./ sqrt(obj.weight_v{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ sqrt(obj.biases_v{1, j});
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % AdaMax
                    obj.weight_v{1, j} = max(obj.beta_rmsprop * obj.weight_v{1, j}, abs(nabla_w{1, j}));
                    obj.biases_v{1, j} = max(obj.beta_rmsprop * obj.biases_v{1, j}, abs(sum_nabla_b{1, j}));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ sqrt(obj.weight_v{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ sqrt(obj.biases_v{1, j});
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % AdaMax
                    obj.weight_v{1, j} = max(obj.beta_rmsprop * obj.weight_v{1, j}, abs(nabla_w{1, j}));
                    obj.biases_v{1, j} = max(obj.beta_rmsprop * obj.biases_v{1, j}, abs(sum_nabla_b{1, j}));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ sqrt(obj.weight_v{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ sqrt(obj.biases_v{1, j});
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end
        
        function [] = update_mini_batch_Nadam(obj, X, Y, eta, ~, ~, reg_fun)
            
            % Updata the weights and the biases by Nadam(Adaptive moment estimation with Nesterov). 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            else
                msg1 = "Hey! The ";
                msg = strcat(msg1, reg_fun);
                msg = strcat(msg, " regularization is not included in this algorithm!");
                error(msg);
            end
        end
        
        function [] = update_mini_batch_AdamW(obj, X, Y, eta, lmbda, ~, reg_fun)
            
            % Updata the weights and the biases by AdamW(Adaptive moment estimaztion with weight decay). 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                                        
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - (eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon) + lmbda*obj.weights{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - (eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon) + lmbda*sign(obj.weights{1, j}));
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end

        function [] = update_mini_batch_NadamW(obj, X, Y, eta, lmbda, ~, reg_fun)
            
            % Updata the weights and the biases by NadamW(Adaptive moment estimation with Nesterov and weight decay). 
            
            obj.t_num = obj.t_num + 1;
            t = obj.t_num;
            sum_nabla_b = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = zeros(obj.sizes(j+1), 1);
            end
            
            m = size(X, 2);
            
            [nabla_b, nabla_w] = obj.backprop(X, Y);
            for j = 1 : length(obj.sizes) - 1
                sum_nabla_b{1, j} = sum(nabla_b{1, j}, 2) / m;
                nabla_w{1, j} = nabla_w{1, j} / m;
            end
            
            if reg_fun == "L2"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - (eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon) + lmbda*obj.weights{1, j});
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            elseif reg_fun == "L1"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - (eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon) + lmbda*sign(obj.weights{1, j}));
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            elseif reg_fun == "Max-norm"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                    
                    norms = vecnorm(obj.weights{1, j}');
                    for i = 1 : length(norms)
                        if norms(i) < obj.max_norm
                            norms(i) = 1;
                        else
                            norms(i) = obj.max_norm / norms(i);
                        end
                    end
                    obj.weights{1, j} = diag(norms) * obj.weights{1, j};
                end
            elseif reg_fun == "None"
                for j = 1 : length(obj.sizes) - 1
                    % MOMENTUM
                    % Does a weighted average for the history ponits.
                    obj.weight_m{1, j} = (obj.beta_momentum * obj.weight_m{1, j}) + ((1 - obj.beta_momentum) * nabla_w{1, j});
                    obj.biases_m{1, j} = (obj.beta_momentum * obj.biases_m{1, j}) + ((1 - obj.beta_momentum) * sum_nabla_b{1, j});
                    % Corrected weighted average, this is done to adress
                    % the inbalance on the first points since V0=0, the
                    % first few points will be off.
                    weight_m_corrected = obj.weight_m{1, j} / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.biases_m{1, j} / (1 - power(obj.beta_momentum, t));
                    
                    % Nadam
                    weight_m_corrected = obj.beta_momentum * weight_m_corrected + ((1 - obj.beta_momentum) * nabla_w{1, j}) / (1 - power(obj.beta_momentum, t));
                    biases_m_corrected = obj.beta_momentum * biases_m_corrected + ((1 - obj.beta_momentum) * sum_nabla_b{1, j}) / (1 - power(obj.beta_momentum, t));
                    
                    % RMS PROP
                    obj.weight_v{1, j} = (obj.beta_rmsprop * obj.weight_v{1, j}) + ((1 - obj.beta_rmsprop) * power(nabla_w{1, j}, 2));
                    obj.biases_v{1, j} = (obj.beta_rmsprop * obj.biases_v{1, j}) + ((1 - obj.beta_rmsprop) * power(sum_nabla_b{1, j}, 2));
                    weight_v_corrected = obj.weight_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    biases_v_corrected = obj.biases_v{1, j} / (1 - power(obj.beta_rmsprop, t));
                    
                    obj.weights{1, j} = obj.weights{1, j} - eta * weight_m_corrected ./ (sqrt(weight_v_corrected) + obj.epsilon);
                    obj.biases{1, j} = obj.biases{1, j} - eta * biases_m_corrected ./ (sqrt(biases_v_corrected) + obj.epsilon);
                end
            else
                msg = "Hey! You do not specify the regularization function correctly!";
                error(msg);
            end
        end

        function [nabla_B, nabla_W] = backprop(obj, X, Y)
            
            % Return a tuple ``(nabla_b, nabla_w)`` representing the
            % gradient for the cost function C_x.  ``nabla_b`` and
            % ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            % to ``self.biases`` and ``self.weights``.
            
            % Dropout the units(hidden and visible).
            flag = cell(1, length(obj.sizes) - 1);
            for i = 1 : length(obj.sizes) - 1
                flag{1, i} = rand(obj.sizes(i), 1) < obj.keep_prop;
            end
            
            mini_batch_size = size(X, 2);
            nabla_W = cell(1, length(obj.sizes) - 1);
            nabla_B = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                nabla_W{1, j} = zeros(obj.sizes(j+1), obj.sizes(j));
                nabla_B{1, j} = zeros(obj.sizes(j+1), mini_batch_size);
            end
            
            % feedforward
            activation = X;
            activation = activation .* repmat(flag{1, 1}, 1, mini_batch_size);        % Dropout the input neurons.
            activations = cell(1, length(obj.sizes));  % list to store all the activations, layer by layer
            activations{1} = activation;
            zs = cell(1, length(obj.sizes) - 1);     % list to store all the z vectors, layer by layer
            for j = 1 : length(obj.sizes) - 2
                B = repmat(obj.biases{1, j}, 1, mini_batch_size);
                Z = obj.weights{1, j} * activation + B;
                zs{j} = Z;
                activation = sigmoid(Z);
                zs{j} = zs{j} .* repmat(flag{1, j+1}, 1, mini_batch_size);
                activation = activation .* repmat(flag{1, j+1}, 1, mini_batch_size);
                activations{j+1} = activation;
            end
            
            % The output layer is unnecessary to be dropouted.
            z = obj.weights{1, end} * activation + repmat(obj.biases{1, end}, 1 , mini_batch_size);
            zs{length(obj.sizes) - 1} = z;
            activation = sigmoid(z);
            activations{length(obj.sizes)} = activation;
            
            % backward pass
            delta = obj.cost.delta(zs{end}, activations{end}, Y);
            nabla_B{end} = delta;
            nabla_W{end} = delta * activations{end-1}';
            % Note that the variable l in the loop below is used a little
            % differently to the notation in Chapter 2 of the book.  Here,
            % l = 1 means the last layer of neurons, l = 2 is the
            % second-last layer, and so on.  It's a renumbering of the
            % scheme in the book, used here to take advantage of the fact
            % that Python can use negative indices in lists.
            for l = 2 : obj.num_layers - 1
                Z = zs{end+1-l};
                sp = sigmoid_prime(Z);
                delta = (obj.weights{end+2-l}' * delta) .* sp .* repmat(flag{1, end+2-l}, 1, mini_batch_size);
                nabla_B{end+1-l} = delta;
                nabla_W{end+1-l} = delta * activations{end-l}';
            end
        end
        
        function [nabla_B, nabla_W] = grad_cal(obj, X, Y, weight_ahead, bia_ahead)
            
            % Calculate the gradient w.r.t weight_ahead and bia_ahead
            
            mini_batch_size = size(X, 2);
            nabla_W = cell(1, length(obj.sizes) - 1);
            nabla_B = cell(1, length(obj.sizes) - 1);
            for j = 1 : length(obj.sizes) - 1
                nabla_W{1, j} = zeros(obj.sizes(j+1), obj.sizes(j));
                nabla_B{1, j} = zeros(obj.sizes(j+1), mini_batch_size);
            end
            
            % feedforward
            activation = X;
            activations = cell(1, length(obj.sizes));  % list to store all the activations, layer by layer
            activations{1} = activation;
            zs = cell(1, length(obj.sizes) - 1);     % list to store all the z vectors, layer by layer
            for j = 1 : length(obj.sizes) - 1
                B = repmat(bia_ahead{1, j}, 1, mini_batch_size);
                Z = weight_ahead{1, j} * activation + B;
                zs{j} = Z;
                activation = sigmoid(Z);
                activations{j+1} = activation;
            end
            
            % backward pass
            delta = obj.cost.delta(zs{end}, activations{end}, Y);
            nabla_B{end} = delta;
            nabla_W{end} = delta * activations{end-1}';
            % Note that the variable l in the loop below is used a little
            % differently to the notation in Chapter 2 of the book.  Here,
            % l = 1 means the last layer of neurons, l = 2 is the
            % second-last layer, and so on.  It's a renumbering of the
            % scheme in the book, used here to take advantage of the fact
            % that Python can use negative indices in lists.
            for l = 2 : obj.num_layers - 1
                Z = zs{end+1-l};
                sp = sigmoid_prime(Z);
                delta = (weight_ahead{end+2-l}' * delta) .* sp;
                nabla_B{end+1-l} = delta;
                nabla_W{end+1-l} = delta * activations{end-l}';
            end
        end
        
        function cor_num = evaluate(obj, X, Y)
            
            % Count the number of the correctely classified one.

            [~, test_results] = max(obj.feedforward(X));
            [~, test_expects] = max(Y);
            cor_num = sum(test_expects == test_results, 2);
        end
        
        function result_accuracy = accuracy(obj, X, Y)
            
            % Evaluate the accuracy of the output and the target.
            
            cor_num = evaluate(obj, X, Y);
            result_accuracy = cor_num ./ size(X, 2);
        end
        
        function cost = total_cost(obj, X, Y)
            n = size(X, 2);
            A = obj.feedforward(X);
            cost = sum(obj.cost.fn(A, Y)) / n;
        end
        
        function predict = net_ouput(obj, X, Y)
            
            % Outout the network's output.
            
            n = size(Y, 2);
            m = size(Y, 1);

            [~, test_results] = max(obj.feedforward(X));
            
            predict = zeros(m, n);
            for i = 1 : n
                predict(test_results(i), i) = 1;
            end
        end
            
    end
end