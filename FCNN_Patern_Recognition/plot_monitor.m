function [num_figure] = plot_monitor(num_figure, monitor_cost, monitor_accuracy,...
    testing_cost, testing_accuracy, training_cost,...
    training_accuracy, validation_cost, validation_accuracy)
    % Monitor the number of figure.
    
    epochs = max([length(testing_cost) length(testing_accuracy)...
        length(training_cost) length(training_accuracy)...
        length(validation_cost) length(validation_accuracy)]);
    line_width = 1;
    
    % Show the changes of the cost during training process.
    if monitor_cost
        num_figure = num_figure + 1;
        figure(num_figure)
        hold on
        h1 = plot(1:epochs, training_cost, '-r','LineWidth',line_width);
        h2 = plot(1:epochs, testing_cost,'-b','LineWidth',line_width);
        h3 = plot(1:epochs, validation_cost,'-g','LineWidth',line_width);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend([h1 h2 h3], 'Cost on training data','Cost on test data','Cost on validation data','FontSize',10)
        legend('boxoff')

        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = semilogy(1:epochs, training_cost, '-r','LineWidth',line_width);
        hold on
        h2 = semilogy(1:epochs, testing_cost,'-b','LineWidth',line_width);
        h3 = semilogy(1:epochs, validation_cost,'-g','LineWidth',line_width);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Cost', 'FontSize', 14)
        legend([h1 h2 h3], 'Cost on training data','Cost on test data','Cost on validation data','FontSize',10)
        legend('boxoff')
    end

    % Show the changes of the accuracy during training process.
    if monitor_accuracy
        num_figure = num_figure + 1;
        figure(num_figure)
        h1 = plot(1:epochs, training_accuracy, '-r','LineWidth',line_width);
        hold on
        h2 = plot(1:epochs, testing_accuracy,'-b','LineWidth',line_width);
        h3 = plot(1:epochs, validation_accuracy,'-g','LineWidth',line_width);
        xlabel('Epoch', 'FontSize', 14)
        ylabel('Accuracy', 'FontSize', 14)
        legend([h1 h2 h3], 'Accuracy on training data','Accuracy on test data','Accuracy on validation data','FontSize',10,'Location','SouthEast')
        legend('boxoff')
    end
end