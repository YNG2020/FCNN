function [] = write_result(tr_predict,te_predict,tr_output,...
            te_output,result_file,num_classification,train_num,test_num)
    if num_classification == 2
        cor_num_tr_A = length(find(tr_output(1, 1:train_num(1)) == tr_predict(1, 1:train_num(1))));
        cor_num_tr_B = length(find(tr_output(1, train_num(1)+1:sum(train_num)) == tr_predict(1, train_num(1)+1:sum(train_num))));
        cor_num_tr = cor_num_tr_A + cor_num_tr_B;
        precision_tr_A = cor_num_tr_A / train_num(1);
        precision_tr_B = cor_num_tr_B / train_num(2);
        precision_tr = cor_num_tr / sum(train_num);

        cor_num_te_A = length(find(te_output(1, 1:test_num(1)) == te_predict(1, 1:test_num(1))));
        cor_num_te_B = length(find(te_output(1, test_num(1)+1:sum(test_num)) == te_predict(1, test_num(1)+1:sum(test_num))));
        cor_num_te = length(find(te_output(1, 1:sum(test_num)) == te_predict(1, 1:sum(test_num))));
        precision_te_A = cor_num_te_A / test_num(1);
        precision_te_B = cor_num_te_B / test_num(2);
        precision_te = cor_num_te / sum(test_num);

        cor_num_A = cor_num_tr_A + cor_num_te_A;
        cor_num_B = cor_num_tr_B + cor_num_te_B;

        precision_A = cor_num_A / (train_num(1) + test_num(1));
        precision_B = cor_num_B / (train_num(2) + test_num(2));

        precision_All = (cor_num_A + cor_num_B) / (train_num(1) + test_num(1) + train_num(2) + test_num(2));

        result = [precision_tr_A precision_tr_B precision_te_A precision_te_B precision_tr precision_te precision_A precision_B precision_All];
        STD_All = std(result);
        result = [result STD_All];

        xlswrite(result_file,{'Tr-A-Acc','Tr-B-Acc','Te-A-Acc','Te-B-Acc','Tr-Acc','Te-Acc','A-Acc','B-Acc','All-Acc','STD'},'Sheet1','C3');
        xlswrite(result_file,result,'Sheet1','C4');
    elseif num_classification == 3
        cor_num_tr_A = length(find(tr_output(1, 1:train_num(1)) == tr_predict(1, 1:train_num(1))));
        cor_num_tr_B = length(find(tr_output(2, train_num(1)+1:train_num(1)+train_num(2)) == tr_predict(2, train_num(1)+1:train_num(1)+train_num(2))));
        cor_num_tr_C = length(find(tr_output(3, train_num(1)+train_num(2)+1:sum(train_num)) == tr_predict(3, train_num(1)+train_num(2)+1:sum(train_num))));
        cor_num_tr = cor_num_tr_A + cor_num_tr_B + cor_num_tr_C;
        precision_tr_A = cor_num_tr_A / train_num(1);
        precision_tr_B = cor_num_tr_B / train_num(2);
        precision_tr_C = cor_num_tr_C / train_num(3);
        precision_tr = cor_num_tr / sum(train_num);

        cor_num_te_A = length(find(te_output(1, 1:test_num(1)) == te_predict(1, 1:test_num(1))));
        cor_num_te_B = length(find(te_output(2, test_num(1)+1:test_num(1)+test_num(2)) == te_predict(2, test_num(1)+1:test_num(1)+test_num(2))));
        cor_num_te_C = length(find(te_output(3, test_num(1)+test_num(2)+1:sum(test_num)) == te_predict(3, test_num(1)+test_num(2)+1:sum(test_num))));
        cor_num_te = cor_num_te_A + cor_num_te_B + cor_num_te_C;
        precision_te_A = cor_num_te_A / test_num(1);
        precision_te_B = cor_num_te_B / test_num(2);
        precision_te_C = cor_num_te_C / test_num(3);
        precision_te = cor_num_te / sum(test_num);

        cor_num_A = cor_num_tr_A + cor_num_te_A;
        cor_num_B = cor_num_tr_B + cor_num_te_B;
        cor_num_C = cor_num_tr_C + cor_num_te_C;

        precision_A = cor_num_A / (train_num(1) + test_num(1));
        precision_B = cor_num_B / (train_num(2) + test_num(2));
        precision_C = cor_num_C / (train_num(3) + test_num(3));

        precision_All = (cor_num_A + cor_num_B + cor_num_C) / (train_num(1) + test_num(1) + train_num(2) + test_num(2) + train_num(3) + test_num(3));

        result = [precision_tr_A precision_tr_B precision_tr_C precision_te_A precision_te_B precision_te_C precision_tr precision_te precision_A precision_B precision_C precision_All];
        STD_All = std(result);
        result = [result STD_All];

        xlswrite(result_file,{'Tr-A-Acc','Tr-B-Acc','Tr-C-Acc','Te-A-Acc','Te-B-Acc','Te-C-Acc','Tr-Acc','Te-Acc','A-Acc','B-Acc','C-Acc','All-Acc','STD'},'Sheet1','C3');
        xlswrite(result_file,result,'Sheet1','C4');
    end
end