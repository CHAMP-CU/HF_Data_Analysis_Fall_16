% warning('off','MATLAB:strrep:InvalidInputType')

file_name = 'CHAMP Test Questionnaire (Responses)';
range = 'A1:GB7';
make_tsv_file(file_name, range)


% [~,~,data] = xlsread(file_name, 1, range);
% 
% % Remove any newline or return characters from the strings
% for i = 1:size(data,1)
%     for j = 54:55%size(data,2)
%         % Check if the data is a string
%         if ischar(data{i,j})
%             % If it is, remove any return characters
%             data{i,j} = regexprep(data{i,j},'\r\n|\r|\n|','. ');
%         end
%     end
% end
% 
% % Convert and write to a .tsv file
% T = cell2table(data);
% writetable(T,'test_table.txt','Delimiter','\t')
% 
% % Load the new data to test
% data2 = tdfread('test_table.txt','\t');