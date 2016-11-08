function [] = make_tsv_file(xcel_file, range)

% Purpose: Function to convert the response data into a useful tsv file
%          This is needed because excel and google sheets can't export a
%          tsv file in a useful manner
% Authors: Thomas Jeffries
% Created: 2161104
% Modified: 20161104
%
% Inputs:   xcel_file:  .xls file that has the data, downloaded from the 
%                       google forms questionnaire responses.
%           range:      Which portion of the xcel file to read. 
%                       e.g. 'A1:BB22'
%           
% Outputs:  a tsv file to be used by the champ.py script

% The num and text data aren't used, but I put in them anyways just in case
% we want to change it later.
[num, txt, data] = xlsread(xcel_file, 1, range);

% Remove any newline or return characters from the strings
for i = 1:size(data,1)
    for j = 1:size(data,2)
        % Check if the data is a string
        if ischar(data{i,j})
            % If it is, remove any return characters
            data{i,j} = regexprep(data{i,j},'\r\n|\r|\n|','. ');
        end
    end
end

% Convert and write to a .tsv file
T = cell2table(data);
writetable(T,'HF_data_fall2016.txt','Delimiter','\t')
