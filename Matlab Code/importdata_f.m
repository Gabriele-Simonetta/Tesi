function [hidroPV,hidroC,hidroCas,date,nameC,namePV,nameCas] = importdata(Dmax,anno,filename)
 opts = spreadsheetImportOptions("NumVariables", 2);

        % Specify sheet and range
        opts.Sheet = anno;
        opts.DataRange = strcat('A2:B',Dmax); 

% Specify column names and types
opts.VariableNames = ["Orario", "ParmaPonteVerdiLivelloIdrometrico23050m"];
opts.VariableTypes = ["datetime", "double"];

% Specify variable properties
opts = setvaropts(opts, "Orario", "InputFormat", "");

% Import the data
pos = strcat(cd,'\',filename);
tbl = readtable(pos, opts, "UseExcel", false);

%% Convert to output type
date = tbl.Orario;
hidroPV = tbl.ParmaPonteVerdiLivelloIdrometrico23050m;
anno=opts.Sheet;
stazione='Ponte_Verdi';
namePV=strcat(stazione,'_',anno);

opts = spreadsheetImportOptions("NumVariables", 2);
    opts.Sheet = anno;
    opts.DataRange = strcat('D2:E',Dmax); 

 % Specify column names and types
opts.VariableNames = ["Orario1", "ColornoAipoIdrometroTorrenteParma44852m"];
opts.VariableTypes = ["datetime", "double"];

% Specify variable properties
opts = setvaropts(opts, "Orario1", "InputFormat", "");

% Import the data
tbl = readtable(pos, opts, "UseExcel", false);

%% Convert to output type
date = tbl.Orario1;
hidroC= tbl.ColornoAipoIdrometroTorrenteParma44852m;
stazione='Colorno';
nameC=strcat(stazione,'_',anno)

opts = spreadsheetImportOptions("NumVariables", 2);
    opts.Sheet = anno;
    opts.DataRange = strcat('G2:H',Dmax); 

 % Specify column names and types
opts.VariableNames = ["Orario1", "CasalmaggioreLivelloIdrometrico14019m"];
opts.VariableTypes = ["datetime", "double"];

% Specify variable properties
opts = setvaropts(opts, "Orario1", "InputFormat", "");

% Import the data
tbl = readtable(pos, opts, "UseExcel", false);

%% Convert to output type
date = tbl.Orario1;
hidroCas = tbl.CasalmaggioreLivelloIdrometrico14019m;
stazione='Casal_Maggiore';
nameCas=strcat(stazione,'_',anno)
end

