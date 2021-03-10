clear all
clc
close all
f=1;
%% Importdata
for w=2020 %Impostare l'anno da elaborare o la serie consecutiva di anni 2013:2019
    anni1=[2017,2015,2014,2013,2019,2018];
    anni2=[2016,2012,2020];
    if find(anni1==w)~= ' '
       Dmax ='52705'; % Dimensione del vettore dati
    else
        Dmax = '52849';
    end
 anno=num2str(w);
 filename = "Colorno Aipo - Idrometro Torrente Parma.xlsx"; %nome del file
 hmin=0.10;
 %slope=0.02;
 DT=24*7;
 [hidroPV,hidroC,hidroCas,date,nameC,namePV,nameCas]=importdata_f(Dmax,anno,filename); %funzione che recupera i dati di Ponte verdi (PV) e Colorno (C) e Casalmaggiore (Cas)
 date=datetime(date, 'InputFormat', 'dd/MM/uuuu HH:mm', 'Format', 'dd-MM-uuuu HH:mm');
%% Interpolazione valori nulli
[hidroPV] = interpolate_f(hidroPV);
[hidroC] = interpolate_f(hidroC);
[hidroCas] = interpolate_f(hidroCas);
%% Discretizzazione ogni 30 minuti (mn==30) NB! dipende dai valori presenti inizialmente!
j=1;
for i=1:length(date)
    [yyyy, mm, dd, h, mn,ss] = datevec(date(i));
    if mn==0 || mn==30 %|| mn==40
        dat_(j)= date(i);
        hidroC_(j)=hidroC(i); 
        hidroPV_(j)=hidroPV(i);
        hidroCas_(j)=hidroCas(i);
        j=j+1;
    end
end
hidroPV_=hidroPV_';
hidroC_=hidroC_';
hidroCas_=hidroCas_';
dat_=dat_';

inizio=["21-06-2012 00:00","18-06-2013 00:00","12-08-2014 00:00","15-05-2015 00:00","08-07-2016 00:00", "08-07-2017 00:00","22-06-2018 00:00","30-06-2019 00:00","30-06-2020 00:00"];
fine=["25-09-2012 00:00","20-10-2013 00:00","08-10-2014 00:00","03-10-2015 00:00","03-11-2016 00:00", "03-11-2017 00:00","24-11-2018 00:00","14-10-2019 00:00","30-09-2020 00:00"];
posin = find(dat_==(inizio(w-2011))); %taglio magre
posfin = find(dat_==(fine(w-2011)));
pi=0;
j=1;
i=1;
for i=posin:posfin
    pi(j)=i;
    j=j+1;
end
hidroPVa_=hidroPV_;
hidroCa_=hidroC_;
hidroCasa_=hidroCas_;
data_=dat_;
data_(pi,:)=[];
hidroCasa_(pi,:)=[];
hidroCa_(pi,:)=[];
hidroPVa_(pi,:)=[];

onde=table(data_,hidroPVa_,hidroCasa_, hidroCa_); % [da1,eval(['ondaPV' num2str(k)]),eval(['ondaC' num2str(k)])]
filename=strcat('30min_onde_corr_',num2str(w),'.txt');
writetable(onde,filename,'Delimiter',',','WriteVariableNames',0)          
% %% Figure
    f=f+1;
    figure(f); %plot 
            plot(dat_, hidroPV_,'Color', [1.0 0.667 0.0]);
            hold on
            plot(data_, hidroPVa_,'r');
            grid on 
            grid minor 
            title('Ponte Verdi')
            xlabel('Mesi/gg');
            ylabel('Quota idrica [m]');
            saveas(gcf,strcat('Ponte Verdi diff','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
            saveas(gcf,strcat('Ponte Verdi diff','_',num2str(w),'.png')); %salvataggio grafico formato .png
    f=f+1;
    figure(f); %plot 
            plot(dat_, hidroC_,'c');
            hold on
            plot(data_, hidroCa_,'b');
            grid on 
            grid minor 
            title('Colorno')
            xlabel('Mesi/gg');
            ylabel('Quota idrica [m]');
            saveas(gcf,strcat('Colorno diff','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
            saveas(gcf,strcat('Colorno diff','_',num2str(w),'.png')); %salvataggio grafico formato .png

    f=f+1;
    figure(f); %plot 
            plot(dat_, hidroCas_,'g');
            hold on
            plot(data_, hidroCasa_,'Color', [ 0.0275    0.6392    0.1882]);
            grid on 
            grid minor 
            title('Casalmaggiore')
            xlabel('Mesi/gg');
            ylabel('Quota idrica [m]');
            saveas(gcf,strcat('Casalmaggiore diff','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
            saveas(gcf,strcat('Casalmaggiore diff','_',num2str(w),'.png')); %salvataggio grafico formato .png

    f=f+1;
    figure(f); %plot 
            plot(dat_, hidroPV_, 'Color', [1.0 0.667 0.0],'DisplayName','Ponte Verdi');
            hold on
            plot(data_, hidroPVa_, 'r');
            plot(dat_, hidroC_,'c','DisplayName','Colorno');
            plot(data_, hidroCa_,'b');
            plot(dat_, hidroCas_,'g','DisplayName',' Casalmaggiore');
            plot(data_, hidroCasa_, 'Color', [ 0.0275    0.6392    0.1882]);
            grid off 
            grid minor 
            legend('Ponte Verdi','','Colorno','','Casalmaggiore','')
            title('Altezza idrometrica nelle 3 stazioni')
            xlabel('Mesi');
            ylabel('Quota idrica [m]');
            saveas(gcf,strcat('Tre stazioni a confronto diff','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
            saveas(gcf,strcat('Tre stazioni a confronto diff','_',num2str(w),'.png')); %salvataggio grafico formato .png
      
% 
% %% Eliminazione dati sotto ad hmin e un range DT
% range=DT*2;
% j=1;
% k=1;
% i=1;
% e=1;
% hidroC=hidroC_';
% hidroCas=hidroCas_';
% hidroPV=hidroPV_';
% data=dat_';
% dati=0;
% clear date hidroPV_ hidroC_ dat_ hidroCas_
% while i<length(hidroPV)
%     if i<j
%         i=j;
%     end    
%     if hidroPV(i)<hmin
%        j=i;
%        r=j+range; % shift finestra temporale 
%        while   hidroPV(j)<hmin && j<=r
%             j=j+1;
%             datae(e)= data(j);
%             dati (e,1)= hidroPV(j);
%             dati (e,2)= hidroC(j);
%             e=e+1;
%        end
%        hidroPV_(k)=hidroPV(j);
%         data_(k,:)=data(j,:);
%         hidroC_(k)=hidroC(j);
%         j=j+1;
%       else
%         hidroPV_(k)=hidroPV(i);
%         data_(k,:)=data(i,:);
%         hidroC_(k)=hidroC(i);
%         j=j+1;
%     end
%     i=i+1;
%     k=k+1;
% end
% hidroPV_=hidroPV_';
% hidroC_=hidroC_';
%        onde=[hidroPV_,hidroC_]; % [da1,eval(['ondaPV' num2str(k)]),eval(['ondaC' num2str(k)])]
%        filename=strcat('onde',num2str(w),'.txt');
%        fileID = fopen(filename,'w');
%        fprintf(fileID,'%4.2f,%4.2f \n',onde');
% 
% 
% f=f+1;
% figure(f); %plot 
%         plot(data_, hidroPV_);
%         grid on 
%         grid minor 
%         title('Stazione idrometrica Parma Ponte Verdi')
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('Parma Ponte Verdi anno','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('Parma Ponte Verdi anno','_',num2str(w),'.png')); %salvataggio grafico formato .png
% f=f+1;
% figure(f); %plot 
%         plot(data_, hidroC_);
%         grid on 
%         grid minor 
%         title('Stazione idrometrica Colorno')
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('Colorno anno','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('Colorno anno','_',num2str(w),'.png')); %salvataggio grafico formato .png
% if dati==0 
%     f=f+1;
%     figure(f); %plot 
%         plot(data,hidroPV,'r')
%         hold on
%         grid on
%         grid minor 
%         title('Stazione idrometrica Ponte verdi')
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('Parma Ponte Verdi anno completo','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('Parma Ponte Verdi anno completo','_',num2str(w),'.png')); %salvataggio grafico formato .png
%     f=f+1;
%     figure(f); %plot 
%         plot(data,hidroC,'r')
%         hold on
%         grid on
%         grid minor 
%         title('Stazione idrometrica Colorno')
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('Colorno anno completo','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('Colorno anno completo','_',num2str(w),'.png')); %salvataggio grafico formato .png
% 
%     else
%         f=f+1;
%         figure(f); %plot 
%             plot(data,hidroPV,'r')
%             hold on
%             plot(datae,dati(:,1),'k.')
%             grid on
%             grid minor 
%             title('Stazione idrometrica Ponte verdi')
%             xlabel('Mesi/gg');
%             ylabel('Quota idrica [m]');
%             saveas(gcf,strcat('Parma Ponte Verdi anno_completo','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%             saveas(gcf,strcat('Parma Ponte Verdi anno_completo','_',num2str(w),'.png')); %salvataggio grafico formato .png
%         f=f+1;
%         figure(f); %plot 
%             plot(data,hidroC,'r')
%             hold on
%             plot(datae,dati(:,2),'k.')
%             grid on
%             grid minor 
%             title('Stazione idrometrica Colorno')
%             xlabel('Mesi/gg');
%             ylabel('Quota idrica [m]');
%             saveas(gcf,strcat('Colorno anno completo','_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%             saveas(gcf,strcat('Colorno anno completo','_',num2str(w),'.png')); %salvataggio grafico formato .png

% end
% %% Pendenza
% 
% y1=abs(diff(hidroPV)); 
% % calcolo pendendza media
% i=1;
% j=1;
% fin=DT/;
% while i<length(y1)
%     while j<fin && i+j<length(y1)
%         y1_1(j)=y1(i+j);
%         j=j+1;
%     end
%     y11(i,1)= mean (y1_1);
%     i=i+1;
%     j=1;
% end
% %selezione onde in base alla finestra di tempo e alla pendenza 
% i=1;
% k=1;
% j=1;
% o=1;
% while i<length(hidroPV_)
%     r=i+fin*2;
%     if r>length(hidroPV_)
%         r=length(hidroPV_)-1;
%     end
%     
%     while i<length(y1) && y11(i)>slope
%          while j<r && i<length(hidroPV_)+1 % per evitare che i scappi oltre alla lunghezza di hidroPV_
%             eval(['ondaPV' num2str(k) '(j,1)=hidroPV_(i);']);
%             eval(['dataonda' num2str(k) '(j,1)=data_(i);']);
%             eval(['ondaC' num2str(k) '(j,1)=hidroC_(i);']);
%        j=j+1;
%        i=i+1;
%         end
%         i=i+1;
%     end
%     if j>1
% %       f=f+1;
%       figure(f); %plot onde
%         plot(eval(['dataonda' num2str(k)]), eval(['ondaPV' num2str(k)]));
%         grid on 
%         grid minor 
%         title(strcat('Stazione idrometrica Parma Ponte Verdi, onda numero:',num2str(k),',anno:',num2str(w)))
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('ondaPV',num2str(k),'_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('ondaPV',num2str(k),'_',num2str(w),'.png')); %salvataggio grafico formato .png
%       f=f+1;
%       figure(f); 
%         plot(eval(['dataonda' num2str(k)]), eval(['ondaC' num2str(k)]));
%         grid on 
%         grid minor 
%         title(strcat('Stazione idrometrica di Colorno, onda numero:',num2str(k),',anno:',num2str(w)))
%         xlabel('Mesi/gg');
%         ylabel('Quota idrica [m]');
%         saveas(gcf,strcat('ondaC',num2str(k),'_',num2str(w),'.fig')); %salvataggio grafico formato .fig
%         saveas(gcf,strcat('ondaC',num2str(k),'_',num2str(w),'.png')); %salvataggio grafico formato .png
%         %salvataggio txt dati
%        da1=datevec(eval(['dataonda' num2str(k)]));
%        da1=[da1(:,3),da1(:,2),da1(:,1),da1(:,4),da1(:,5)];
%        dati=[da1,eval(['ondaPV' num2str(k)]),eval(['ondaC' num2str(k)])]; % [da1,eval(['ondaPV' num2str(k)]),eval(['ondaC' num2str(k)])]
%        filename=strcat('onda',num2str(k),'_',num2str(w),'.txt');
%        fileID = fopen(filename,'w');
%        fprintf(fileID,'%d/%d/%d %2d:%d %4.2f,%4.2f \n',dati'); % '%d/%d/%d,%2d:%d,%4.2f,%4.2f \n'

%       k=k+1;
%       j=1;
%     end
%     i=i+1;
% end
clearvars -except w f
end