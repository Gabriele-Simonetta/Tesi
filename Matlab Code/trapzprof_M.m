%% SECTION TITLE
% DESCRIPTIVE TEXT
clear all
clc
Profilo=load('PV.txt');
Onda=load(
ma=max(Profilo(:,2));
mi=min(Profilo(:,2));
k=0;
for z=mi:0.1:ma
for i=1:length(Profilo)
if Profilo(i,2)<=z
    x(i)=Profilo(i,1);
    y(i)= abs(Profilo(i,2)-z);
end
end
I=trapz(x,y);
k=k+1;
V(k,:)=[z-mi;I];
% figure(1);
% plot(x,y);
% grid on
% grid minor
figure(2);
plot(Profilo(:,1),Profilo(:,2));
grid on
end