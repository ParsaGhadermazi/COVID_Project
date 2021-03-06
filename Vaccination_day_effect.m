S01= 11080000/118
I0=1
S0=S01/S01
I0=I0/S01
k1=0.588
k2=0.0612
k3=1.4*10^(-8)
% k4=linspace(0,1,1000)
k4=0.032;
t=linspace(0,100,10000);
params(1)=k1
params(2)=k2
params(3)=k3

params(5)=S0
params(6)=I0
Days=[10 20 28 29 30 31 40 50 100]


for i=1:length(Days)
    params(4)=0     
    [time1,Conc1]=ode45(@System,linspace(0,Days(i),1000),[S0,I0],[],params);
    params(4)=k4
    [time2,Conc2]=ode45(@System,linspace(Days(i),101,1000),Conc1(end,:),[],params)
    time=[time1;time2]
    Conc=[Conc1;Conc2]
    Ans(i).C=Conc;
    Ans(i).t=time;
        
end     

% for i=1:length(k4)
%     maximum(i)=max(Concentrations(i).Data(:,2))
% end

% hold on
% for i=1:length(k4)
%     plot(k4,maximum)
% end
hold on

for i=1:length(Days)
   
    plot(Ans(i).t(:),Ans(i).C(:,2)*S01)

end
for i=1:length(Days)
   Leg{i}= strcat('Vaccination start ',num2str(Days(i)))
end
legend(Leg)
title('Curve behavior for different vaccination Dates')
ylabel('Active Cases')
xlabel('Days')

function [dcdt]=System(t,C,params)

k1=params(1);k2=params(2); k3=params(3); k4=params(4);
S0=params(5);I0=params(6);
dcdt(1)=-k1*C(1)*C(2)+k3*(S0+I0-C(1)-C(2))-k4*C(1);
dcdt(2)=k1*C(1)*C(2)-k2*C(2);
dcdt=dcdt';


end


