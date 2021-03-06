S01= 11080000/118
I0=1
S0=S01/S01
I0=I0/S01
k1=0.588
k2=0.0612
k3=1.4*10^(-8)
% k4=linspace(0,1,1000)
k4=[0.001 0.002 0.004 0.008 0.016 0.032]
t=linspace(0,100,10000);
params(1)=k1
params(2)=k2
params(3)=k3
params(4)=0
params(5)=S0
params(6)=I0
% 
for i=1:length(k4)
   params(4)=k4(i)
   [time,Conc]=ode45(@System,t,[S0,I0],[],params);
   Concentrations(i).Data=Conc*S01;
    
end

% for i=1:length(k4)
%     maximum(i)=max(Concentrations(i).Data(:,2))
% end

% hold on
% for i=1:length(k4)
%     plot(k4,maximum)
% end
hold on

for i=1:length(k4)
   
    plot(t(:),Concentrations(i).Data(:,2))
    legend(strcat('k4= ',num2str(k4(i))))
end
for i=1:length(k4)
   Leg{i}= strcat('k4= ',num2str(k4(i)))
end
legend(Leg)
title('Curve behavior for different vaccination rates')
ylabel('Active Cases')
xlabel('Days')

function [dcdt]=System(t,C,params)

k1=params(1);k2=params(2); k3=params(3); k4=params(4);
S0=params(5);I0=params(6);
dcdt(1)=-k1*C(1)*C(2)+k3*(S0+I0-C(1)-C(2))-k4*C(1);
dcdt(2)=k1*C(1)*C(2)-k2*C(2);
dcdt=dcdt';


end
