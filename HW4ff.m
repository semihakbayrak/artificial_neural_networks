function HW4ff()
lambda = 1.4; %scaling factor given to us
R1 = 1; %R1,R2,C1,C2 values to form the dynamic equation
R2 = 1;
C1 = 1;
C2 = 1;
t12 = 1; %matrix values given to us 
t21 = 1;
    function y = activate(x)
        y = 2/pi * atan(pi*x*lambda/2); %activation function g(.)
    end
    revactivate= @(x) 2/(pi*lambda) * tan(pi*x/2); %inverse of activation function
    %Energy function
    function E = Energy(v1,v2)
        E = -1/2*(v1*v2+v2*v1) + integral(revactivate,0,v1)/R1 + integral(revactivate,0,v2)/R2;
    end
u = [0.1 -0.05]; %initial potential values in the 4th quadrant
v = activate(u); %output of neurons
delta = [0 0]; %will be using to store the changes in potentials
delta(1) = v(2)*t21/C1 - u(1)/R1; %dynamic equations for the network
delta(2) = v(1)*t12/C2 - u(2)/R2;
vone = [];
vtwo = [];
vone(1) = v(1);
vtwo(1) = v(2);
E = Energy(v(1),v(2));
Ematrix = [];
Ematrix(1) = E;
figure(1)
plot(v(1),v(2),'ro');
hold on;
%main algorithm
i = 0;
while 1
    i = i+1;
    u1 = u(1) + delta(1);
    u2 = u(2) + delta(2);
    u = [u1 u2];
    v = activate(u)
    delta(1) = v(2)/C1 - u(1)/R1; %dynamic equations for the network
    delta(2) = v(1)/C2 - u(2)/R2;
    E = Energy(v(1),v(2))
    vone(i+1) = v(1);
    vtwo(i+1) = v(2);
    Ematrix(i+1) = E;
    plot(v(1),v(2),'ro');
    hold on;
    if (Ematrix(i+1)-Ematrix(i))<0.0001
        break
    end
end
plot3(vone,vtwo,Ematrix)
figure(2)
plot3(vone,vtwo,Ematrix,'Linewidth',1.2)
figure(3)
EM = zeros(i+1,i+1);
ls1 = linspace(1,-1,i+1);
ls2 = linspace(1,-1,i+1);
for j=1:(i+1)
    for k=1:(i+1)
        if(j==k)
            continue;
        end
        EM(j,k)=Energy(ls1(j),-ls2(k));
    end
end
contour(ls1,ls2,EM,'ShowText','on');
end