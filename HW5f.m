figure
sphere
hold on
x=[];
y=[];
z=[];
for i=1:90
    if i<=30           %for data vectors in the first category
        min1 = 0;
        max1 = 2/20;
        min2 = 13/20;
        max2 = 15/20;
    elseif i<=60       %for data vectors in the second category
        min1 = 3/20;
        max1 = 5/20;
        min2 = 18/20;
        max2 = 20/20;
    else               %for data vectors in the third category
        min1 = 6/20;
        max1 = 8/20;
        min2 = 13/20;
        max2 = 15/20;
    end
    alfa = 2*pi*((max1-min1)*rand+min1); %alfa and beta angles to define x,y,z compounds of datas
    beta = pi*((max2-min2)*rand+min2);   
    z(i) = sin(beta);
    x(i) = cos(beta)*cos(alfa);
    y(i) = cos(beta)*sin(alfa);
    %plotting input datas
    plot3(x,y,z,'bo')
end
hold on
%drawing 3 weights with length 1 randomly
w1=[];
w2=[];
w3=[];
alfa = 2*pi*rand(1,3);
beta = pi*rand(1,3);
w1(3) = sin(beta(1));
w1(1) = cos(beta(1))*cos(alfa(1));
w1(2) = cos(beta(1))*sin(alfa(1));
w2(3) = sin(beta(2));
w2(1) = cos(beta(2))*cos(alfa(2));
w2(2) = cos(beta(2))*sin(alfa(2));
w3(3) = sin(beta(3));
w3(1) = cos(beta(3))*cos(alfa(3));
w3(2) = cos(beta(3))*sin(alfa(3));
%plotting initial weights
plot3(w1(1),w1(2),w1(3),'m*','markers',50)
hold on
plot3(w2(1),w2(2),w2(3),'m*','markers',50)
hold on
plot3(w3(1),w3(2),w3(3),'m*','markers',50)
hold on
n = 0.2; %learning coefficient
%main algorithm
for i=1:90
    o=[0 0 0];
    zay = [x(i) y(i) z(i)];
    %3 output values
    o(1) = x(i)*w1(1) + y(i)*w1(2) + z(i)*w1(3);
    o(2) = x(i)*w2(1) + y(i)*w2(2) + z(i)*w2(3);
    o(3) = x(i)*w3(1) + y(i)*w3(2) + z(i)*w3(3);
    %determination of winner output and weight updating with normalization
    if o(1)>o(2)
        if o(1)>o(3)
            deltaw = n*(zay-w1);
            w1 = w1 + deltaw;
            w1 = w1/sqrt(w1(1)^2+w1(2)^2+w1(3)^2)
        else
            deltaw = n*(zay-w3);
            w3 = w3 + deltaw;
            w3 = w3/sqrt(w3(1)^2+w3(2)^2+w3(3)^2)
        end
    else
        if o(2)>o(3)
            deltaw = n*(zay-w2);
            w2 = w2 + deltaw;
            w2 = w2/sqrt(w2(1)^2+w2(2)^2+w2(3)^2)
        else
            deltaw = n*(zay-w3);
            w3 = w3 + deltaw;
            w3 = w3/sqrt(w3(1)^2+w3(2)^2+w3(3)^2)
        end
    end
    %plotting weight updates
    plot3(w1(1),w1(2),w1(3),'k*')
    hold on
    plot3(w2(1),w2(2),w2(3),'k*')
    hold on
    plot3(w3(1),w3(2),w3(3),'k*')
    hold on
end
%final weights in the plot
plot3(w1(1),w1(2),w1(3),'w*','markers',50)
hold on
plot3(w2(1),w2(2),w2(3),'w*','markers',50)
hold on
plot3(w3(1),w3(2),w3(3),'w*','markers',50)