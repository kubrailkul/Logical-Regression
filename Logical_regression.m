df=importdata('Data1.csv');

x1=df(:,1);
x2=df(:,2);
y=df(:,3);
sum1=0;
sum2=0;


for i=1:100
  if y(i)==1
      sum1=sum1+1;
       plot(x1(i),x2(i),'kx','Color','r','MarkerSize',4)
       hold on
          class1(sum1,1)=(x1(i));
          class1(sum1,2)=(x2(i));
     
  else 
      sum2=sum2+1;
      plot(x1(i),x2(i),'ko','Color','b','MarkerSize',4)
       hold on
      
        classminus1(sum2,1)=(x1(i));
        classminus1(sum2,2)=(x2(i));
          
  end


end



X0=ones((sum2+sum1),1);
X1=[class1(:,1);classminus1(:,1)];
X2=[class1(:,2);classminus1(:,2)];
X_general_unnormalize=[X0 X1 X2];
X_general=[ X0 mat2gray(X1) mat2gray(X2)];


Ygen=[zeros(size(class1(:,1))) ; ones(size(classminus1(:,1)))];


training_number=length(X0);

theta=randi([20,200],3,1); %initial random values of parameters.

error_func=-1./(training_number).*(Ygen.*log(1./(1+exp(-(theta(1,1).*X_general(:,1)+theta(2,1).*X_general(:,2)+theta(3,1).*X_general(:,3)))))+(1-Ygen).*log(1-1./(1+exp(-(theta(1,1).*X_general(:,1)+theta(2,1).*X_general(:,2)+theta(3,1).*X_general(:,3))))));
learning_rate=16;
epsilon=1e-10*ones(length(X0),1);
iteration=0;
maxiteration=800000;

%This loop part is required to find suitable paramaters for model. 
while   sum(abs(error_func(:,1))>epsilon)~=0  && (maxiteration>iteration)
    iteration=iteration+1;
    for i=1:1:training_number
        temp1=theta(1,1)-learning_rate*(1/(1+exp(-(theta(1,1)*X_general(i,1)+theta(2,1)*X_general(i,2)+theta(3,1)*X_general(i,3))))-Ygen(i,1))*X_general(i,1);
        temp2=theta(2,1)-learning_rate*(1/(1+exp(-(theta(1,1)*X_general(i,1)+theta(2,1)*X_general(i,2)+theta(3,1)*X_general(i,3))))-Ygen(i,1))*X_general(i,2);
        temp3=theta(3,1)-learning_rate*(1/(1+exp(-(theta(1,1)*X_general(i,1)+theta(2,1)*X_general(i,2)+theta(3,1)*X_general(i,3))))-Ygen(i,1))*X_general(i,3);
        
         
        theta(1,1)=temp1;
        theta(2,1)=temp2;
        theta(3,1)=temp3;
          
         
       error_func(i,1)=-1/(training_number)*(Ygen(i,1)*log(1/(1+exp(-(theta(1,1)*X_general(i,1)+theta(2,1)*X_general(i,2)+theta(3,1)*X_general(i,3)))))+(1-Ygen(i,1))*log(1-1/(1+exp(-(theta(1,1)*X_general(i,1)+theta(2,1)*X_general(i,2)+theta(3,1)*X_general(i,3))))));
          

    end
end

%classification border 
  h_theta_x=  theta(1,1).*X_general(:,1)+theta(2,1).*X_general(:,2)+(-(theta(1,1)/theta(3,1)))-((theta(2,1)/theta(3,1)).*X_general(:,2));
 
         plot(X1,h_theta_x,'Color','g');
   hold on;
   