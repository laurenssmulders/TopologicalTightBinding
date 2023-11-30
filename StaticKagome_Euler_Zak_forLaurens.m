% Kagome Hamiltonian static        

% Calculates the EulerInv for the singularites aroung Gamma point       
% non-Abelian Berry curvature  2nd&3rd bands (SurfaceTerm) and the LineIntegral around the BZ patch(BoundaryTerm)   

% nn hopping only, Gamma point is already Euler=1    


set(0,'DefaultFigureWindowStyle','docked')
set(groot,'defaultFigureCreateFcn',@(fig,~)addToolbarExplorationButtons(fig))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------------- INPUT -----------------------------------------
J=1;   %nn tunneling, -t below
J1=J;  J2=J;  J3=J;   %tunneling strength along three directions
JJ=0;  
JJJ=0;  

DeltaA=10;    %sublattice offset will be diag([DeltaA 0 DeltaC])
DeltaB=0;
DeltaC=-10;


kN=101;  %k-grid   for nonAbel calculation. will use same grid for integral and derivatives
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Hoffset=diag([DeltaA DeltaB DeltaC]);


%nearest neighbor vectors, within basis yani
d1=[1 0]/2;
d2=[1/2 sqrt(3)/2]/2;
d3=[-1/2 sqrt(3)/2]/2;
 
% lattice vectors, unit length a=1 for these
a1=2*d1;
a2=2*d2; 
a3=2*d3;


rA=a3/2;   %for origin at the center of hxgn, distance to A site. It's sign doesn't matter for V matrices below (we think) 
rB=a2/2;
rC=a1/2;


% reciprocal lattice vectors
b1=[2*pi -2*pi/sqrt(3)];
b2=[0 4*pi/sqrt(3)];


% reciprocal vector coefficients defining the BZ
alpha1=linspace(-.5,.5,kN); 
alpha2=linspace(-.5,.5,kN); %alpha1;
[Alpha1, Alpha2]=meshgrid(alpha1,alpha2);
kxRange=Alpha1*b1(1)+Alpha2*b2(1);  % BZ 
kyRange=Alpha1*b1(2)+Alpha2*b2(2);





% ----calculate and save e.st.s at grid points, and gauge fix if needed   
EnBands=zeros(3,size(kxRange,1),size(kxRange,2));
psi1set=zeros(3,size(kxRange,1),size(kxRange,2)); 
psi2set=zeros(3,size(kxRange,1),size(kxRange,2));   
psi3set=zeros(3,size(kxRange,1),size(kxRange,2));
for ck1=1:size(kxRange,2)   %along alpha1,b1 which is the 2nd dim in meshgrid
    for ck2=1:size(kxRange,1)  %along alpha2,b2 
        k=[kxRange(ck2,ck1); kyRange(ck2,ck1)];
        
        Hnnnn=JJJ*diag([cos((a3)*k) cos((a2)*k) cos((a1)*k)]);
        
        
        h1bare=-2*[0 J1*cos(a1*k/2)+JJ*cos((a3+a2)/2*k) 0; 0 0 0; 0 0 0];   %Hamiltonian along a single direction w/o lambda increase  
        h2bare=-2*[0 0 J2*cos(a2*k/2)+JJ*cos((a3-a1)/2*k); 0 0 0; 0 0 0];
        h3bare=-2*[0 0 0; 0 0 J3*cos(a3*k/2)+JJ*cos((a1+a2)/2*k); 0 0 0];
        H=h1bare+h2bare+h3bare +h1bare'+h2bare'+h3bare' +Hnnnn +Hoffset;  %full Hamiltonian        
        
        [Evec,D]=eig(H);      [D,ind]=sort(real(diag(D)));  
        EnBands(:,ck2,ck1)=D;   
        psi1=Evec(:,ind(1));
        psi2=Evec(:,ind(2));
        psi3=Evec(:,ind(3));
        
        if any(abs(1-diag(Evec'*Evec))>1e-6)  % abs(psi2'*psi3)>1e-6 || abs(psi2'*bareGst(:,ckx,cky))>1e-6 || abs(psi3'*bareGst(:,ckx,cky))>1e-6  %e.st.s are not orthogonal
            disp('schur')
            [Evec,T]=schur(Heff);   if max(abs(imag(T)))>1e-7,   error('this is imaginary');  end;    [Eval,ind]=sort(real(diag(T)));
            psi1=Evec(:,ind(1));   
            psi2=Evec(:,ind(2));
            psi3=Evec(:,ind(3));
        end
        
        psi1set(:,ck2,ck1)=psi1;  psi2set(:,ck2,ck1)=psi2;  psi3set(:,ck2,ck1)=psi3;
        
        
%         % ----- gauge fixing -----------
%         if cky~=1
%             if squeeze(psi2set(:,ckx,cky-1))'*psi2<0,  psi2=-psi2; end
%             if squeeze(psi3set(:,ckx,cky-1))'*psi3<0,  psi3=-psi3; end
%         else
%             if ckx~=1
%                 if squeeze(psi2set(:,ckx-1,cky))'*psi2<0,  psi2=-psi2; end
%                 if squeeze(psi3set(:,ckx-1,cky))'*psi3<0,  psi3=-psi3; end
%             end            
%         end        
%         psi2set(:,ckx,cky)=psi2;  psi3set(:,ckx,cky)=psi3;       
    end
end


figure;  
mesh(kxRange/pi,kyRange/pi,squeeze(EnBands(1,:,:)));   
hold on;    meshc(kxRange/pi,kyRange/pi,squeeze(EnBands(2,:,:))); 
hold on;    mesh(kxRange/pi,kyRange/pi,squeeze(EnBands(3,:,:))); 
xlabel('k_x/\pi'); ylabel('k_y/\pi');   zlabel('Energy'); 
title(['Static Kagome. JJ=' num2str(JJ) ',JJJ=' num2str(JJJ) '.\Delta=[' num2str(DeltaA) ',' num2str(DeltaB) ',' num2str(DeltaC) ']'])




max(abs(imag(psi1set(:)))), max(abs(imag(psi2set(:)))), max(abs(imag(psi3set(:))))





%% 
%% ----------------- Berry Phase calculation --------------------------------
kxRange=Alpha1*b1(1)+Alpha2*b2(1);  % BZ 
kyRange=Alpha1*b1(2)+Alpha2*b2(2);

% ----------- Along b2 -----------------------------------        
ck1=5;       % fixed index positions on @b1*alpha1(ck1). kxRange(:,ck1=1)=0, will increase ck2

Wmatrix=ones(1,3);    %for 1st,2nd,3rd e.st.s
% Uk=[psi1set(:,1,ck1) psi2set(:,1,ck1) psi3set(:,1,ck1)];   %E.st. matrix [psi1 psi2 psi3] at the starting point of b2,alpha2, so ck2=1
for ck2=2:(size(kxRange,1)-1)  %along alpha2,b2, excluding last point    
    Wmatrix(1)=psi1set(:,ck2,ck1)'*psi1set(:,ck2-1,ck1)*Wmatrix(1);
    Wmatrix(2)=psi2set(:,ck2,ck1)'*psi2set(:,ck2-1,ck1)*Wmatrix(2);
    Wmatrix(3)=psi3set(:,ck2,ck1)'*psi3set(:,ck2-1,ck1)*Wmatrix(3);
end
Vb2=diag([exp(-1i*rA*b2.') exp(-1i*rB*b2.') exp(-1i*rC*b2.')]);
% Uk_next=Vb2*[psi1set(:,1,ck1) psi2set(:,1,ck1) psi3set(:,1,ck1)];

Wmatrix(1)=(Vb2*psi1set(:,1,ck1))'*psi1set(:,size(kxRange,1)-1,ck1)*Wmatrix(1);
Wmatrix(2)=(Vb2*psi2set(:,1,ck1))'*psi2set(:,size(kxRange,1)-1,ck1)*Wmatrix(2);
Wmatrix(3)=(Vb2*psi3set(:,1,ck1))'*psi3set(:,size(kxRange,1)-1,ck1)*Wmatrix(3);

 
disp(['BerryPhases along b2=[' num2str(round(imag(log(eig(Wmatrix(1))))/pi,5)) ','...
                               num2str(round(imag(log(eig(Wmatrix(2))))/pi,5))  ',' ...
                               num2str(round(imag(log(eig(Wmatrix(3))))/pi,5))  ']*pi'])




% ----------- Along b1 -----------------------------------  
ck2=5;       % fixed index positions on @b2*alpha2(ck2). kxRange(ck2,:)=increases,kyRange(ck2,:)=decreases for increasing ck1

Wmatrix=ones(1,3);    %for 1st,2nd,3rd e.st.s
Uk=[psi1set(:,ck2,1), psi2set(:,ck2,1), psi3set(:,ck2,1)];   %E.st. matrix [psi1 psi2 psi3] at the starting point of b1,alpha1, so ck1=1
for ck1=1:(size(kxRange,2)-1)   %along alpha1,b1 which is the 2nd dim in meshgrid
    Uk_next=[psi1set(:,ck2,ck1), psi2set(:,ck2,ck1), psi3set(:,ck2,ck1)];
    
    Wmatrix(1)=Uk_next(:,1)'*Uk(:,1)*Wmatrix(1);
    Wmatrix(2)=Uk_next(:,2)'*Uk(:,2)*Wmatrix(2);
    Wmatrix(3)=Uk_next(:,3)'*Uk(:,3)*Wmatrix(3);
    
    Uk=Uk_next;   %for the next iteration
end
Vb1=diag([exp(-1i*rA*b1.') exp(-1i*rB*b1.') exp(-1i*rC*b1.')]);
Uk_next=Vb1*[psi1set(:,ck2,1) psi2set(:,ck2,1) psi3set(:,ck2,1)];

Wmatrix(1)=Uk_next(:,1)'*Uk(:,1)*Wmatrix(1);
Wmatrix(2)=Uk_next(:,2)'*Uk(:,2)*Wmatrix(2);
Wmatrix(3)=Uk_next(:,3)'*Uk(:,3)*Wmatrix(3);
 
disp(['BerryPhases along b1=[' num2str(round(imag(log(eig(Wmatrix(1))))/pi,5)) ','...
                               num2str(round(imag(log(eig(Wmatrix(2))))/pi,5))  ',' ...
                               num2str(round(imag(log(eig(Wmatrix(3))))/pi,5))  ']*pi'])




% ----------- Along b2 ----- in for loop -----------------------------------        
ck1_range=1:5:size(kxRange,2);       % fixed index positions on @b1*alpha1(ck1). kxRange(:,ck1=1)=0, will scan ck2 at each these ck1


Wmatrix_b1=ones(3,numel(ck1_range));     %for 1st,2nd,3rd e.st.s
Gamma_b1=ones(3,numel(ck1_range))*10;   %Berry phases from Wmatrix
for cn=1:numel(ck1_range)  
    ck1=ck1_range(cn);
    
    for ck2=2:(size(kxRange,1)-1)   %along alpha2,b2, excluding last point
        Wmatrix_b1(1,cn)=psi1set(:,ck2,ck1)'*psi1set(:,ck2-1,ck1)*Wmatrix_b1(1,cn);
        Wmatrix_b1(2,cn)=psi2set(:,ck2,ck1)'*psi2set(:,ck2-1,ck1)*Wmatrix_b1(2,cn);
        Wmatrix_b1(3,cn)=psi3set(:,ck2,ck1)'*psi3set(:,ck2-1,ck1)*Wmatrix_b1(3,cn);
    end
    Wmatrix_b1(1,cn)=(Vb2*psi1set(:,1,ck1))'*psi1set(:,size(kxRange,1)-1,ck1)*Wmatrix_b1(1,cn);
    Wmatrix_b1(2,cn)=(Vb2*psi2set(:,1,ck1))'*psi2set(:,size(kxRange,1)-1,ck1)*Wmatrix_b1(2,cn);
    Wmatrix_b1(3,cn)=(Vb2*psi3set(:,1,ck1))'*psi3set(:,size(kxRange,1)-1,ck1)*Wmatrix_b1(3,cn);
    
    Gamma_b1(1,cn)=imag(log(eig(Wmatrix_b1(1,cn))));
    Gamma_b1(2,cn)=imag(log(eig(Wmatrix_b1(2,cn))));
    Gamma_b1(3,cn)=imag(log(eig(Wmatrix_b1(3,cn))));
end 

figure;  
subplot(2,1,1);  plot(ck1_range/size(kxRange,2),abs(Gamma_b1(1,:))/pi,'k*-',ck1_range/size(kxRange,2),abs(Gamma_b1(2,:))/pi,'mo-',ck1_range/size(kxRange,2),abs(Gamma_b1(3,:))/pi,'bd-');  box on
xlabel('|b_1|');  ylabel('|Berry Phase| / \pi');   title(['Static Kagome. JJ=' num2str(JJ) ',JJJ=' num2str(JJJ) '.\Delta=[' num2str(DeltaA) ',' num2str(DeltaB) ',' num2str(DeltaC) ']'])
legend('Band 1','Band 2','Band 3')



% ----------- Along b1 ---- in for loop -------------------------------  
ck2_range=1:5:size(kxRange,1);       % fixed index positions on @b2*alpha2(ck2). kxRange(ck2,:)=increases,kyRange(ck2,:)=decreases for increasing ck1

Wmatrix_b2=ones(3,numel(ck2_range));     %for 1st,2nd,3rd e.st.s
Gamma_b2=ones(3,numel(ck2_range))*10;   %Berry phases from Wmatrix
for cn=1:numel(ck2_range)  
    ck2=ck2_range(cn);
    
    Uk=[psi1set(:,ck2,1), psi2set(:,ck2,1), psi3set(:,ck2,1)];   %E.st. matrix [psi1 psi2 psi3] at the starting point of b1,alpha1, so ck1=1
    for ck1=2:size(kxRange,2)   %along alpha1,b1 which is the 2nd dim in meshgrid
        Uk_next=[psi1set(:,ck2,ck1), psi2set(:,ck2,ck1), psi3set(:,ck2,ck1)];

        Wmatrix_b2(1,cn)=Uk_next(:,1)'*Uk(:,1)*Wmatrix_b2(1,cn);
        Wmatrix_b2(2,cn)=Uk_next(:,2)'*Uk(:,2)*Wmatrix_b2(2,cn);
        Wmatrix_b2(3,cn)=Uk_next(:,3)'*Uk(:,3)*Wmatrix_b2(3,cn);

        Uk=Uk_next;   %for the next iteration
    end
    Uk_next=Vb1*[psi1set(:,ck2,1) psi2set(:,ck2,1) psi3set(:,ck2,1)];
    Wmatrix_b2(1,cn)=Uk_next(:,1)'*Uk(:,1)*Wmatrix_b2(1,cn);
    Wmatrix_b2(2,cn)=Uk_next(:,2)'*Uk(:,2)*Wmatrix_b2(2,cn);
    Wmatrix_b2(3,cn)=Uk_next(:,3)'*Uk(:,3)*Wmatrix_b2(3,cn);
    
    Gamma_b2(1,cn)=imag(log(eig(Wmatrix_b2(1,cn))));
    Gamma_b2(2,cn)=imag(log(eig(Wmatrix_b2(2,cn))));
    Gamma_b2(3,cn)=imag(log(eig(Wmatrix_b2(3,cn))));
end

subplot(2,1,2);  plot(ck2_range/size(kxRange,1),abs(Gamma_b2(1,:))/pi,'k*-',ck2_range/size(kxRange,1),abs(Gamma_b2(2,:))/pi,'mo-',ck2_range/size(kxRange,1),abs(Gamma_b2(3,:))/pi,'bd-');  box on
xlabel('|b_2|');   ylabel('|Berry Phase| / \pi');  
legend('Band 1','Band 2','Band 3')











%% ============= THE PATCH EULER CLASS, NN Kagome only ===================== Gamma
kxRange_patch=linspace(-.5*pi,.5*pi,kN);    % Rectangular, around Gamma
kyRange_patch=kxRange_patch;  
dk=kxRange_patch(2)-kxRange_patch(1);  
kxExtd=[kxRange_patch  kxRange_patch(end)+dk];   
kyExtd=kxExtd;  %have kN+1 elements

% ----calculate and save e.st.s at grid points, and gauge fix
EnBands=zeros(3,kN+1,kN+1);
bareGst=zeros(3,kN+1,kN+1);    %no gauge fixing 
psi2set=zeros(3,kN+1,kN+1);   
psi3set=zeros(3,kN+1,kN+1); 
for ckx=1:kN+1
    for cky=1:kN+1
        k=[kxExtd(ckx); kyExtd(cky)];
        
        h1bare=-J1*2*[0 cos(a1*k/2) 0; cos(a1*k/2) 0 0; 0 0 0];   %Hamiltonian along a single direction w/o increase  
        h2bare=-J2*2*[0 0 cos(a2*k/2); 0 0 0; cos(a2*k/2) 0 0];
        h3bare=-J3*2*[0 0 0; 0 0 cos(a3*k/2); 0 cos(a3*k/2) 0];
        H=h1bare+h2bare+h3bare +Hoffset;  %full Hamiltonian        
        
        [Evec,D]=eig(H);      [D,ind]=sort(diag(D));  
        EnBands(:,ckx,cky)=D;   
        bareGst(:,ckx,cky)=Evec(:,ind(1));   %before any gauge-fixing attempt
        psi2=Evec(:,ind(2));
        psi3=Evec(:,ind(3));
        
        if any(abs(1-diag(Evec'*Evec))>1e-6)  % abs(psi2'*psi3)>1e-6 || abs(psi2'*bareGst(:,ckx,cky))>1e-6 || abs(psi3'*bareGst(:,ckx,cky))>1e-6  %e.st.s are not orthogonal
            disp('schur')
            [Evec,T]=schur(Heff);   if max(abs(imag(T)))>1e-7,   error('this is imaginary');  end;    [Eval,ind]=sort(real(diag(T)));
            bareGst(:,ckx,cky)=Evec(:,ind(1));   %before any gauge-fixing attempt
            psi2=Evec(:,ind(2));
            psi3=Evec(:,ind(3));
        end
        
        
        % ----- gauge fixing -----------
        if cky~=1
            if squeeze(psi2set(:,ckx,cky-1))'*psi2<0,  psi2=-psi2; end
            if squeeze(psi3set(:,ckx,cky-1))'*psi3<0,  psi3=-psi3; end
        else
            if ckx~=1
                if squeeze(psi2set(:,ckx-1,cky))'*psi2<0,  psi2=-psi2; end
                if squeeze(psi3set(:,ckx-1,cky))'*psi3<0,  psi3=-psi3; end
            end            
        end        
        psi2set(:,ckx,cky)=psi2;  psi3set(:,ckx,cky)=psi3;       
    end
end


figure;  
mesh(kxExtd,kyExtd,squeeze(EnBands(1,:,:))'/pi);   
hold on;    meshc(kxExtd,kyExtd,squeeze(EnBands(2,:,:))'/pi); 
hold on;    mesh(kxExtd,kyExtd,squeeze(EnBands(3,:,:))/pi); 
xlabel('k_x'); ylabel('k_y');  zlabel('Energy/\pi'); 
title('Static Kagome')


figure;  
surf(kxExtd,kyExtd,squeeze(EnBands(1,:,:))/pi);   
hold on;    surf(kxExtd,kyExtd,squeeze(EnBands(2,:,:))/pi); 
hold on;    surf(kxExtd,kyExtd,squeeze(EnBands(3,:,:))/pi); 
xlabel('k_x'); ylabel('k_y');  zlabel('Energy/\pi'); 
title('Static Kagome')



% ----- now calculate non-Abel. BerryCurv. -----------------------
Omega23=zeros(kN,kN);   %Euler integrand, i.e. non-Abelian Berry curv for 2nd&3rd bands
intg_x=zeros(1,kN);
for ckx=1:kN
    for cky=1:kN
        dx_psi2=( squeeze(psi2set(:,ckx+1,cky))-squeeze(psi2set(:,ckx,cky)) )/dk;    %der. of 2nd st. derk=dk 
        dy_psi2=( squeeze(psi2set(:,ckx,cky+1))-squeeze(psi2set(:,ckx,cky)) )/dk;
        
        dx_psi3=( squeeze(psi3set(:,ckx+1,cky))-squeeze(psi3set(:,ckx,cky)) )/dk;    %der. of 3rd st. derk=dk 
        dy_psi3=( squeeze(psi3set(:,ckx,cky+1))-squeeze(psi3set(:,ckx,cky)) )/dk;  
        
        Omega23(ckx,cky)=dx_psi2'*dy_psi3-dy_psi2'*dx_psi3;     
    end
    %trying to remove divergences in Omega23 whose range lets say btwn [-5 5] at largest   
    temp23=squeeze(Omega23(ckx,:));
    ind=find(temp23>10);
    for cn=1:numel(ind)
        if ind(cn)==1 
            if ckx>1, temp23(1)=Omega23(ckx-1,1); else, temp23(1)=temp23(5);   end  %if the first element is diverging, replace it w/ prev ckx, or if it's really the first element w/ the 5th cky element
        else, temp23(ind(cn))=temp23(ind(cn)-1);   %for all other diverging elements, replace it w/ prev element
        end
    end
    ind=find(temp23<-10); %0.33);
    for cn=1:numel(ind)
        if ind(cn)==1
            if ckx>1, temp23(1)=Omega23(ckx-1,1); else, temp23(1)=temp23(5);  end %if the first element is divergind, replace it w/ the 5th element
        else, temp23(ind(cn))=temp23(ind(cn)-1);   %for all other diverging elements, replace it w/ prev element
        end
    end
    Omega23(ckx,:)=temp23;    
    
%     if ckx>1   %tring to remove flat lines of really large numbers
%         if any(abs(Omega23(ckx,:)-Omega23(ckx-1,:))>9)
%             ckx,
%             Omega23(ckx,abs(Omega23(ckx,:)-Omega23(ckx-1,:))>9)=Omega23(ckx-1,abs(Omega23(ckx,:)-Omega23(ckx-1,:))>9);
%         end
%     end
    
    intg_x(ckx)=sum( (Omega23(ckx,1:kN-1)+Omega23(ckx,2:kN))/2 )*dk;   %take ky-integral
end
putvar(kxRange_patch,kyRange_patch,Omega23)

if max(abs(imag(Omega23(:))))>1e-8, warning('\Omega_23 is imaginary'); end
figure;   surf(kxRange_patch/pi,kyRange_patch/pi,real(Omega23).');      %zlim([0 2])
xlabel('k_x/\pi');  ylabel('k_y/\pi');  zlabel('nonAbel. BerryCurv._{23} around \Gamma');    drawnow


SurfTerm=1/2/pi*sum( (intg_x(1:kN-1)+intg_x(2:kN))/2 )*dk,








% ------ Boundary Term around the BZ patch Omega23 is calculated -------
A23bottom=zeros(1,kN);    %Berry conn. <psi2|dk psi3>, integrand of the boundary term
for ckx=1:kN  %going right at bottom edge cky=1     
    A23bottom(ckx)=(squeeze(psi2set(:,ckx,1)))'*( squeeze(psi3set(:,ckx+1,1))-squeeze(psi3set(:,ckx,1)) )/dk;     
end
if max(abs(imag(A23bottom)))>1e-8, warning('A23 is imaginary'); end
hold on; plot3(kxRange_patch/pi,kyRange_patch(1)*ones(1,kN)/pi,real(A23bottom),'r.')
BoundaryTerm=sum( (A23bottom(1:kN-1)+A23bottom(2:kN))/2 )*dk; 


A23right=zeros(1,kN);    %Berry conn. for rigth edge upward
for cky=1:kN       %going up on right edge, ckx=kNx
    A23right(cky)=(squeeze(psi2set(:,kN,cky)))'*( squeeze(psi3set(:,kN,cky+1))-squeeze(psi3set(:,kN,cky)) )/dk;
end
if max(abs(imag(A23right)))>1e-8, warning('A23 is imaginary'); end
hold on; plot3(kxRange_patch(kN)*ones(1,kN)/pi,kyRange_patch/pi,real(A23right),'b.')
BoundaryTerm=BoundaryTerm +sum( (A23right(1:kN-1)+A23right(2:kN))/2 )*dk; 


A23top=zeros(1,kN-1);    %Berry conn. for top edge, moving leftward
for ckx=1:kN-1  %going left at top edge cky=kNx   
    A23top(ckx)=(squeeze(psi2set(:,kN-ckx+1,kN)))'*( squeeze(psi3set(:,kN-ckx,kN))-squeeze(psi3set(:,kN-ckx+1,kN)) )/dk;  
end
% --- 
k=[kxRange_patch(1)-dk; kyRange_patch(kN)];
h1bare=-J1*2*[0 cos(a1*k/2) 0; cos(a1*k/2) 0 0; 0 0 0];   %Hamiltonian along a single direction w/o lambda increase  
h2bare=-J2*2*[0 0 cos(a2*k/2); 0 0 0; cos(a2*k/2) 0 0];
h3bare=-J3*2*[0 0 0; 0 0 cos(a3*k/2); 0 cos(a3*k/2) 0];
H=h1bare+h2bare+h3bare;  %full Hamiltonian
[Evec,Eval]=eig(H);    if max(abs(imag(Eval)))>1e-7,   error('this is imaginary');  end;    Eval=real(Eval);
[~,ind]=sort(diag(Eval));
psi3=Evec(:,ind(3));  %3rd band
if squeeze(psi3set(:,1,kN))'*psi3<0,  psi3=-psi3; end
A23top(kN)=(squeeze(psi2set(:,1,kN)))'*( psi3-squeeze(psi3set(:,1,kN)) )/dk;   %last data point of A23top@top left corner
if max(abs(imag(A23top)))>1e-8, warning('A23 is imaginary'); end
hold on; plot3(kxRange_patch(kN:-1:1)/pi,kyRange_patch(kN)*ones(1,kN)/pi,real(A23top),'r.')
BoundaryTerm=BoundaryTerm +sum( (A23top(1:kN-1)+A23top(2:kN))/2 )*dk;


A23left=zeros(1,kN-1);    %Berry conn. for left edge, moving down
for cky=1:kN-1      %going down on left edge, ckx=1
    A23left(cky)=(squeeze(psi2set(:,1,kN-cky+1)))'*( squeeze(psi3set(:,1,kN-cky))-squeeze(psi3set(:,1,kN-cky+1)) )/dk;
end
% --- 
k=[kxRange_patch(1); kyRange_patch(1)-dk];
h1bare=-J1*2*[0 cos(a1*k/2) 0; cos(a1*k/2) 0 0; 0 0 0];   %Hamiltonian along a single direction w/o lambda increase  
h2bare=-J2*2*[0 0 cos(a2*k/2); 0 0 0; cos(a2*k/2) 0 0];
h3bare=-J3*2*[0 0 0; 0 0 cos(a3*k/2); 0 cos(a3*k/2) 0];
H=h1bare+h2bare+h3bare;  %full Hamiltonian 
[Evec,Eval]=eig(H);    if max(abs(imag(Eval)))>1e-7,   error('this is imaginary');  end;    Eval=real(Eval);
[~,ind]=sort(diag(Eval));
psi3=Evec(:,ind(3));  %3rd band
if squeeze(psi3set(:,1,1))'*psi3<0,  psi3=-psi3; end
A23left(kN)=(squeeze(psi2set(:,1,1)))'*( psi3-squeeze(psi3set(:,1,1)) )/dk;
if max(abs(imag(A23left)))>1e-8, warning('A23 is imaginary'); end
hold on; plot3(kxRange_patch(1)*ones(1,kN)/pi,kyRange_patch(kN:-1:1)/pi,real(A23left),'r.')
BoundaryTerm=(BoundaryTerm +sum( (A23left(1:kN-1)+A23left(2:kN))/2 )*dk )/2/pi


Euler23_nonAbel=SurfTerm-BoundaryTerm

title(['(SurfIntg=' num2str(round(SurfTerm,3)) ')-(BoundaryTerm=' num2str(round(BoundaryTerm,3)) ') = ' num2str(round(Euler23_nonAbel,3))])







