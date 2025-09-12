% Clear workspace and command window
clear all
close all
clc
%------- Input and Output file names
InputFile="C:\Users\rohit\OneDrive\Desktop\term project power system\ISO-NE-case3\ISO-NE_case3.csv";
OutFile='Report.txt';

%--- Constants ---- do not modify them if not necessary
typeAllowed={'F','VM','VA','IM','IA'};  %!!! do not change the sequence
dimAllowed={'HZ','KV','DEG','AMP','DEG'}; %!!! do not change the seqience
typeAllowedE={'Bus','Ln','Gen','Load','HVDC','FACTS'};  % allowed types of transmission elements
separatorAllowed={':'};
firstColumn={'Time';'T';'sec'}; %!!! do not change the seqience
thresholdIm=[0 3000]; % min and max AMP threshold for current sanity check
thresholdVm=[0.2 900]; % min and max KV threshold for voltage sanity check
thresholdF=1.0; % HZ threshold for frequency sanity check
F_cutoff=0.1; % Hz  low-band cutoff frequency for spectra plotting
%--------------------------------------------------------

disp('Start of data analysis')
tic

fid = fopen(OutFile,'wt'); % Open outut file
if exist(InputFile, 'file')~=2
    % PMU file does not exist
    fprintf(fid, ' *** Error: PMU file does not exist:   %s \n',InputFile);
    fprintf(' *** Error: PMU file does not exist:   %s \n',InputFile);
    fclose(fid) ;    
end
% PMU file exists
fprintf(fid, '--- Loading data from file: %s  \n\n',InputFile);
 
%--- Load PMU data   
[num,txt] = xlsread(InputFile);
    
%--- If last column(s) of data contain all NaN then Matlab does not read them
%   need to add columns with NaNs
[~,col]=size(txt);
[samples,j]=size(num);
if col>j; num(:,j+1:col)=NaN; end
       
%--- Assign specific lines of title and time vector
LineFull=txt(1, 2:end)';
typeS=txt(2, 2:end)';
dimS=txt(3, 2:end)';
LineShort=txt(4, 2:end)';
Time=num(:,1); % time
col=col-1; % the number of PMU data columns

Tstart=txt{4,1}; % Start date/time of PMU data period

%--- Verify the syntax of the first column header
if ~cellfun(@strcmp, txt(1,1),firstColumn(1))
    fprintf(fid, '*** Warning: First comumn header is   %s   instread of   %s\n',txt{1,1},firstColumn{1});
end
if ~cellfun(@strcmp, txt(2,1),firstColumn(2))
    fprintf(fid, '*** Warning: First comumn header is   %s   instread of   %s\n',txt{2,1},firstColumn{2});
end
if ~cellfun(@strcmp, txt(3,1),firstColumn(3))
    fprintf(fid, '*** Warning: First comumn header is   %s   instread of   %s\n',txt{3,1},firstColumn{3});
end

% -- Verify the dimension of measurements in header
[~,temp]=ismember(dimS,dimAllowed);
for i=1:length(dimS)
    if temp(i)==0
        fprintf(fid, '*** Warning: not allowed dimension of measurements  %s  in column %4.0f \n',dimS{i},(i+1));
    end 
end

%--- Verify the type of measurements in header
[~,temp]=ismember(typeS,typeAllowed);
for i=1:length(typeS)
    if temp(i)==0
        fprintf(fid, '*** Warning: not allowed type of measurements  %s  in column %4.0f \n',typeS{i},(i+1));
    end 
end

% -- Verify the match of type and dimension of measurements
for i=1:col
    for j=1:length(dimAllowed)
      if cellfun(@strcmp, typeAllowed(j),typeS(i))
         if ~cellfun(@strcmp, dimAllowed(j),dimS(i))
           fprintf(fid, '*** Error: not allowed combination of type  %s  and dimension  %s of measurements in column %4.0f \n',typeS{i},dimS{i},(i+1));
         end 
      end
    end
end

%--- Verify time vector
for i=1:samples-1; dT(i)=Time(i+1)-Time(i); end % time increments between samples
fs=floor(1/median(dT)+0.4);
if fs <30
    fprintf(fid, '*** Error: sampling rate is less than 30 fps: %4.0f \n',fs);
end
for i=1:samples-1
    if dT(i)>1/fs*1.1  %   10% tolerance
        fprintf(fid, '*** Error: time interval at sample %5.0f  (%4.4f) exceeds time for sampling rate %3.4f\n',i,abs(dT(i)),1/fs);
    end
end
for i=1:samples-1
    if dT(i)<0 
        fprintf(fid, '*** Error: time interval at sample %5.0f is negative: %5.3f\n',i,dT(i));
    end
end

%--- Verify the structure of SignalName (first line of header) per the number of separators
for i=1:col
    ind = find(LineFull{i,1}==separatorAllowed{1,1});
    if length(ind)<3
        fprintf(fid, '*** Warning: not recommended SignalName structure in the first line of header: %s   column %5.0f \n',LineFull{i,1},i);
    else
        % Verify the type of transmission element
        if ~ismember(typeAllowedE,{LineFull{i,1}(ind(2)+1:ind(3)-1)})
           fprintf(fid, '*** Warning: non-standard type of transmission element measurements  %s  in column %4.0f \n',LineFull{i,1}(ind(2)+1:ind(3)-1),(i+1));
        end
    end
end
 
F=[]; Vm=[]; Va=[]; Im=[]; Ia=[];
indF=[]; indVm=[]; indVa=[]; indIm=[]; indIa=[];
%--- Sepatate signals into corresponding matrices
for i=1:col
   switch temp(i)
        case 1
          F=[F num(:,i+1)]; indF=[indF i+1];
        case 2
          Vm=[Vm num(:,i+1)]; indVm=[indVm i+1];
        case 3
          Va=[Va num(:,i+1)]; indVa=[indVa i+1];
        case 4
          Im=[Im num(:,i+1)]; indIm=[indIm i+1];
        case 5
          Ia=[Ia num(:,i+1)]; indIa=[indIa i+1];
   end
end
Nsig=zeros(1,5); % number of signals 1-F, 2-Vm, 3-Va, 4-Im,5-Ia
if ~isempty(F); Nsig(1)=length(indF); end
if ~isempty(Vm); Nsig(2)=length(indVm); end
if ~isempty(Va); Nsig(3)=length(indVa); end
if ~isempty(Im); Nsig(4)=length(indIm); end
if ~isempty(Ia); Nsig(5)=length(indIa); end
fprintf(fid, 'Input data set consists of: \n');
fprintf(fid, ' %5.0f  frequency signals \n',Nsig(1));
fprintf(fid, ' %5.0f  voltage magnitude signals \n',Nsig(2));
fprintf(fid, ' %5.0f  voltage angle signals \n',Nsig(3));
fprintf(fid, ' %5.0f  current magnitude signals \n',Nsig(4));
fprintf(fid, ' %5.0f  current angle signals \n',Nsig(5));
fprintf(fid, '\n');

%--- Check the consistency of I and V phasors by the number of signals
if Nsig(2)~= Nsig(3)
    fprintf(fid, '\n*** Warning: the number of voltage magnitude and phase signals is not the same\n');
end
if Nsig(4)~= Nsig(5) 
    fprintf(fid, '\n*** Warning: the number of current magnitude and phase signals is not the same\n');
end

%--- Verify an ability to calculate MW and Mvar flow for lines with given current
%=== Count lines with I and fill in LineID for thse lines and Substation IDs for all signals
if Nsig(4)==0
    fprintf(fid, '*** Warning: data set does not contain current signals\n');
else
   for i=1:col
       buf1=find(LineFull{i,1}==separatorAllowed{1,1});
       Sub{i,1}=LineFull{i,1}(buf1(1)+1:buf1(2)-1); % name of Substation 
   end
%=== Fill in column indices of input signal for lines with current
%    ind: 1-Im 2-Ia 3-f 4-Vm 5-Va
   ind=zeros(Nsig(4),5);
   ind(:,1)=find(temp==4); % indices of Im
   LineID(:,1)=LineFull(ind(:,1));

   for j=1:Nsig(4) % loop per the number of lines with current
      for i=1:col
        if cellfun(@strcmp, LineFull(i),LineID(j)) & cellfun(@strcmp, typeS(i),typeAllowed(5)); ind(j,2)=i; end % Ia
        if cellfun(@strcmp, LineFull(i),LineID(j)) & cellfun(@strcmp, typeS(i),typeAllowed(1)); ind(j,3)=i; end % F
        if cellfun(@strcmp, LineFull(i),LineID(j)) & cellfun(@strcmp, typeS(i),typeAllowed(2)); ind(j,4)=i; end % Vm
        if cellfun(@strcmp, LineFull(i),LineID(j)) & cellfun(@strcmp, typeS(i),typeAllowed(3)); ind(j,5)=i; end % Va
      end
   end
% For a current signal, fill in indices for F, Vm, Va from the same substation if F or V is not in input
% Use the quantity at the same substation listed first in input
   for j=1:Nsig(4)
      if ind(j,3)==0 % Line with current in input data does not have related frequency
          temp1=find(ismember(Sub,Sub(ind(j,1)))& temp==1);
          if ~isempty(temp1)
            ind(j,3)=temp1(1);
            fprintf(fid, '> Frequency for line  %s  can be linked to substation  %s (column %4.0f) \n',LineID{j},Sub{ind(j,3)},(ind(j,3)+1));
          end
      end
      if ind(j,4)==0 % Line with current in input data does not have related voltage magnitude
          temp1=find(ismember(Sub,Sub(ind(j,1)))& temp==2);
          if ~isempty(temp1)
            ind(j,4)=temp1(1);
            fprintf(fid, '> Voltage magnitude for line  %s   can be linked to substation  %s (column %4.0f) \n',LineID{j},Sub{ind(j,4)},(ind(j,4)+1));
          end
      end
      if ind(j,5)==0 % Line with current in input data does not have related voltage angle
          temp1=find(ismember(Sub,Sub(ind(j,1)))& temp==3);
          if ~isempty(temp1)
            ind(j,5)=temp1(1);
            fprintf(fid, '> Voltage angle for line  %s   can be linked to substation  %s (column %4.0f) \n',LineID{j},Sub{ind(j,5)},(ind(j,5)+1));
          end
      end        
   end
   
   %--- Check the completness of the data set for each line with current
   %    A complete set includes Im Ia F Vm Va
   temp1=sum(ismember(ind,0)');
   Nll=sum(ismember(temp1,0)'); % number of lines with full set of data
   if Nll>0
       fprintf(fid, '\n--There are %4.0f transmission elements with full data set(f,Vm,Va,Im,Ia) \n',Nll);
       for i=1:Nsig(4)
           if temp1(i)==0;fprintf(fid, '%s \n',LineID{i}); end
       end
   else
       fprintf(fid, '\n*** Warning: there are no transmission elements with full data set(f,Vm,Va,Im,Ia) \n');
   end
   fprintf(fid, '\n');
   for i=1:Nsig(4)
      if ind(i,2)==0; fprintf(fid, ' *** Warning: current angle for line  %s  is not in data set \n',LineID{i});  end
      if ind(i,3)==0; fprintf(fid, ' *** Warning: frequency for line  %s  is not in data set \n',LineID{i});  end
      if ind(i,4)==0; fprintf(fid, ' *** Warning: voltage magnitude for line  %s  is not in data set \n',LineID{i}); end
      if ind(i,5)==0; fprintf(fid, ' *** Warning: voltage angle for line  %s  is not in data set \n',LineID{i}); end      
   end
end

%--- Estimate frequency of the system
if Nsig(1)>0
   Fmed= median(F,'omitnan');
   Fmm=median(Fmed,'omitnan');
   Fsys=0;
   if Fmm >50-thresholdF & Fmm <50+thresholdF;    Fsys=50.0; end
   if Fmm >60-thresholdF & Fmm <60+thresholdF;    Fsys=60.0; end
   if Fsys==0; fprintf(fid, '*** Error: system frequency is not 50 nor 60 Hz\n'); end

   %--- Check the median frequency  per threshold
   for i=1:Nsig(1)
      if Fmed(i)~=NaN & (Fmed(i)<Fsys-thresholdF | Fmed(i)>Fsys+thresholdF)
        fprintf(fid, '*** Warning: median frequency in column  %5.0f is out of range; median is: %4.2f\n',indF(i),Fmed(i));
      end
   end
end

%--- Check the median of current magnitude  per threshold
if Nsig(4)>0
   for i=1:length(Im(1,:))
       temp=mean(Im(:,i),'omitnan');
       if temp<thresholdIm(1) | temp>thresholdIm(2)
          fprintf(fid, '*** Warning: Avarage magnitude of current in column  %5.0f is out of range; average value is: %5.2f\n',indIm(i),temp);
       end
   end
end

%--- Check the median of voltage magnitude  per threshold
if Nsig(2)>0
   for i=1:length(Vm(1,:))
       temp=mean(Vm(:,i),'omitnan');
       if temp<thresholdVm(1) | temp>thresholdVm(2)
          fprintf(fid, '*** Warning: Avarage magnitude of voltage in column  %5.0f is out of range; average value is: %5.2f\n',indVm(i),temp);
       end
   end
end

%--- Unwrap angles and remove average per measurement frame
% Unwarap angles
if Nsig(3)>0
   Va_un = unwrap(Va(:,1:end)/180*pi)*180/pi; % unwrapping must be done in radians but not in degrees
   Va_relative=Va_un-mean(Va_un','omitnan')';
end
if Nsig(5)>0
   Ia_un = unwrap(Ia(:,1:end)/180*pi)*180/pi;
   Ia_relative=Ia_un-mean(Ia_un','omitnan')';
end

%--- Estimate spectrum of oscillations from Vm and Im
f = fs/2*linspace(0,1,samples/2+1)'; %frequency

if Nsig(4)>0
   xx=Im;
   for j=1:samples; if abs(xx(j))<0.01; xx(j)=NaN; end; end % replace zeros by NaN
   xx=xx - ones(samples,1)*mean(xx,'omitnan'); % remove bias
   for i=1:length(xx(1,:)) % replace NaN by 1e-9
      for j=1:samples; if isnan(xx(j,i));  xx(j,i)=1e-9; end;  end
   end
   if Nsig(4)==1
       sIm =(abs(2*fft(xx, samples)/samples)')'; % magnitude of Im spectrum; sum of all transmission elements
   else
     sIm = sum(abs(2*fft(xx, samples)/samples)')'; % magnitude of Im spectrum; sum of all transmission elements
   end
end
if Nsig(3)>0
   xx=Vm;
   for j=1:samples; if abs(xx(j))<0.01; xx(j)=NaN; end; end % replace zeros by NaN
   xx=xx - ones(samples,1)*mean(xx,'omitnan');
   for i=1:length(xx(1,:))
      for j=1:samples;  if isnan(xx(j,i));  xx(j,i)=1e-9; end;  end
   end
   if Nsig(3)==1
     sVm =(abs(2*fft(xx, samples)/samples)')'; % magnitude of Vm spectrum; sum of all transmission elements
   else
     sVm = sum(abs(2*fft(xx, samples)/samples)')'; % magnitude of Vm spectrum; sum of all transmission elements
   end
end

%--- Estimate spectrum of oscillations from F
if Nsig(1)>0
   xx=F;
   for j=1:samples; if abs(xx(j))<0.01; xx(j)=NaN; end; end % replace zeros by NaN
   xx=xx - ones(samples,1)*mean(xx,'omitnan'); % remove bias
   for i=1:length(xx(1,:)) % replace NaN by 1e-9
      for j=1:samples; if isnan(xx(j,i));  xx(j,i)=1e-9; end;  end
   end
   if Nsig(1)==1
       sF =(abs(2*fft(xx, samples)/samples)')'; % magnitude of F spectrum; sum of all transmission elements
   else
     sF = sum(abs(2*fft(xx, samples)/samples)')'; % magnitude of F spectrum; sum of all transmission elements
   end
end



%--- Plot: current, voltage, frequency, angles and spectrum
if Nsig(4)>0
  figure (1) % Current magnitude
  plot(Time,Im)
  xlabel('Time, s','FontSize',10)
  ylabel('AMP','FontSize',10)
  title('Current magnitudes','FontSize',11)
end

if Nsig(2)>0
  figure (2) % Voltage magnitude
  plot(Time,Vm)
  xlabel('Time, s','FontSize',10)
  ylabel('KV','FontSize',10)
  title('Voltage magnitudes; phase-to-ground','FontSize',11)
end

if Nsig(1)>0
  figure (3) % Frequency
  plot(Time,F)
  xlabel('Time, s','FontSize',10)
  ylabel('Hz','FontSize',10)
  title('Frequency','FontSize',11)
  ylim([(Fsys-thresholdF) (Fsys+thresholdF)])
end

if Nsig(5)>0
  figure (4) % Current angle
  plot(Time,Ia_relative)
  xlabel('Time, s','FontSize',10)
  ylabel('Degrees','FontSize',10)
  title('Angle of currents; relative to average in measurement frame','FontSize',11)
end

if Nsig(3)>0
  figure (5)  % Voltage angle
  plot(Time,Va_relative)
  xlabel('Time, s','FontSize',10)
  ylabel('Degrees','FontSize',10)
  title('Angle of voltages; relative to average in measurement frame','FontSize',11)
end

if Nsig(1)>0 | Nsig(2)>0 | Nsig(4)>0 
   [~,ind2]=sort(abs(f-F_cutoff),'ascend'); % ind2(1) index of F_cutoff
   if Nsig(4)>0; sIm=sIm/max(sIm(ind2(1):length(f))); end % normolize to have 1 p.u. at F_cutoff
   if Nsig(2)>0; sVm=sVm/max(sVm(ind2(1):length(f))); end
   if Nsig(1)>0; sF=sF/max(sF(ind2(1):length(f))); end

   figure(6)  % Spectra of Im, Vm, F
   le={};
   if Nsig(4)>0
     plot(f(ind2(1):end),sIm(ind2(1):length(f)),'b')
     le=[le 'Current'];
     hold on
   end
   if Nsig(2)>0
      plot(f(ind2(1):end),sVm(ind2(1):length(f)),'r')
      le=[le 'Voltage'];
      hold on
   end
   if Nsig(1)>0
      plot(f(ind2(1):end),sF(ind2(1):length(f)),'k')
      le=[le 'Frequency'];
      hold on
   end
   hold off
   xlabel('Frequency,Hz','FontSize',10)
   ylabel('PSD','FontSize',10)
   h=legend(le);set(h,'Location','NorthEast','fontsize',8);
   title('Spectrum of oscillations from magnitudes','FontSize',11)
end

disp('Competed data analysis. See Report file for details')
toc

fclose all;

%------- Input and Output file names
InputFile="C:\Users\rohit\OneDrive\Desktop\term project power system\ISO-NE-case3\ISO-NE_case3.csv";  % Adjust path as needed
[num,txt] = xlsread(InputFile);
[~,col]=size(txt);
[samples,j]=size(num);
if col>j; num(:,j+1:col)=NaN; end
       
%--- Assign specific lines of title and time vector
LineFull=txt(1, 2:end)';
typeS=txt(2, 2:end)';
dimS=txt(3, 2:end)';
LineShort=txt(4, 2:end)';
t=num(:,1); % time
col=col-1; % the number of PMU data columns

%% Extract PMU data components
for i=1:col/5
    f2(:,i) = num(:,5*i-3);     % Frequency
    Vm(:,i) = num(:,5*i-2);     % Voltage magnitude
    Va(:,i) = num(:,5*i-1);     % Voltage angle
    Im(:,i) = num(:,5*i);       % Current magnitude
    Ia(:,i) = num(:,5*i+1);     % Current angle
end

%% DMD for ISO-NE_Case_3
% Case 3
iGenVol = Vm'/138;              % Normalize voltage magnitude
iGenVolAng = Va';               % Voltage angle
fini = f2';                     % Frequency
r = 70;                         % Rank for DMD

%% Handling NAN Values and data cleaning
TFv = fix(find(isnan(iGenVol)>0)/(r/2))+1;
TFa = fix(find(isnan(iGenVolAng)>0)/(r/2))+1;
j=1;k=1;
for i=1:max(size(iGenVol))
    if i~=TFv(j)
        GenVol(:,k) = iGenVol(:,i);
        GenVolAng(:,k) = iGenVolAng(:,i);
        freq(:,k) = fini(:,i);
        k=k+1;
    else
        if j==max(size(TFv))
            j=j;
        else
            j=j+1;
        end
    end
end
TFv2 = fix(find(isnan(GenVol)>0)/(r/2))+1;

%% Low Pass Filtering
len = max(size(GenVol));
dGenVol = GenVol(:,1:len-1)-GenVol(:,2:len);
dGenAng = GenVolAng(:,1:len-1)-GenVolAng(:,2:len);
Fs = 1/(t(2)-t(1)); % Sampling frequency
num_buses = size(dGenVol, 1); % Number of voltage measurement buses

for i=1:r/2
    [dGenVolfilt(i,:), Voltld] = lowpass(dGenVol(i,:), 3, Fs);
    dGenVolfiltsm(i,:) = smoothdata(dGenVolfilt(i,:),'gaussian',20);
end

%% Prepare data for DMD analysis
dGVfs = dGenVolfiltsm(:, 1110:max(size(dGenVolfiltsm)));
tstep = r*3+1;
len = i+tstep*3;
tc = 1.1;

for i=1:r/2
    dGenVolini(i,:) = dGVfs(i,1)*ones(1,len-1);
end
Y = dGVfs(:, 1:len-1);
Y2 = dGVfs(:, 2:len);

%% Full-time DMD-IS (for Y Mat)
dGenVol = dGVfs(:,tc:len+tc-2)-dGVfs(:,tc+1:len+tc-1);
dGenAng = dGenAng(:,tc:len+tc-2)-dGenAng(:,tc+1:len+tc-1);
for i=1:r/2
    dGenVolini(i,:) = dGenVol(i,1)*ones(1,len-1);
    dGenAngini(i,:) = dGenAng(i,1)*ones(1,len-1);
end

Y = [dGenVol(:, 1:max(size(dGenVol))-1); dGenAng(:, 1:max(size(dGenVol))-1); dGenVolini(:, 1:max(size(dGenVol))-1); dGenAngini(:, 1:max(size(dGenVol))-1)];
Y2 = [dGenVol(:, 2:max(size(dGenVol))); dGenAng(:, 2:max(size(dGenVol)))];

[Uy,Sy,Vy] = svd(Y);

Uy_til = Uy(:,1:2*r);
Sy_til = Sy(1:2*r,1:2*r);
Vy_til = Vy(:, 1:2*r); 

Gy = Y2*Vy_til/Sy_til*ctranspose(Uy_til);

Uy_til1 = ctranspose(Uy_til(1:r,:)');
Uy_til2 = ctranspose(Uy_til(r+1:2*r,:)');

Ay_bar = Y2*Vy_til/Sy_til*ctranspose(Uy_til1);
By_bar = Y2*Vy_til/Sy_til*ctranspose(Uy_til2);

[Uhy,Shy,Vhy] = svd(Y2);
Ay_til = ctranspose(Uhy)*Ay_bar*Uhy;
By_til = ctranspose(Uhy)*By_bar*Uhy;  % Fixed missing multiplication with Uhy

[W1y, SS1y] = eig(Ay_til);
[W2y, SS2y] = eig(By_til);

Phi2 = Ay_bar*Uhy*W1y;
Gam2 = W1y\ctranspose(Uhy)*Uy_til1*Sy_til*Vy_til';  % Calculate mode coefficients

Phi2_mag = abs(Phi2);
Phi2_real = real(Phi2);
Phi2_im = imag(Phi2);
Phi2_ang = atan(Phi2_im./Phi2_real)*180/pi;

Gam2_mag = abs(Gam2);
Gam2_avg = mean(Gam2_mag, 2);
Phi2_dam = -Phi2_real./sqrt(Phi2_real.^2+Phi2_im.^2);
Phi2_std = Phi2_real.*Phi2_im./sqrt(Phi2_real.^2+Phi2_im.^2);
Phi2_stda = Phi2_std(1:r/2,:);
Phi2_stdb = Phi2_std(r/2+1:r,:);
Phi2_stand = Phi2_stda.*Phi2_stdb;

for i=1:r
    Gam2_energy(i,1) = norm(Gam2(i,:));
end
asf2 = log(diag(SS1y));  % Extract diagonal elements
rho2 = real(asf2)/(t(2)-t(1));
f2 = imag(asf2)/(t(2)-t(1))/(2*pi);

[x,A] = maxk(Gam2_energy, 4);
if f2(A(1))>0
    A = A(1);
else
    A = A(2);
end
%% DMD for time segment B
B = 86;  % Time segment B
X = [dGenVol(:, B:max(size(dGenVol))-1); dGenAng(:, B:max(size(dGenVol))-1); dGenVolini(:, B:max(size(dGenVol))-1); dGenAngini(:, B:max(size(dGenVol))-1)];
X2 = [dGenVol(:, B+1:max(size(dGenVol))); dGenAng(:, B+1:max(size(dGenVol)))];

[U,S,V] = svd(X);

U_til = U(:,1:2*r);
S_til = S(1:2*r,1:2*r);
V_til = V(:, 1:2*r);

G = X2*V_til/S_til*ctranspose(U_til);

U_til1 = ctranspose(U_til(1:r,:)');
U_til2 = ctranspose(U_til(r+1:2*r,:)');

A_bar = X2*V_til/S_til*ctranspose(U_til1);
B_bar = X2*V_til/S_til*ctranspose(U_til2);

[Uh,Sh,Vh] = svd(X2);
A_til = ctranspose(Uh)*A_bar*Uh;
B_til = ctranspose(Uh)*B_bar*Uh;

[W1, SS1] = eig(A_til);
[W2, SS2] = eig(B_til);

Phi1 = A_bar*Uh*W1;
Gam1 = W1\ctranspose(Uh)*U_til1*S_til*V_til';

Phi1_mag = abs(Phi1);
Phi1_real = real(Phi1);
Phi1_im = imag(Phi1);
Phi1_ang = atan(Phi1_im./Phi1_real)*180/pi;

Gam1_mag = abs(Gam1);

Phi1_dam = -Phi1_real./sqrt(Phi1_real.^2+Phi1_im.^2);
Phi1_std = Phi1_real.*Phi1_im./sqrt(Phi1_real.^2+Phi1_im.^2);
Phi1_stda = Phi1_std(1:r/2,:);
Phi1_stdb = Phi1_std(r/2+1:r,:);
Phi1_stand = Phi1_stda.*Phi1_stdb;

for i=1:r
    Gam1_energy(i,1) = norm(Gam1(i,:));
end
asf1 = log(diag(SS1));
rho1 = real(asf1)/(t(2)-t(1));
f1 = imag(asf1)/(t(2)-t(1))/(2*pi);

%% DMD for time segment C
C = 190;  % Time segment C
Xl = [dGenVol(:, C:max(size(dGenVol))-1); dGenAng(:, C:max(size(dGenVol))-1); dGenVolini(:, C:max(size(dGenVol))-1); dGenAngini(:, C:max(size(dGenVol))-1)];
Xl2 = [dGenVol(:, C+1:max(size(dGenVol))); dGenAng(:, C+1:max(size(dGenVol)))];

[Ul,Sl,Vl] = svd(Xl);

Ul_til = Ul(:,1:2*r);
Sl_til = Sl(1:2*r,1:2*r);
Vl_til = Vl(:, 1:2*r);

Gl = Xl2*Vl_til/Sl_til*ctranspose(Ul_til);

Ul_til1 = ctranspose(Ul_til(1:r,:)');
Ul_til2 = ctranspose(Ul_til(r+1:2*r,:)');

Al_bar = Xl2*Vl_til/Sl_til*ctranspose(Ul_til1);
Bl_bar = Xl2*Vl_til/Sl_til*ctranspose(Ul_til2);

[Uhl,Shl,Vhl] = svd(Xl2);
Al_til = ctranspose(Uhl)*Al_bar*Uhl;
Bl_til = ctranspose(Uhl)*Bl_bar*Uhl;

[Wl1, SSl1] = eig(Al_til);
[Wl2, SSl2] = eig(Bl_til);

Phil1 = Al_bar*Uhl*Wl1;
Gaml1 = Wl1\ctranspose(Uhl)*Ul_til1*Sl_til*Vl_til';

Phil1_mag = abs(Phil1);
Phil1_real = real(Phil1);
Phil1_im = imag(Phil1);
Phil1_ang = atan(Phil1_im./Phil1_real)*180/pi;

Gaml1_mag = abs(Gaml1);

Phil1_dam = -Phil1_real./sqrt(Phil1_real.^2+Phil1_im.^2);
Phil1_std = Phil1_real.*Phil1_im./sqrt(Phil1_real.^2+Phil1_im.^2);
Phil1_stda = Phil1_std(1:r/2,:);
Phil1_stdb = Phil1_std(r/2+1:r,:);
Phil1_stand = Phil1_stda.*Phil1_stdb;

for i=1:r
    Gaml1_energy(i,1) = norm(Gaml1(i,:));
end
asfl1 = log(diag(SSl1));
rhol1 = real(asfl1)/(t(2)-t(1));
fl1 = imag(asfl1)/(t(2)-t(1))/(2*pi);

%% OCF Values & Source Determination
f2diag = f2;
f1diag = f1;
fl1diag = fl1;

[hphp, hp_ind] = min(abs(f1diag-f2diag(A)));
[lplp, lp_ind] = min(abs(fl1diag-f2diag(A)));

MM_ft = (Phi2_std(:,A));
MM_hp = (Phi1_std(:,hp_ind));
MM_lp = (Phil1_std(:,lp_ind));

OCF_ft = abs(MM_ft(1:r/2).*MM_ft(r/2+1:r));
OCF_hp = (MM_ft(1:r/2)-MM_hp(1:r/2)).*(MM_ft(r/2+1:r)-MM_hp(r/2+1:r));
OCF_lp = (MM_ft(1:r/2)-MM_lp(1:r/2)).*(MM_ft(r/2+1:r)-MM_lp(r/2+1:r));
nOCF_hp = abs(OCF_hp)/max(abs(OCF_hp)).*(OCF_ft>2*mean(OCF_ft));
nOCF_lp = abs(OCF_lp)/max(abs(OCF_lp)).*(OCF_ft>2*mean(OCF_ft));
OCF_tt = nOCF_hp.*nOCF_lp;

OCF_tt2 = abs(OCF_hp)/max(abs(OCF_hp)).*abs(OCF_lp)/max(abs(OCF_lp));

[ttmax, ttmax_ind] = max(OCF_tt2);

fprintf("OCF_tt value: %f\n", OCF_tt2(ttmax_ind));
%fprintf("The oscillation frequency is %f Hz\n", f2(A));

% Define bus names based on the ISO-NE_map
Bus_name = {'Sub2 Ln2', 'Sub2 Ln3', 'Sub2 Ln4', 'Sub5 Ln10', 'Sub5 Ln11', 'Sub5 Ln12', ...
            'Sub7 Ln15', 'Sub7 Ln16', 'Sub7 Gen2', 'Sub10 Ln22', 'Sub10 Ln23', 'Sub10 Ln24', ...
            'Sub11 Ln25', 'Sub11 Ln1-11'};
SOURCE = Bus_name{ttmax_ind};

fprintf("The source bus is located at %s\n", SOURCE);

nOCF_hp = abs(OCF_hp)/max(abs(OCF_hp));
nOCF_lp = abs(OCF_lp)/max(abs(OCF_lp));
OCFs = [nOCF_hp, nOCF_lp, OCF_tt2];


%% Plotting Results
% Plot 1: Voltage Magnitude vs Time for All Buses
figure('Position', [100, 100, 1200, 600]);

% Define time and voltage signals
time = t;
original_signals = dGenVol;
smoothed_signals = dGenVolfiltsm;

% Check for size mismatch and adjust if necessary
num_buses = size(original_signals, 1);
common_length = min(size(original_signals, 2), size(smoothed_signals, 2));
time_plot = time(1:common_length);

% Plot original signals for all buses
hold on;
for bus = 1:num_buses
    plot(time_plot, original_signals(bus, 1:common_length), 'LineWidth', 1.2);
end

% Add labels, title, and legend
xlabel('Time (s)', 'FontSize', 12);
ylabel('Voltage Deviation (p.u.)', 'FontSize', 12);
title('Voltage Magnitude vs Time for All Buses', 'FontSize', 14);
legend(Bus_name, 'Location', 'BestOutside');

% Adjust plot appearance
grid on;
set(gca, 'FontSize', 12);
xlim([min(time_plot) max(time_plot)]);
ylim([min(original_signals(:)) * 1.1, max(original_signals(:)) * 1.1]);
hold off;

% Plot 2: Dominant Frequency and OCF Values
figure('Position', [100, 100, 1200, 800]);


% For plotting the OCF values (part b)
subplot(2,1,2);

% Create a grouped bar chart
bar_data = [nOCF_hp, nOCF_lp, OCF_tt2];
b = bar(bar_data);
set(b(1), 'FaceColor', [0, 0.4470, 0.7410]); % Blue for nOCF_hp
set(b(2), 'FaceColor', [0.8500, 0.3250, 0.0980]); % Orange for nOCF_lp
set(b(3), 'FaceColor', [0.9290, 0.6940, 0.1250]); % Yellow for OCF_tt

% Add labels and legend
set(gca, 'XTickLabel', Bus_name, 'XTickLabelRotation', 45);
ylabel('OCF Values');
legend('nOCF_{hp}', 'nOCF_{lp}', 'OCF_{tt}');
title(' Oscillation Contribution Factors');
grid on;

% Highlight the source bus
hold on;
highlight_idx = ttmax_ind;
plot(highlight_idx, OCF_tt2(highlight_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(highlight_idx, OCF_tt2(highlight_idx) + 0.05, 'Source', 'Color', 'r', 'FontWeight', 'bold');
hold off;

% Plot 3: Phase Voltage for All Buses
figure('Position', [100, 100, 1200, 600]);

hold on;
num_buses = size(Va, 2); % Number of buses
for bus = 1:num_buses
    plot(t, Va(:, bus), 'LineWidth', 1.2);
end

% Add labels, title, and legend
xlabel('Time (s)', 'FontSize', 12);
ylabel('Voltage Angle (degrees)', 'FontSize', 12);
title('Phase Voltage (Voltage Angle) vs Time for All Buses', 'FontSize', 14);
legend(Bus_name, 'Location', 'BestOutside');

% Adjust plot appearance
grid on;
set(gca, 'FontSize', 12);
xlim([min(t), max(t)]);
hold off;