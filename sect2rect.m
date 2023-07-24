%% Sectorial image to Rectangular image Converter
% Created by Omri Adler, 27.10.2021 - 28.10.2021
% 
% Syntax:
% RectImg = sect2rect(SectImg,rectM,rectN,leftTopMN,leftBotMN,rightTopMN,rightBotMN,sym)
%
% Input:
% SectImg - The sectorial image.
% rectM - Defines the margins on the first index M.
% rectN - Defines the margins on the second index N.
% leftTopMN - Top point indeces [m,n] on the left side.
% leftBotMN - Bottom point indeces [m,n] on the left side.
% rightTopMN - Top point indeces [m,n] on the right side.
% rightBotMN - Bottom point indeces [m,n] on the right side.
% sym - Symmetric angles correction ('on','off').
%
% Output:
% RectImg - The rectangular image.
%
% Notes:
% The image borders should include the top and the bottom of the sector.
% The image borders points should be defined after the margins are removed.
% The degrees are calculated counter-clockwise and 0 is on the right side.
%
% Example ('US_Example.png'):
%{
SectImg = imread('US_Example.png');
RectImg = ...
sect2rect(SectImg,25:200,20:260,[28,102],[95,22],[25,160],[75,220],'off');
%}

function RectImg = sect2rect(SectImg,rectM,rectN,leftTopMN,leftBotMN,rightTopMN,rightBotMN,sym)
%% Crop the relevant ROI (Region Of Interest) from the image:
% Removes the margins, keeps only the inner the rectangle:
ImgDouble = im2double(SectImg);
ImgCut = ImgDouble(rectM,rectN,:);

% Converts the "colored" image into grayscale:
ImgGray = rgb2gray(ImgCut);

% Defines borders for the image sector:
% y1 = a*x1 + b, y2 = a*x2 + b
aL = (leftBotMN(1)-leftTopMN(1))/(leftBotMN(2)-leftTopMN(2));
bL = leftTopMN(1)-leftTopMN(2)*(leftBotMN(1)-leftTopMN(1))/(leftBotMN(2)-leftTopMN(2));
aR = (rightBotMN(1)-rightTopMN(1))/(rightBotMN(2)-rightTopMN(2));
bR = rightTopMN(1)-rightTopMN(2)*(rightBotMN(1)-rightTopMN(1))/(rightBotMN(2)-rightTopMN(2));
% aL*x+bL = aR*x+bR
xTop = (bR-bL)/(aL-aR);
x_mid = (size(ImgGray,2) + 1)/2;
yTop = aL*xTop+bL;


%% Image expansion correction:
if round(yTop) > 1
    dy = 2*round(yTop - 1);
    % Removing the margins, keeping only the inner the rectangle:
    ImgDouble = im2double(SectImg);
    ImgCut = ImgDouble(rectM((1+dy):end),rectN,:);
    % Converts the "colored" image into grayscale:
    ImgGray = rgb2gray(ImgCut);
    % Dots location correction:
    leftBotMN(1) = leftBotMN(1) - dy;
    leftTopMN(1) = leftTopMN(1) - dy;
    rightBotMN(1) = rightBotMN(1) - dy;
    rightTopMN(1) = rightTopMN(1) - dy;
elseif round(yTop) < 1
    dy = 2*round(1 - yTop);
    % Image expansion correction:
    ImgGray = cat(1,zeros(size(ImgGray,2),dy),ImgGray);
    % Dots location correction:
    leftBotMN(1) = leftBotMN(1) + dy;
    leftTopMN(1) = leftTopMN(1) + dy;
    rightBotMN(1) = rightBotMN(1) + dy;
    rightTopMN(1) = rightTopMN(1) + dy;
end
if round(xTop - x_mid) > 0
    % Image expansion correction:
    ImgGray = cat(2,ImgGray,zeros(size(ImgGray,1),2*round(xTop - x_mid)));
elseif round(x_mid - xTop) > 0
    dx = 2*round(x_mid - xTop);
    % Image expansion correction:
    ImgGray = cat(2,zeros(size(ImgGray,1),dx),ImgGray);
    % Dots location correction:
    leftBotMN(2) = leftBotMN(2) + dx;
    leftTopMN(2) = leftTopMN(2) + dx;
    rightBotMN(2) = rightBotMN(2) + dx;
    rightTopMN(2) = rightTopMN(2) + dx;
end

% Updates parameters:
aL = (leftBotMN(1)-leftTopMN(1))/(leftBotMN(2)-leftTopMN(2));
bL = leftTopMN(1)-leftTopMN(2)*(leftBotMN(1)-leftTopMN(1))/(leftBotMN(2)-leftTopMN(2));
aR = (rightBotMN(1)-rightTopMN(1))/(rightBotMN(2)-rightTopMN(2));
bR = rightTopMN(1)-rightTopMN(2)*(rightBotMN(1)-rightTopMN(1))/(rightBotMN(2)-rightTopMN(2));


%% Defines borders for the image sector:
R = size(ImgGray,1);
borders = zeros(size(ImgGray));
for y = 1:size(borders,1)
    for x = 1:size(borders,2)
        if (x <= xTop)&&(y >= aL*x+bL)&&((x-xTop)^2+(y-yTop)^2 <= R^2)
            borders(y,x) = 1;
        elseif (x > xTop)&&(y >= aR*x+bR)&&((x-xTop)^2+(y-yTop)^2 <= R^2)
            borders(y,x) = 1;
        end
    end
end
borders = logical(borders);
dots = zeros(size(ImgGray));
dots(leftTopMN(1),leftTopMN(2)) = 1;
dots(leftBotMN(1),leftBotMN(2)) = 1;
dots(rightTopMN(1),rightTopMN(2)) = 1;
dots(rightBotMN(1),rightBotMN(2)) = 1;

% Plots the borders:
figure
image(cat(3,cat(3,ImgGray,ImgGray),ImgGray));
title('Original Image');
%saveas(gcf,'SectImg.fig');
%saveas(gcf,'SectImg.jpg');
figure
image(cat(3,cat(3,borders,dots),zeros(size(ImgGray))));
title('Sectorial region');
%saveas(gcf,'Borders.fig');
%saveas(gcf,'Borders.jpg');
figure
image(cat(3,cat(3,max(borders,ImgGray),max(dots,ImgGray)),ImgGray));
title('Original Image + Sectorial region');
%saveas(gcf,'SectBorders.fig');
%saveas(gcf,'SectBorders.jpg');

% Can be used later in histogram:
%{
borders_col = borders(:);
[borders_ind,~] = find(borders_col);
%}


%% Creating the squared X matrix:
% Makes a squared image (mid bottom):
ImgSqr = rec2sqr(ImgGray);

% Reshape parameters:
Rmax = floor(size(ImgSqr,1)/2)-1;
% Degree step:
% No spaces (Preserving data): dDeg < 360/(2*pi)*atan(2/Rmax)
eps = 0.01;
dDeg = 360/(2*pi)*atan(2/Rmax) - eps;
if strcmp(sym,'off')
    % Open degree (left side border):
    % alpha = 360/(2*pi)*atan(|dy|/|dx|) + 180
    openDeg = 360/(2*pi)*atan(abs(leftBotMN(1)-leftTopMN(1))/abs(leftBotMN(2)-leftTopMN(2))) + 180;
    % Close degree (right side border):
    % beta = 360 - 360/(2*pi)*atan(|dy|/|dx|)
    closeDeg = 360 - 360/(2*pi)*atan(abs(rightBotMN(1)-rightTopMN(1))/abs(rightBotMN(2)-rightTopMN(2)));
elseif strcmp(sym,'on')
    % Symmetric case:
    alpha = 360/(2*pi)*min(atan(abs(leftBotMN(1)-leftTopMN(1))/abs(leftBotMN(2)-leftTopMN(2))),...
        atan(abs(rightBotMN(1)-rightTopMN(1))/abs(rightBotMN(2)-rightTopMN(2))));
    % Open degree (left side border):
    % alpha = 360/(2*pi)*atan(|dy|/|dx|) + 180
    openDeg = alpha + 180;
    % Close degree (right side border):
    % beta = 360 - 360/(2*pi)*atan(|dy|/|dx|)
    closeDeg = 360 - alpha;
else
    error('''sym'' is not defined properly');
end


%% The transformation (into rectangular shape):
RectImg = shapeTrans(ImgSqr,openDeg*(2*pi/360),dDeg*(2*pi/360),closeDeg*(2*pi/360),0,1,Rmax);

% Plots the rectangular image:
figure
imagesc(RectImg);
colormap gray;
title('Rectangular Image');
%saveas(gcf,'RectImg.fig');
%saveas(gcf,'RectImg.jpg');


%% The inverse transformation (into sectorial shape):
ImgSqr2 = IshapeTrans(ImgSqr,RectImg,openDeg*(2*pi/360),dDeg*(2*pi/360),closeDeg*(2*pi/360),0,1,Rmax);


%% Creates the un-squared matrix:
SectImg2 = sqr2rec(ImgSqr2,ImgGray);

% Plots the sectorial image:
figure
imagesc(SectImg2);
colormap gray;
title('Sectorial Image Reconstruction');
%saveas(gcf,'SectImg2.fig');
%saveas(gcf,'SectImg2.jpg');
end


%% Converting a rectangular matrix into a squared matrix:
% Prepares for the Polar R-Theta view conversion.
function Xsqr = rec2sqr(Xrec)
Xsqr = zeros(2*max(size(Xrec))+1);
n_mid = (size(Xsqr,2) + 1)/2;
m_mid = (size(Xsqr,1) + 1)/2;
m1 = ceil(m_mid);
m2 = ceil(m_mid) + size(Xrec,1) - 1;
n1 = ceil(n_mid - size(Xrec,2)/2);
n2 = ceil(n_mid + size(Xrec,2)/2 - 1);
Xsqr(m1:m2,n1:n2) = Xrec;
end


%% Converting a squared matrix into a rectangular matrix:
% Prepares for the Cartesian X-Y view conversion.
function XrecNew = sqr2rec(Xsqr,XrecExample)
n_mid = (size(Xsqr,2) + 1)/2;
m_mid = (size(Xsqr,1) + 1)/2;
m1 = ceil(m_mid);
m2 = ceil(m_mid) + size(XrecExample,1) - 1;
n1 = ceil(n_mid - size(XrecExample,2)/2);
n2 = ceil(n_mid + size(XrecExample,2)/2 - 1);
XrecNew = Xsqr(m1:m2,n1:n2);
end


%% 2D Polar Shape Transform (for a specific section)
%
% Syntax: mat_out = shapeTrans(mat_in,thetaStart,thetaD,thetaEnd,rStart,rD,rEnd)
%
% Input:
% mat_in - The original matrix.
% thetaStart - The initial angle [radians].
% thetaD - The angles step [radians].
% thetaEnd - The final angle [radians].
% rStart - The initial radius [pixels]. (natural numbers)
% rD - The radii step [pixels]. (natural numbers)
% rEnd - The final radius [pixels]. (natural numbers)
%
% Output:
% mat_out - Shape transform of "mat_in".
%
%
% Example parameters:
% Theta limits:
% thetaStart = 0; % [radians]
% thetaD = 5*(2*pi/360); % [radians]
% thetaEnd = 135*(2*pi/360); % [radians]
% Radius limits:
% rStart = 0; % [pixels]
% rD = 1; % [pixels]
% rEnd = 5; % [pixels]


function mat_out = shapeTrans(mat_in,thetaStart,thetaD,thetaEnd,rStart,rD,rEnd)

% Creating an FT matrix:
Mvec = rStart:rD:rEnd;
Mout = length(Mvec);
Nvec = thetaStart:thetaD:thetaEnd;
Nout = length(Nvec);
mat_out = zeros(Mout,Nout);
% Define coordinates:
midXin = (size(mat_in,2) + 1)/2;
midYin = (size(mat_in,1) + 1)/2;

% Initialize index:
n = 1;
for theta = thetaStart:thetaD:thetaEnd
    
    % Initialize index:
    m = 1;
    
    for r = rStart:rD:rEnd
        
        %% Changing coordinates:
        Xin = midXin + r*cos(theta);
        Yin = midYin - r*sin(theta);
        
        %% Conditions reading version:
        % None in position:
        if (ceil(Xin) ~= floor(Xin))&&(ceil(Yin) ~= floor(Yin))
            pix = ( 1 - (Xin - floor(Xin)) ) * ( 1 - (Yin - floor(Yin)) ) * mat_in(floor(Yin),floor(Xin)) + ...
                ( 1 - (Xin - floor(Xin)) ) * ( 1 - (ceil(Yin) - Yin) ) * mat_in(ceil(Yin),floor(Xin)) + ...
                ( 1 - (ceil(Xin) - Xin) ) * ( 1 - (Yin - floor(Yin)) ) * mat_in(floor(Yin),ceil(Xin)) + ...
                ( 1 - (ceil(Xin) - Xin) ) * ( 1 - (ceil(Yin) - Yin) ) * mat_in(ceil(Yin),ceil(Xin));
            
            % Only X is in position ->
        elseif (ceil(Xin) == floor(Xin))&&(ceil(Yin) ~= floor(Yin))
            pix = ( 1 - (Yin - floor(Yin)) ) * mat_in(floor(Yin),round(Xin)) + ...
                ( 1 - (ceil(Yin) - Yin) ) * mat_in(ceil(Yin),round(Xin));
            
            % Only Y is in position ->
        elseif (ceil(Xin) ~= floor(Xin))&&(ceil(Yin) == floor(Yin))
            pix = ( 1 - (Xin - floor(Xin)) ) * mat_in(round(Yin),floor(Xin)) + ...
                ( 1 - (ceil(Xin) - Xin) ) * mat_in(round(Yin),ceil(Xin));
            
            % Both in position.
        else
            pix = mat_in(round(Yin),round(Xin));
        end
        
        % Pixel summing:
        mat_out(m,n) = pix;
        
        % Index updates:
        m = m + 1;
    end
    
    % Index updates:
    n = n + 1;
end

end


%% 2D Inverse Polar Shape Transform (for a specific section)
%
% Syntax: mat_out = IshapeTrans(mat_in,mat_mid,thetaStart,thetaD,thetaEnd,rStart,rD,rEnd)
%
% Input:
% mat_in - The original matrix.
% mat_mid - The transformed matrix.
% thetaStart - The initial angle [radians].
% thetaD - The angles step [radians].
% thetaEnd - The final angle [radians].
% rStart - The initial radius [pixels]. (natural numbers)
% rD - The radii step [pixels]. (natural numbers)
% rEnd - The final radius [pixels]. (natural numbers)
%
% Output:
% mat_out - Shape transform of "mat_mid".
%
%
% Example parameters:
% Theta limits:
% thetaStart = 0; % [radians]
% thetaD = 5*(2*pi/360); % [radians]
% thetaEnd = 135*(2*pi/360); % [radians]
% Radius limits:
% rStart = 0; % [pixels]
% rD = 1; % [pixels]
% rEnd = 5; % [pixels]


function mat_out = IshapeTrans(mat_in,mat_mid,thetaStart,thetaD,thetaEnd,rStart,rD,rEnd)

% Creating an FT matrix:
mat_out = zeros(size(mat_in));
fix_mat = zeros(size(mat_in));
% Define coordinates:
midXin = (size(mat_in,2) + 1)/2;
midYin = (size(mat_in,1) + 1)/2;

% Initialize index:
n = 1;
for theta = thetaStart:thetaD:thetaEnd
    
    % Initialize index:
    m = 1;
    
    for r = rStart:rD:rEnd
        
        pix = mat_mid(m,n);
        
        %% Changing coordinates:
        Xout = midXin + r*cos(theta);
        Yout = midYin - r*sin(theta);
        
        %% Conditions writing version:
        % None in position:
        if (ceil(Xout) ~= floor(Xout))&&(ceil(Yout) ~= floor(Yout))
            mat_out(floor(Yout),floor(Xout)) = mat_out(floor(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * ( 1 - (Yout - floor(Yout)) ) * pix;
            mat_out(ceil(Yout),floor(Xout)) = mat_out(ceil(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * ( 1 - (ceil(Yout) - Yout) ) * pix;
            mat_out(floor(Yout),ceil(Xout)) = mat_out(floor(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * ( 1 - (Yout - floor(Yout)) ) * pix;
            mat_out(ceil(Yout),ceil(Xout)) = mat_out(ceil(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * ( 1 - (ceil(Yout) - Yout) ) * pix;
            
            % Only X is in position ->
        elseif (ceil(Xout) == floor(Xout))&&(ceil(Yout) ~= floor(Yout))
            mat_out(floor(Yout),round(Xout)) = mat_out(floor(Yout),round(Xout)) + ...
                ( 1 - (Yout - floor(Yout)) ) * pix;
            mat_out(ceil(Yout),round(Xout)) = mat_out(ceil(Yout),round(Xout)) + ...
                ( 1 - (ceil(Yout) - Yout) ) * pix;
            
            % Only Y is in position ->
        elseif (ceil(Xout) ~= floor(Xout))&&(ceil(Yout) == floor(Yout))
            mat_out(round(Yout),floor(Xout)) = mat_out(round(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * pix;
            mat_out(round(Yout),ceil(Xout)) = mat_out(round(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * pix;
            
            % Both in position.
        else
            mat_out(round(Yout),round(Xout)) = mat_out(round(Yout),round(Xout)) + ...
                pix;
        end
        
        
        %% Conditions fixing version:
        % None in position:
        if (ceil(Xout) ~= floor(Xout))&&(ceil(Yout) ~= floor(Yout))
            fix_mat(floor(Yout),floor(Xout)) = fix_mat(floor(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * ( 1 - (Yout - floor(Yout)) ) * 1;
            fix_mat(ceil(Yout),floor(Xout)) = fix_mat(ceil(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * ( 1 - (ceil(Yout) - Yout) ) * 1;
            fix_mat(floor(Yout),ceil(Xout)) = fix_mat(floor(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * ( 1 - (Yout - floor(Yout)) ) * 1;
            fix_mat(ceil(Yout),ceil(Xout)) = fix_mat(ceil(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * ( 1 - (ceil(Yout) - Yout) ) * 1;
            
            % Only X is in position ->
        elseif (ceil(Xout) == floor(Xout))&&(ceil(Yout) ~= floor(Yout))
            fix_mat(floor(Yout),round(Xout)) = fix_mat(floor(Yout),round(Xout)) + ...
                ( 1 - (Yout - floor(Yout)) ) * 1;
            fix_mat(ceil(Yout),round(Xout)) = fix_mat(ceil(Yout),round(Xout)) + ...
                ( 1 - (ceil(Yout) - Yout) ) * 1;
            
            % Only Y is in position ->
        elseif (ceil(Xout) ~= floor(Xout))&&(ceil(Yout) == floor(Yout))
            fix_mat(round(Yout),floor(Xout)) = fix_mat(round(Yout),floor(Xout)) + ...
                ( 1 - (Xout - floor(Xout)) ) * 1;
            fix_mat(round(Yout),ceil(Xout)) = fix_mat(round(Yout),ceil(Xout)) + ...
                ( 1 - (ceil(Xout) - Xout) ) * 1;
            
            % Both in position.
        else
            fix_mat(round(Yout),round(Xout)) = fix_mat(round(Yout),round(Xout)) + ...
                1;
        end
        
        % Index updates:
        m = m + 1;
    end
    
    % Index updates:
    n = n + 1;
end

%% Fixing method (after writing):
fix_mat(fix_mat == 0) = 1;
fix_mat = 1./fix_mat;
mat_out = fix_mat.*mat_out;

end
