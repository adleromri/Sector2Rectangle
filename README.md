# Sector2Rectangle
A converter between sectorial view (such as in ultrasound) into a rectangular view (such as in normal video).

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
