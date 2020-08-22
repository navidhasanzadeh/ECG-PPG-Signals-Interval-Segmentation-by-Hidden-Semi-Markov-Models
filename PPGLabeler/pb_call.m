function pb_call(hObject, eventData)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% fig
global state
global points
state = mod(state+1,length(points));
hObject2 = findobj(gcf,'Type','uicontrol');
hObject2 = hObject2(end);
hObject2.String = points{state+1};
drawnow
end

