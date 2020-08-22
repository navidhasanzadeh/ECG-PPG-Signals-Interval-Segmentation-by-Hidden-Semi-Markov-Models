function pundo_call(hObject, eventData)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% fig
global state
global points
textObject = findobj(gcf,'Type','text');
scatterObject = findobj(gcf,'Type','scatter');
if(~isempty(scatterObject))
    if(length(textObject)<2)
        delete(textObject);    
    else
        delete(textObject(1));
    end
    if(length(scatterObject)<2)
        delete(scatterObject);    
    else
        delete(scatterObject(1));    
    end
    state = mod(state-1,length(points));
    hObject = findobj(gcf,'Type','uicontrol');
    hObject = hObject(end);
    hObject.String = points{state+1};
    drawnow
end
end


