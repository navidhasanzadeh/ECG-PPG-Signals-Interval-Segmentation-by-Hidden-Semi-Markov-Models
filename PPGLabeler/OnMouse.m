function [clicked_x,clicked_y]=OnMouse(hObject,~)
global state
global points
axes_handle = gca;
pt = get(axes_handle, 'CurrentPoint');
clicked_x=pt(1,1);
clicked_y=pt(1,2);
hold on
sc = findobj(gca,'Type','scatter');
xCoord = get(sc, 'XData');
% if(length(xCoord)<5)
    scatter(clicked_x,clicked_y,60,'filled');
    if mod(state,2)==0
        ypos =  clicked_y-0.25;
    else
        ypos = clicked_y + 0.25;
    end    
    text(clicked_x, ypos,  [points(state+1)],'Color','red','FontSize',10,'FontWeight','bold');
    state = mod(state+1,length(points));
    hObject = findobj(gcf,'Type','uicontrol');
    if(length(hObject)>1)
        hObject = hObject(end);
    end
    hObject.String = points{state+1};
drawnow
% else
%     disp('All points are selected. Plaese Enter to continue or ESC to reselect the points.');
% end
end