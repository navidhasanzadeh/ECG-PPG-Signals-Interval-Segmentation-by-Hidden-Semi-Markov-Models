% close all,clear all
% By Navid Hasanzadeh 810197115
% By Soroush Dehghan 810197139
warning off
partNum = 1;
if (~exist('Part_1','var'))
    load(['data\Part_',num2str(partNum),'.mat'])
end
global points
points = {'P_{on}';'P_{off}';'Q';'J';'T_{off}'};
labels = {'B','P','B','R','T'};
freq = 125;
freq_out = 500;
duration = 2;
mkdir('out');
for sample=1:20:size(Part_1,2)
    sampleData = cell2mat(Part_1(sample));
    if(length(sampleData)>12*freq*duration)
        for seg=1:1:3
            PPG = sampleData(1,1+(seg-1)*freq * duration:seg*freq * duration);
            ECG = sampleData(3,1+(seg-1)*freq * duration:seg*freq * duration);
            ECG = interp(ECG, freq_out/freq);
            ECG = ECG-mean(ECG);
            ECG = ECG./std(ECG);
            x = 0:1/freq_out:duration;
            x(end)=[];
            y = ECG;
            f=figure(1);            
            set(gcf, 'Units', 'inches', 'Position', [0 0 14 0.4*14]);
            movegui(gcf,'center')
            global state
            state = 0;
            ButtonH=uicontrol('Parent',f,'Style','pushbutton','String',points(state+1),'Units','normalized','Position',[0.01 0.03 0.07 0.05],'Visible','on','Callback',@pb_call);
            ButtonU=uicontrol('Parent',f,'Style','pushbutton','String','Undo(U)','Units','normalized','Position',[0.01 0.13 0.07 0.05],'Visible','on','Callback',@pundo_call);
            set(f,'WindowButtonDownFcn',@OnMouse)
            plot(x,y,'LineWidth',2)
            xlabel('Keyboard Shortcuts:: U=Undo    N=Change     G=Ignore   Enter: Save&Next   Esc=Exit')
            suptitle(['ECG - Part ',num2str(partNum),' - ',num2str(sample),' - ', num2str(seg)]);
            grid on
            while 1
                k=0;
                while 1
                    while (k~=1)
                        k = waitforbuttonpress;
                    end
                    value = double(get(gcf,'CurrentCharacter'));
                    if ~isempty(value)
                        break;
                    end
                    k=0;
                    display('Keyboard is not English')
                end
                sc = findobj(gca,'Type','scatter');
                ignoreFlag = 0;
                if value==13 %enter
                    xCoord = cell2mat(get(sc, 'XData'));
                    yCoord = cell2mat(get(sc, 'YData'));
                    te = findobj(gca,'Type','text');
                    selectedlabels = {};
                    for i=length(te):-1:1
                        selectedlabels{end+1} = te(i).String{1};
                    end
                    close(f)
                    break;
                elseif(value==27)
                    return;
                elseif(value==85 || value==117)
                    pundo_call();                    
                elseif(value==71 || value==103)
                    ignoreFlag = 1;
                    close(f)
                    break;
                elseif(value==78 || value==110)
                    pb_call();
                end
            end
            if(ignoreFlag~=1)
                try
                    x=x';
                    A = repmat(x,[1 length(xCoord)]);
                    [minValue,closestIndex] = min(abs(A-xCoord'));
                    closestValue = x(closestIndex) ;
                    % figure
                    % plot(x,y)
                    % hold on
                    % scatter(x(closestIndex),y(closestIndex))
                    closestIndex = sort(closestIndex);
                    signalLabel = x';
                    %labeling
                    startLabel = selectedlabels{1};
                    startLabelIndex = find(contains(points,startLabel));
                    endLabel = selectedlabels{end};
                    endLabelIndex = find(contains(points,endLabel));
                    signalLabel(1:closestIndex(1))=labels{startLabelIndex};
                    for i=2:1:length(selectedlabels)
                        signalLabel(closestIndex(i-1):closestIndex(i)) = labels{1+mod(startLabelIndex+i-2,length(labels))};
                    end
                    signalLabel(closestIndex(end):end)= labels{1+mod( endLabelIndex,length(labels))};
                    signalLabel = char(signalLabel)
                    dlmwrite(['out\ECG-',num2str(partNum),'.csv'],ECG,'delimiter',',','-append');
                    dlmwrite(['out\ECGLabel-',num2str(partNum),'.csv'],signalLabel,'delimiter',',','-append');
                    dlmwrite(['out\id-',num2str(partNum),'.csv'],[partNum,sample,seg],'delimiter',',','-append');
                    display('Saved.')
                catch
                    display('Error:(');
                end
            end
            ignoreFlag = 0;
        end
    end
end

