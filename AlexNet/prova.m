camera = webcam;                           % Connect to camera                           % Load neural net
while true  
    picture = snapshot(camera);            % Take picture
    picture = imresize(picture,[227,227]); % Resize
    label = classify(netTransfer, picture);       % Classify 
    image(picture);                        % Show picture
    title(char(label));                    % Show label
    drawnow;   
end