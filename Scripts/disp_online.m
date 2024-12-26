function [] = disp_online(str)
fileID=fopen('out.txt','a');
fwrite(fileID,[str newline]);
fclose(fileID);
disp(str)
end