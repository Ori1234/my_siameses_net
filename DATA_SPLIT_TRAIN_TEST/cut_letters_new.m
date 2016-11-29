function cut_letters(csv_file,image_dir,output_dir,margin)
if nargin < 4
	margin=5
end
if nargin < 3
	output_dir='all1/';
end
if nargin < 2
	image_dir='images/';
end
if nargin < 1
	csv_file='WordsCoordinates_processed.csv';
end
fid=fopen(csv_file);
line=fgetl(fid); %advance on table headings
line=fgetl(fid);
counter=1
while ischar(line)
	disp(counter)
	counter=counter+1
	if counter<5534
		continue
	end
	try
	l=strsplit(line,',');
	image_num=l(1);
	word_num=l(2);
	word=l(3);
	rect=l(10);
	a=num2str(double(word{1}))
	word=strrep(a,' ','_')
	real_word=l(3)
	
    
	image_name=strcat(image_dir, image_num, '.jpg');
    current_dir=strcat(output_dir,word);
    if ~exist(current_dir, 'dir')
      mkdir(current_dir);
    end

	segmented_image_path=strcat(current_dir,'/', image_num, '_', word_num, '.jpg')
	segmented_txt_path=strcat(current_dir,'/TAATIK.txt');
	
	a1=imread(image_name{1});
	eval(rect{1});
	w=imcrop(a1, rect);

	
	imwrite(w,segmented_image_path{1});
	
	disp(['writing word ', real_word{1},' to ',segmented_txt_path])
	f_txt=fopen(segmented_txt_path,'w');	
	fprintf(f_txt,'%s\n',real_word{1});
	fclose(f_txt);
	catch
	    disp('An error occurred while retrieving information from the internet.');
	end
	line=fgetl(fid);
end
fclose(fid);

