function cut_letters(csv_file,image_dir,output_dir,margin)
if nargin < 4
	margin=5
end
if nargin < 3
	output_dir='all2/';
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
	f_txt=fopen(segmented_txt_path,'w','n','ISO-8859-1');	
	fprintf(f_txt,'%s',real_word{1});
	fclose(f_txt);
	catch
	    disp('An error occurred while retrieving information from the internet.');
	end
	line=fgetl(fid);
end
fclose(fid);


%{
this work
rdsCoordinates_processed1.csv';
  4 fid=fopen(csv_file);
  5 line=fgetl(fid); %advance on table headings
  6 line=fgetl(fid);
  7 counter=1
  8 
  9 
 10 
 11 while ischar(line)
 12         counter=counter+1;
 13         try
 14         l=strsplit(line,',');
 15         image_num=l(1);
 16         word_num=l(2);
 17         word=l(3);
 18         rect=l(10);
 19         a=num2str(double(word{1}));
 20         word=strrep(a,' ','_');
 21         real_word=l(3);
 22 
 23 
 24 
 25         segmented_txt_path=strcat(output_dir,num2str(counter),'_e');
 26 
 27 
 28         disp(['writing word ', real_word{1},' to ',segmented_txt_path])
 29 
 30 
 31 %       f_txt=fopen(segmented_txt_path,'w','n','windows-1255');  %I think now this is good with the edit distance
 32 %       fprintf(f_txt,'%s\n',real_word{1});
 33 %       fclose(f_txt);
 34 
 35 
 36         f_txt1=fopen(strcat(segmented_txt_path,'1'),'w');
 37         fprintf(f_txt1,'%s\n',real_word{1});
 38         fclose(f_txt1);
 39 
 40 
 41         line=fgetl(fid);
 42 end
 43 %fclose(fid);
 44 end
  %}           
