--1) for each subdir in all:

--1) read files in dir
--2) for each file crop in all and resize
require 'image'
require 'data_augmentation.lua'
data_main_folder=arg[1] or '../../../DATA/real_data_RGB/all'
output_dir=arg[2] or '../../../DATA/real_data_RGB/augmented'

function augment_data(source_dir,output_dir)
	os.execute("mkdir -p " .. output_dir)
	for pic_path in io.popen("ls  "..source_dir.."/*.png"):lines() do
		my_crop(pic_path,output_dir)
	end
end


function my_crop(pic_path,output_dir)
	print(pic_path)
	pic_basename=basename(pic_path)
	im=image.load(pic_path)
	
	size=55
		
	image.save(output_dir..'/'..pic_basename,im)
	for j = 1, #CROP_POS24 do
            im1= zoomout(crop(im, CROP_POS24[j], size))
	    image.save(output_dir..'/'..pic_basename..CROP_POS24[j][1]..CROP_POS24[j][2]..'.png',im1)
            im1= zoomout(crop(im, CROP_POS28[j], size))
	    image.save(output_dir..'/'..pic_basename..CROP_POS28[j][1]..CROP_POS28[j][2]..'_.png',im1)
         end
end

function basename(str)
	local name = string.gsub(str, "(.*/)(.*)", "%2")
	return name
end

---------------------------------
for subfolder in io.popen("ls "..data_main_folder):lines() do
	print(subfolder)
	augment_data(data_main_folder..'/'..subfolder,output_dir..'/'..subfolder)
end


