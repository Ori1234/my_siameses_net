--th set_aside_for_real_test.lua /home/wolf/oriterne/BAVLI/outputs/ /home/wolf/oriterne/BAVLI/outputs/train/ /home/wolf/oriterne/BAVLI/outputs/test/ 0.3
math.randomseed(os.time())
orig_dir=arg[1] or '/home/wolf1/oriterne/DATA/real_data_RGB/all/'
train_dir=arg[2] or '/home/wolf1/oriterne/DATA/real_data_RGB/train/'
test_dir=arg[3] or '/home/wolf1/oriterne/DATA/real_data_RGB/test/'
exclude_precent=arg[4] or 0.5

--TODO check arguments and add slash at end if missing

os.execute("rm -r " .. train_dir)
os.execute("rm -r " .. test_dir)
os.execute("mkdir -p " .. train_dir)
os.execute("cp -r " ..orig_dir..'* '.. train_dir)
os.execute("mkdir -p " .. test_dir)
local total_count=1


function unescape(file_name)
	res=string.gsub(file_name," ","\\ ")
	res=string.gsub(res,'%(','\\(')
	res=string.gsub(res,'%)','\\)')
	return res
end

count=0
function move_dir(source,target)
os.execute("mkdir -p " .. target)
for f in io.popen("ls  "..source):lines() do
	total_count=total_count+1
	if math.random(100)<100*exclude_precent then
		count=count+1
		f1=unescape(f)
		io.popen("mv "..source..'/'..f1.." "..target..'/')
		print('moving '..f1)
	end
end
end

for dir in io.popen("ls "..train_dir):lines() do
	move_dir(train_dir..'/'..dir,test_dir..'/'..dir)
end


--local count=0
--for s,f in pairs(fonts) do
--	if math.random(100)<100*exclude_precent then
--		count=count+1
--		f1=string.gsub(f," ","\\ ")
--		io.popen("mv "..train_dir.."*"..f1.." "..test_dir)
--		print('moving '..f1)
--	end

--end

print(count/total_count)
