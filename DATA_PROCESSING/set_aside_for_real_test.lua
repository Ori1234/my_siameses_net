math.randomseed(os.time())
train_dir=arg[1]
test_dir=arg[2]
exclude_precent=arg[3] or 0.2

os.execute("mkdir -p " .. test_dir)
local total_count=1
--l='alef'
--fonts={}
--for f in io.popen("ls  "..train_dir..l.."*"):lines() do
--	total_count=total_count+1
--	f1=string.match(f, ".-([^/]-[^%.]+)$")
--	f1=string.sub(f1,5)
--	print(f1)
--	table.insert(fonts,f1)
--end


local count=0
for f in io.popen("ls  "..train_dir):lines() do
	total_count=total_count+1
	if math.random(100)<100*exclude_precent then
		count=count+1
		f1=string.gsub(f," ","\\ ")
		--print(f1)
		f1=string.gsub(f1,'%(','\\(')
		--print(f1)
		f1=string.gsub(f1,'%)','\\)')
		--print(f1)
		io.popen("mv "..train_dir..f1.." "..test_dir)
		print('moving '..f1)
	end
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
