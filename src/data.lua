require 'torch'	-- torch
require 'nnx'		-- provides a normalization operator

-- see if the file exists
local function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

local output
if (opt.load) then
	local file=(opt.data_dir.. '/'.. opt.input_file ..'.tch')
	if (not file_exists(file) ) then
		print ('File not found: ' .. file )
		os.exit(1) 
	end

	output = torch.load(opt.data_dir.. '/'.. opt.input_file ..'.tch')

	print ('Train/Test/Validation sets loaded from file')
else
	local function table_len(tab)
		count=0
		for _ in pairs (tab) do
			count= count +1
		end
		return count
	end

	local icd10_etiology2indx={}
	for k,v in ipairs {' ','0','1','2','3','4','5','6','7','8','9'} do 
		icd10_etiology2indx[v]=k 
	end


	-- get all lines from a file, returns an empty 
	-- list/table if the file does not exist
	local function read_mortality(file)
		print (sys.COLORS.red .. '==> Reading mortality file: ' .. file);
		if not file_exists(file) then
			print ('File not found: ' .. file )
			os.exit(1) 
		end
		local incidence_count= {}
		local icd10_group2indx={}
		for line in io.lines(file) do 
			local underlying_cause_113 = tonumber(string.sub(line,154,156))
			if (underlying_cause_113>135) then
				print ('Wrong 113-recode underlying cause! ' .. underlying_cause_113)
				os.exit(1)
			elseif (underlying_cause_113<112) then	
				local num_ent_axis_conditions=tonumber(string.sub(line,163,164))
				for i=0,num_ent_axis_conditions-1 do
					local ind=165+i*7
					local code=string.sub(line,ind+2,ind+5)
					if (icd10_group2indx[string.sub(code,1,1)] == nil) then
						icd10_group2indx[string.sub(code,1,1)]=table_len(icd10_group2indx)+1
					end	
				end
				if (incidence_count[underlying_cause_113] == nil) then
					incidence_count[underlying_cause_113]=1
				else	
					incidence_count[underlying_cause_113]=incidence_count[underlying_cause_113]+1
				end
				
			end
		end
		local size=0
		for k,v in pairs(incidence_count) do
			if (v > 999) then
				size = size + v
			end
		end
		local num_group=table_len(icd10_group2indx)
		local num_etio=table_len(icd10_etiology2indx)

		local data=torch.Tensor(size,20,100+num_group+num_etio):fill(-1)
		local underlying=torch.LongTensor(size)
		local max_ent =0
		local ind=0
		for line in io.lines(file) do 
		--[[local residential_status=tonumber(string.sub(line,20,20))
			local sex=string.sub(line,69,69)
			local age_days=0
			if (string.sub(line,70,70) == '1') then
				age_days=tonumber(string.sub(line,71,73))*365
			elseif (string.sub(line,70,70) == '2') then
				age_days=tonumber(string.sub(line,71,73))*30.41
			elseif (string.sub(line,70,70) == '4') then
				age_days=tonumber(string.sub(line,71,73))
			elseif (string.sub(line,70,70) == '5') then
				age_days=tonumber(string.sub(line,71,73))/24.0
			elseif (string.sub(line,70,70) == '6') then
				age_days=tonumber(string.sub(line,71,73))/(24.0*60)
			elseif (string.sub(line,70,70) == '9') then
					age_days=-1
			else
				print ('Wrong age! ' .. torch.round(age_days/36.5)/10)
				os.exit(1)
			end
			local marital=string.sub(line,84,84)
			local manner_of_death=string.sub(line,107,107)]]
			local underlying_cause_113 = tonumber(string.sub(line,154,156))
			local num_ent_axis_conditions=tonumber(string.sub(line,163,164))

			if (num_ent_axis_conditions > max_ent) then 
				max_ent=num_ent_axis_conditions 
			end
			if (num_ent_axis_conditions>20) then
				print ('Wrong num_ent_axis_conditions ! ' .. num_ent_axis_conditions)
				os.exit(1)
			end
			local entity_axis_conds={}
			--print (num_ent_axis_conditions)
			for i=0,num_ent_axis_conditions-1 do
				local ind=165+i*7
				local code=string.sub(line,ind+2,ind+5)
				entity_axis_conds[#entity_axis_conds+1]=code
			end
			if (underlying_cause_113<112 and incidence_count[underlying_cause_113]>999) then -- certified by physician
				ind=ind+1
				for i=1,num_ent_axis_conditions do
					data[ind][20-i+1][icd10_group2indx[string.sub(entity_axis_conds[i],1,1)]]=1
					data[ind][20-i+1][num_group+tonumber(string.sub(entity_axis_conds[i],2,3))+1]=1
					data[ind][20-i+1][num_group+100+icd10_etiology2indx[string.sub(entity_axis_conds[i],4,4)]]=1
				end
				underlying[ind]=underlying_cause_113
			elseif( underlying_cause_113<1) then
				print ('Wrong underlying cause! ' .. underlying_cause_113)
				os.exit(1)
			end

--[[			local num_rec_axis_conditions=tonumber(string.sub(line,341,342))
			if (num_rec_axis_conditions>20) then
				print ('Wrong num_rec_axis_conditions ! ' .. num_rec_axis_conditions)
				os.exit(1)
			end
			local rec_axis_conds={}
			for i=0,num_rec_axis_conditions-1 do
				local ind=344+i*5
				rec_axis_conds[#rec_axis_conds+1]=string.sub(line,ind,ind+3)
			end
]]

  		end
		if (data:size(1) ~= ind) then
			print ('Mismatch in data size and number of entries! ' .. data:size(1) .. ' '.. ind)
			os.exit(1)
		end
		print ('File parsed')
	  	return data,underlying,incidence_count
	end


	local data,underlying,inc_count=read_mortality(opt.data_dir .. opt.input_file)

	local classes={}
	local class_names={}
	local count =0 
	for k,v in pairs(inc_count) do  --excluding  non-frequent natural deaths
		if (v>=1000) then
			count = count +1
			classes[k]=count
			class_names[#class_names+1]=k
		end
	end
	--print (class_names)
	local trsize = torch.round(data:size(1)*.7) 
	local vasize= torch.round(data:size(1) * 0.01)
	local tesize= data:size(1)-trsize -vasize

	local train =torch.Tensor(trsize,data:size(2),data:size(3)):fill(-1)
	local valid =torch.Tensor(vasize,data:size(2),data:size(3)):fill(-1)
	local test = torch.Tensor(tesize,data:size(2),data:size(3)):fill(-1)

	local train_class =torch.LongTensor(trsize):fill(0)
	local valid_class =torch.LongTensor(vasize):fill(0)
	local test_class = torch.LongTensor(tesize):fill(0)

	print ('Tensors created')

	local shuffle = torch.randperm(trsize+tesize+vasize)

	for i=1,trsize do
		local ind=shuffle[i]
		train[i]=data[ind]
		train_class[i]=classes[underlying[ind]]
	end
	print ('Train tensor created')
	for i=1,vasize do
		local ind=shuffle[i+trsize]
		valid[i]=data[ind]
		valid_class[i]=classes[underlying[ind]]
	end

	print ('Validation tensor created')

	for i=1,tesize do
		local ind=shuffle[i+trsize+vasize]
		test[i]=data[ind]
		test_class[i]=classes[underlying[ind]]
	end
	print ('Test tensor created')

	data={}
	collectgarbage()

	-- create train set:
	local trainData = {
		causes = train,
		underlying=train_class
	}

	-- create validation set:
	local validData = {
		causes = valid,
		underlying= valid_class,
	}

	--create test set:
	local testData = {
		causes= test,
		underlying=test_class,
	}

	print ('Train size: ' .. trsize)
	print ('Validation size: '.. vasize)
	print ('Test size: '.. tesize)
	print ('Size total: '..trsize+tesize+vasize)

	output={
		train= trainData,
		valid=validData,
		test= testData,
		class_names=class_names
	}
	torch.save(opt.data_dir..'/'..opt.input_file..'.tch',output)
	print ('Train/Test/Validation sets serialized')
end
-- Exports
return  output
