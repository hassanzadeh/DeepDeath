require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

----------------------------------------------------------------------

-- model:
local t = require 'model'
local model = t.model
local criterion = t.criterion

-- Batch test:

local confusion_test,confusion_train

class_names=data.class_names

confusion_train= optim.ConfusionMatrix(class_names)
confusion_valid= optim.ConfusionMatrix(class_names)
confusion_test = optim.ConfusionMatrix(class_names)

local x= torch.Tensor(opt.batchSize , data.train.causes:size(2) , data.train.causes:size(3)) 
local yt=torch.LongTensor(opt.batchSize)

if opt.type == 'cuda' then 
	x=x:cuda()
	yt= yt:cuda()
end

-- test function
function test()
	model:evaluate()

	local model_reduced=model:clone()
	model_reduced:float()
	model_reduced:remove(6)
	model_reduced:evaluate()
	local features=torch.cat(data.test.underlying:reshape(data.test.underlying:nElement() ,1):float(),model_reduced(data.test.causes),2)

    confusion_train:zero()
    confusion_valid:zero()
    confusion_test:zero()

	local time = sys.clock()

	print(sys.COLORS.red .. '\n==> testing on the test set:')
	
	for t = 1,data.test.causes:size(1),opt.batchSize do
      	-- disp progress
		xlua.progress(t, data.test.underlying:size(1))
	  	local len
      	-- batch fits?
		if (t + opt.batchSize - 1) > data.test.causes:size(1) then
			len=data.test.causes:size(1)-t+1 
		else
			len=opt.batchSize
		end
		-- create mini batch
		local idx = 1
		for i = t,t+len-1 do
			x[idx] = data.test.causes[i]
			yt[idx] = data.test.underlying[i]
			idx = idx + 1
		end
		preds_test= model:forward(x:narrow(1,1,len))
		confusion_test:batchAdd(preds_test, yt:narrow(1,1,len))
	end

	confusion_test:updateValids()
	confusion_train:updateValids()

	print('')
	local train_acc=confusion_train.totalValid * 100.0
--	local valid_acc=confusion_valid.totalValid * 100
	local test_acc=confusion_test.totalValid * 100.0

	print ('Mean class accuracy (train set, test set) ' .. train_acc ..  ' '..  test_acc )
	--[[
	print ("-------------")
	print (confusion_train)
	print (confusion_test)
	print ("-------------")
	]]
	logger:write(tostring(confusion_train))
	logger:write("\n")
	logger:write(tostring(confusion_test))
	logger:write("\n---------\n")
	logger:flush()

	return train_acc,test_acc,features

end

-- Export:
return test

