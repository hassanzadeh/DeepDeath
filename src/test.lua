require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

----------------------------------------------------------------------

-- model:
local t = require 'model'
local model = t.model
local criterion = t.criterion

-- Batch test:

local x_train,x_test,x_valid
local targets_train ,targets_test ,targets_valid
local confusion_test,confusion_train

classes=data.classes

confusion_train= optim.ConfusionMatrix(classes)
confusion_valid= optim.ConfusionMatrix(classes)
confusion_test = optim.ConfusionMatrix(classes)

class_train =data.train.underlying:clone()
class_test =data.test.underlying:clone()
class_valid =data.valid.underlying:clone()

x_train= torch.Tensor(data.train.causes:size(1) , data.train.causes:size(2) , data.train.causes:size(3)) 
x_test= torch.Tensor(data.test.causes:size(1),data.test.causes:size(2) , data.test.causes:size(3)) 
x_val= torch.Tensor(data.valid.causes:size(1) ,data.valid.causes:size(2) , data.valid.causes:size(3)) 

if opt.type == 'cuda' then 
	x_train=x_train:cuda()
	calss_train = class_train:cuda()

	x_valid=x_valid:cuda()
	class_valid= class_valid:cuda()

	x_test=x_test:cuda()
	class_test= class_test:cuda()
end


-- test function
function test()
	model:evaluate()

    confusion_train:zero()
    confusion_valid:zero()
    confusion_test:zero()

	local time = sys.clock()

		-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	local preds_train = model:forward(x_train):clone()
	local preds_valid = model:forward(x_val):clone()
	local preds_test = model:forward(x_test):clone()
	
	confusion_train:batchAdd(preds_train, class_train)
	confusion_valid:batchAdd(preds_valid, class_valid)
	confusion_test:batchAdd(preds_test, class_test)

	confusion_test:updateValids()
	confusion_train:updateValids()

	local train_acc=confusion_train.totalValid * 100
	local valid_acc=confusion_valid.totalValid * 100
	local test_acc=confusion_test.totalValid * 100

	print ('Mean class accuracy (train set, validation set,test set) ' .. train_acc .. ' '.. valid_acc.. ' '..  test_acc )
	--[[print ("-------------")
	print (confusion_train)
	print (confusion_test)
	print ("-------------")]]

	return train_acc,valid_acc,test_acc

end

-- Export:
return test

