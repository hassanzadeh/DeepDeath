require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'image'

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
--local fwmodel = t.model
local criterion = t.criterion
local model_name = t.model_name



-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()


local optimState = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	weightDecay = opt.weightDecay,
	learningRateDecay = opt.learningRateDecay
}


local x = torch.Tensor(opt.batchSize,data.train.cause[1]:size(1),data.train.cause[1]:size(2))
local yt = torch.Tensor(opt.batchSize,data.train.underlying[1]:size(1))
if opt.type == 'cuda' then 
	x = x:cuda()
	yt = yt:cuda()
end


local epoch
local function train()
	model:training()

   -- epoch tracker
	epoch = epoch or 1

   -- local vars
	local time = sys.clock()
	local run_passed = 0
	local mean_dfdx = torch.Tensor():typeAs(w):resizeAs(w):zero()

   -- shuffle at each epoch
	local shuffle = torch.randperm(data.train.cause:size(1))
	local err = 0

	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,data.train.cause:size(1),opt.batchSize do
      -- disp progress
		--xlua.progress(t, data.train.int:size(1))
		collectgarbage()

      -- batch fits?
		if (t + opt.batchSize - 1) > data.train.cause:size(1) then
			break
		end

      -- create mini batch
		local idx = 1
		for i = t,t+opt.batchSize-1 do
			x[idx] = data.train.cause[shuffle[i]]
			yt[idx] = data.train.underlying[shuffle[i]]
			idx = idx + 1
		end

         -- create closure to evaluate f(X) and df/dX
		local eval_E = function(w)
            -- reset gradients
			dE_dw:zero()
	
            -- evaluate function for complete mini batch
			local y = model:forward(x)
			local E = criterion:forward(y,yt)
	
            -- estimate df/dW
			local dE_dy = criterion:backward(y,yt)   
			model:backward(x,dE_dy)


			--dE_dw:clamp(-opt.grad_clip, opt.grad_clip)
            -- return f and df/dX
			return E,dE_dw
		end

         -- optimize on current mini-batch
		if opt.optMethod == 'sgd' then
			optim.sgd(eval_E, w, optimState)
		elseif opt.optMethod == 'asgd' then
			run_passed = run_passed + 1
			mean_dfdx  = asgd(eval_E, w, run_passed, mean_dfdx, optimState)
		elseif opt.optMethod == 'rmsprop' then
			optim.rmsprop(eval_E, w, optimState)
		end                                


	end

   -- time taken
	time = sys.clock() - time
	time = time / data.train.int:size(1)
--print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- next epoch
	epoch = epoch + 1
end

-- Export:
return train
