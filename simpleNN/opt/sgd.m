function model = sgd(prob, param, model, net)

lr = param.lr;
batch_size = param.bsize;
decay = param.decay;

v = cell(param.L,1);
v(:) = {0};

step = 1;

for k = 1 : param.epoch_max
	for j = 1 : ceil(prob.l/batch_size) 
		batch_idx = randsample(prob.l, batch_size);
		[net, loss] = lossgrad_subset(prob, param, model, net, batch_idx, 'fungrad');
		for m = 1 : param.L
			Grad = [net.dlossdW{m} net.dlossdb{m}]/batch_size;                    
			Grad = Grad + [model.weight{m} model.bias{m}]/param.C;
			v{m} = param.momentum*v{m} - lr*Grad;
			model.weight{m} = model.weight{m} + v{m}(:,1:end-1);
			model.bias{m} = model.bias{m} + v{m}(:,end);
		end
		lr = param.lr/(1 + decay*step);
		step = step + 1;
	end
	fprintf('%d-epoch avg. loss: %g\n', k, loss/batch_size);
end

model.param = param;
