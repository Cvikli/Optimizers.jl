

module CrazyOptModule

using Boilerplate: @display, @typeof, @sizes
using Arithmetics: ⌂, mean
using HwAllocator: move2hw
using Pythonish


Base.@kwdef mutable struct CrazyOpt{DEVICE, DEV_FLOAT_1, DEV_FLOAT_2}
	device::DEVICE=Val(:CPU)
	p_ids::Vector{Int64}
	®_ids::Vector{Int64}
	lr::DEV_FLOAT_1
	Δloss::DEV_FLOAT_1
	v::DEV_FLOAT_1
	g::DEV_FLOAT_1
	®v::DEV_FLOAT_2
	# ∑®v::DEV_FLOAT_2
	®g::DEV_FLOAT_2
	# ∑®g::DEV_FLOAT_2
	min::DEV_FLOAT_1
	Δloss_up::DEV_FLOAT_1
end

Base.copy(o::CrazyOpt, device) = begin
	mv2hw = arr -> move2hw(arr, device) 
	CrazyOpt(device,
					o.p_ids,
					o.®_ids,
					o.lr |> mv2hw,
					o.Δloss |> mv2hw,
					o.v |> mv2hw,
					o.g |> mv2hw,
					o.®v |> mv2hw,
					o.®g |> mv2hw,
					o.min |> mv2hw,
					o.Δloss_up |> mv2hw,
)
end
Base.copy(o::CrazyOpt) = CrazyOpt(Val(:CPU),
																	o.p_ids,
																	o.®_ids,
																	o.lr,
																	o.Δloss,
																	o.v,
																	o.g,
																	o.®v,
																	o.®g,
																	o.min,
																	o.Δloss_up,)
opt_init!(p_ids, ®_ids, ®_dim, device) = begin
	mv2hw = arr -> move2hw(arr, device)  
	
	CrazyOpt(device,
		p_ids,
		®_ids,
		[default_lr()] |> mv2hw,
		fill(default_loss(), 0) |> mv2hw,
		fill(default_v(), len(p_ids)) |> mv2hw,
		fill(default_g(), len(p_ids)) |> mv2hw,
		fill(default_®v(), ®_dim, len(®_ids)) |> mv2hw,
		fill(default_®g(), ®_dim, len(®_ids)) |> mv2hw,
		[default_min()] |> mv2hw,
		[default_Δloss_up()] |> mv2hw,
	)
end
save!(o, opt_lr, opt_v, opt_g, opt_®v, opt_®g, opt_min, opt_Δloss_up, opt_Δloss) = begin
	o.lr, o.v, o.g, o.®v, o.®g, o.min, o.Δloss_up, o.Δloss = opt_lr, opt_v, opt_g, opt_®v, opt_®g, opt_min, opt_Δloss_up, opt_Δloss
end

end # module
