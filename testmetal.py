import torch
from line_profiler import LineProfiler

def main():
	# Check that MPS is available
	if not torch.backends.mps.is_available():
	    if not torch.backends.mps.is_built():
	        print("MPS not available because the current PyTorch install was not "
	              "built with MPS enabled.")
	    else:
	        print("MPS not available because the current MacOS version is not 12.3+ "
	              "and/or you do not have an MPS-enabled device on this machine.")

	else:
	    mps_device = torch.device("mps")

	    # Create a Tensor directly on the mps device
	    print(mps_device)
	    #x = torch.ones(5, device="mps")
	    # Or
	    x = torch.ones(5, device="mps")
	    print(x)

	    # Any operation happens on the GPU
	    y = x * 2

	    # Move your model to mps just like any other device
lp = LineProfiler()
lp_wrapper = lp(main)
lp_wrapper()
lp.print_stats()