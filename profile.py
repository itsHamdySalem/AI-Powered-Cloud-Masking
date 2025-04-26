import sys
import torch
import pickle
from torch.profiler import profile, record_function, ProfilerActivity

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_model(model, input_size=(4, 512, 512), batch_size=1, device='cpu'):
    """Profile the model to get operation count."""
    x = torch.randn(batch_size, *input_size).to(device)
    model = model.to(device)
    model.eval()
    
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        with record_function("model_inference"):
            _ = model(x)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    flops_count = 0
    for event in prof.key_averages():
        if hasattr(event, 'flops'):
            flops_count += event.flops
    
    return flops_count

def load_model(model_path):
    """Load the model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def log_model_info(model, flops, log_path):
    """Log model information to a file."""
    param_count = count_parameters(model)
    
    with open(log_path, 'w') as f:
        f.write("Model Information\n")
        f.write("=================\n\n")
        f.write(f"Model Type: {model.__class__.__name__}\n")
        f.write(f"Total Parameters: {param_count:,}\n")
        f.write(f"Total Operations (FLOPs): {flops:,}\n")
        
        f.write("\nModel Architecture\n")
        f.write("=================\n\n")
        f.write(str(model))
    
    print(f"Model information saved to {log_path}")
    print(f"Total Parameters: {param_count:,}")
    print(f"Total Operations (FLOPs): {flops:,}")

def main():
    """Main function to profile the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile model parameters and operations')
    parser.add_argument('--model_path', type=str, default='models/cloud_mask_unet.pkl', 
                        help='Path to the model file')
    parser.add_argument('--log_path', type=str, default='model_logs.txt',
                        help='Path to save the log file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    print("Profiling model...")
    flops = profile_model(model, device=device)
    
    log_model_info(model, flops, args.log_path)

if __name__ == "__main__":
    main()
