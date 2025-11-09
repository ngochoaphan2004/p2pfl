"""
Example of using the HE+DP+Sketch protocol in decentralized federated learning.
This demonstrates the hybrid approach combining homomorphic encryption, 
differential privacy, and sketch-based compression.
"""
import numpy as np
from p2pfl.learning.compression import HEDPSketchCompressor


def create_he_dp_sketch_example():
    """
    Create an example of using HE+DP+Sketch in p2pfl.
    """
    print("Creating HE+DP+Sketch example...")
    
    # Create the HE+DP+Sketch compressor
    he_dp_sketch_compressor = HEDPSketchCompressor(
        dp_clip_norm=1.0,
        dp_rdp_epsilon=1.0,
        dp_delta=1e-5,
        sketch_type="countsketch",
        sketch_compressed_dim_ratio=0.1,  # Compress to 10% of original size
        he_num_parties=3,  # Number of parties in the network
        he_top_l_ratio=0.5,  # Encrypt top 50% of sketched coordinates
        seed=42
    )
    
    print("HE+DP+Sketch compressor created with configuration:")
    print(f"  - DP clip norm: {he_dp_sketch_compressor.dp_clip_norm}")
    print(f"  - DP RDP epsilon: {he_dp_sketch_compressor.dp_rdp_epsilon}")
    print(f"  - Sketch type: {he_dp_sketch_compressor.sketch_compressor.sketch_type}")
    print(f" - Sketch compression ratio: {he_dp_sketch_compressor.sketch_compressor.compressed_dim_ratio}")
    print(f"  - HE top-L ratio: {he_dp_sketch_compressor.he_compressor.top_l_ratio}")
    
    # Show how to apply the compression pipeline
    sample_params = [np.random.randn(100), np.random.randn(50)]
    
    print("\nApplying HE+DP+Sketch compression pipeline...")
    compressed_params, info = he_dp_sketch_compressor.apply_strategy(
        sample_params, 
        round_num=1
    )
    
    print(f"Original parameter sizes: {[p.size for p in sample_params]}")
    print(f"Compressed parameter sizes: {[p.size if hasattr(p, 'size') else len(p) for p in compressed_params]}")
    
    # Show the information captured during compression
    print("\nCompression information:")
    print(f"  - DP info: clip_norm={info['dp_info']['clip_norm']}, epsilon={info['dp_info']['epsilon']}")
    print(f" - Sketch info: type={info['sketch_info']['sketch_type']}, original_dim={info['sketch_info']['original_dim']}, compressed_dim={info['sketch_info']['compressed_dim']}")
    print(f"  - HE info: top_L_indices count={len(info['he_info']['top_l_indices'])}")
    
    # Demonstrate aggregation of multiple compressed models
    print("\nDemonstrating aggregation of multiple compressed models...")
    
    # Simulate compressed models from different nodes
    model1_params, model1_info = he_dp_sketch_compressor.apply_strategy(
        [np.random.randn(100) * 0.1, np.random.randn(50) * 0.1], 
        round_num=1
    )
    model2_params, model2_info = he_dp_sketch_compressor.apply_strategy(
        [np.random.randn(100) * 0.2, np.random.randn(50) * 0.2], 
        round_num=1
    )
    
    compressed_models = [
        (model1_params, model1_info),
        (model2_params, model2_info)
    ]
    
    aggregated_params, aggregated_info = he_dp_sketch_compressor.aggregate_compressed_models(
        compressed_models
    )
    
    print(f"Aggregated model created successfully")
    
    # Show how to set a node as aggregator for HE operations
    he_dp_sketch_compressor.set_as_aggregator(True)
    print("Node set as aggregator for HE operations")


def run_dfl_simulation_with_he_dp_sketch():
    """
    Run a simple DFL simulation with HE+DP+Sketch.
    """
    print("\nSetting up DFL simulation with HE+DP+Sketch...")
    
    # Check if HEDPSketchCompressor is available and callable
    if HEDPSketchCompressor is None or not callable(HEDPSketchCompressor):
        print("HEDPSketchCompressor is not available (requires p2pfl[he,dp])")
        print("DFL simulation skipped.\n")
        return
    
    # Create HE+DP+Sketch compressor
    compressor = HEDPSketchCompressor(
        dp_clip_norm=1.0,
        dp_rdp_epsilon=0.5,  # Lower epsilon for stronger privacy
        dp_delta=1e-5,
        sketch_type="jl",
        sketch_compressed_dim_ratio=0.05,  # Compress to 5% of original
        he_num_parties=5,
        he_top_l_ratio=0.3,  # Encrypt top 30% of coordinates
        seed=123
    )
    
    print("DFL simulation setup with HE+DP+Sketch:")
    print(f"  - Privacy level: RDP epsilon = {compressor.dp_rdp_epsilon}")
    print(f"  - Compression ratio: {compressor.sketch_compressor.compressed_dim_ratio}")
    print(f"  - HE encryption ratio: {compressor.he_compressor.top_l_ratio}")
    
    # Simulate the training process
    print("\nSimulating training rounds with HE+DP+Sketch...")
    
    for round_num in range(1, 4):
        print(f"  Round {round_num}:")
        
        # Simulate local training updates from different nodes
        local_updates = [
            [np.random.randn(784, 128) * 0.01, np.random.randn(128) * 0.01, 
             np.random.randn(128, 10) * 0.01, np.random.randn(10) * 0.01],
            [np.random.randn(784, 128) * 0.01, np.random.randn(128) * 0.01, 
             np.random.randn(128, 10) * 0.01, np.random.randn(10) * 0.01],
            [np.random.randn(784, 128) * 0.01, np.random.randn(128) * 0.01, 
             np.random.randn(128, 10) * 0.01, np.random.randn(10) * 0.01]
        ]
        
        # Apply HE+DP+Sketch to each local update and reverse immediately
        # (In real scenario, aggregation would happen with encrypted values)
        for i, update in enumerate(local_updates):
            compressed_update, info = compressor.apply_strategy(update, round_num=round_num)
            print(f"    Node {i+1}: Applied HE+DP+Sketch compression")
            
            # Set as aggregator to decrypt individual updates
            compressor.set_as_aggregator(True)
            
            # Reverse to get the updated model
            reconstructed_update = compressor.reverse_strategy(compressed_update, info)
            print(f"    Node {i+1}: Decrypted and reconstructed update")
    
    print("\nDFL simulation completed!")


if __name__ == "__main__":
    create_he_dp_sketch_example()
    run_dfl_simulation_with_he_dp_sketch()
    
    print("\n" + "="*60)
    print("HE+DP+Sketch Integration Summary:")
    print("1. Created HEDPSketchCompressor combining HE, DP, and Sketch")
    print("2. Demonstrated the compression pipeline: DP→Sketch→Partial HE")
    print("3. Showed how to aggregate compressed models with homomorphic operations")
    print("4. Simulated multiple training rounds with privacy-preserving compression")
    print("5. All components are integrated into p2pfl's compression framework")
    print("="*60)