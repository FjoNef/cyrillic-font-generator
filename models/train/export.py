"""
Export trained PyTorch model to ONNX format for browser inference.
"""

import argparse
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import FontGeneratorGAN


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    quantize: bool = True,
):
    """
    Export FontGeneratorGAN to ONNX with INT8 dynamic quantization.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Path to save .onnx model
        opset_version: ONNX opset version (default 17)
        quantize: Apply INT8 weight quantization (default True)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load model
    device = torch.device('cpu')  # Export on CPU for compatibility
    model = FontGeneratorGAN(num_chars=66, style_dim=256).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Dummy inputs matching inference contract
    batch_size = 1
    dummy_style_glyphs = torch.randn(batch_size, 10, 1, 128, 128, dtype=torch.float32)
    dummy_char_index = torch.tensor([0], dtype=torch.int64)
    
    # Input/output names
    input_names = ['style_glyphs', 'char_index']
    output_names = ['generated_glyph']
    
    # Dynamic axes for batch dimension
    dynamic_axes = {
        'style_glyphs': {0: 'batch'},
        'char_index': {0: 'batch'},
        'generated_glyph': {0: 'batch'},
    }
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_style_glyphs, dummy_char_index),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Display model info
    graph = onnx_model.graph
    print(f"\nModel inputs:")
    for inp in graph.input:
        print(f"  - {inp.name}: {inp.type}")
    print(f"\nModel outputs:")
    for out in graph.output:
        print(f"  - {out.name}: {out.type}")
    
    # Apply INT8 quantization
    if quantize:
        print("\nApplying INT8 dynamic quantization...")
        quantized_path = output_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(
            model_input=output_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
        )
        
        # Check file sizes
        import os
        original_size = os.path.getsize(output_path) / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        compression = (1 - quantized_size / original_size) * 100
        
        print(f"✓ Quantized model saved to {quantized_path}")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Compression: {compression:.1f}%")
        
        # Replace original with quantized
        print(f"\nReplacing {output_path} with quantized version...")
        os.replace(quantized_path, output_path)
        print("✓ Export complete!")
    else:
        print("✓ Export complete (no quantization)")
    
    print(f"\n{'='*60}")
    print(f"ONNX model ready for inference: {output_path}")
    print(f"Expected inputs:")
    print(f"  - style_glyphs: [B, 10, 1, 128, 128] float32 in [-1, 1]")
    print(f"  - char_index: [B] int64 (0-65)")
    print(f"Expected output:")
    print(f"  - generated_glyph: [B, 1, 128, 128] float32 in [-1, 1]")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Export cGAN model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--output', type=str, default='../v1/generator.onnx',
                        help='Path to save ONNX model (default: ../v1/generator.onnx)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable INT8 quantization')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        quantize=not args.no_quantize,
    )


if __name__ == '__main__':
    main()
