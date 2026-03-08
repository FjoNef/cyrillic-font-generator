"""
Creates a minimal ONNX smoke model for E2E testing.
Input:  style_glyphs  [B, 10, 1, 128, 128]  float32
        char_index    [B]                    int64
Output: generated_glyph [B, 1, 128, 128]    float32

Returns a constant +0.5 tensor. The GlyphVectorizer uses `> 0` as the ink threshold,
so +0.5 is treated as solid ink → produces filled square glyphs (not blank).
Postprocessing: ((1 - 0.5) / 2) * 255 ≈ 64 → dark gray pixels (clearly non-white).
"""
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper


def create_smoke_model(output_path: str) -> None:
    # Constant value: +0.5 → vectorizer sees ink (> 0 threshold)
    # postprocessing: ((1 - 0.5) / 2) * 255 ≈ 63.75 → dark gray pixel (clearly visible, non-white)
    constant_data = np.full((1, 1, 128, 128), 0.5, dtype=np.float32)

    constant_tensor = numpy_helper.from_array(constant_data, name="smoke_constant")

    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant_output"],
        value=constant_tensor,
    )

    # Inputs (declared but ignored — constant output only)
    style_glyphs_input = helper.make_tensor_value_info(
        "style_glyphs", TensorProto.FLOAT, ["batch", 10, 1, 128, 128]
    )
    char_index_input = helper.make_tensor_value_info(
        "char_index", TensorProto.INT64, ["batch"]
    )

    # Output
    generated_glyph_output = helper.make_tensor_value_info(
        "generated_glyph", TensorProto.FLOAT, ["batch", 1, 128, 128]
    )

    # The Constant node outputs shape [1,1,128,128]. We rename it to generated_glyph
    # via an Identity node so the output name matches exactly.
    identity_node = helper.make_node(
        "Identity",
        inputs=["constant_output"],
        outputs=["generated_glyph"],
    )

    graph = helper.make_graph(
        nodes=[constant_node, identity_node],
        name="smoke_generator",
        inputs=[style_glyphs_input, char_index_input],
        outputs=[generated_glyph_output],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.doc_string = (
        "Smoke model for E2E testing. "
        "Outputs constant -0.5 (gray pixels). "
        "Inputs are declared but ignored."
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved smoke model to {output_path}")

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    create_smoke_model("models/v1/smoke_generator.onnx")
