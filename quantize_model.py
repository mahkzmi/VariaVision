from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    print("در حال کوانتیزه کردن مدل...")
    quantize_dynamic(
        model_input="best.onnx",
        model_output="best_int8.onnx",
        weight_type=QuantType.QUInt8
    )
    print("✅ فایل best_int8.onnx با موفقیت ساخته شد!")
except Exception as e:
    print(f"❌ خطا: {e}")