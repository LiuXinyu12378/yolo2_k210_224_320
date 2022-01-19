import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def main():
    model='2022-01-18_22-48-02/2022-01-18_22-48-02_end_.tflite'
    target = 'k210'

    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = target
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'tmp'
    compile_options.w_quant_type = "int8"
    compile_options.quant_type = "int8"
    compile_options.use_mse_quant_w = True
    compile_options.output_type = "uint8"
    compile_options.input_type = "int8"

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # import
    model_content = read_model_file(model)
    compiler.import_tflite(model_content, import_options)

    # compile
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('nncase_yolo.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()