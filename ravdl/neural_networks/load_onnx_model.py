import numpy as np
from .NeuralNetwork import NeuralNetwork
from .layers import BatchNormalization, Dense, Activation, Flatten, MaxPooling2D, Dropout, Conv2D
import onnx
from onnx import numpy_helper
from onnx import shape_inference
import ravop.ravop as R

def load_onnx(loss=None, optimizer=None, model_file_path=None):
    onnx_model = onnx.load(model_file_path)
    infer_model = shape_inference.infer_shapes(onnx_model)
    print(infer_model.graph.value_info[0].type.tensor_type.shape)


    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    input_shape = []
    for net_input in net_feed_input:
        for input_node in onnx_model.graph.input:
            if net_input == input_node.name:
                dim_list = input_node.type.tensor_type.shape.dim
                for dim in dim_list:
                    if dim.dim_value:
                        input_shape.append(dim.dim_value)
                break
    input_shape = tuple(input_shape)
    print("Model Input Shape: ", input_shape)

    output =[node.name for node in onnx_model.graph.output]

    model = NeuralNetwork(loss=loss, optimizer=optimizer)

    for node in onnx_model.graph.node:
        input_flag = False
        for feed_input in net_feed_input:
            if feed_input in node.input:
                input_flag = True
                break
        op_type = node.op_type

        # Dense Layer
        if op_type == 'Gemm':
            init_name = node.input[1]
            transpose_flag = False
            for attr in node.attribute:
                if attr.name == 'transB':
                    transpose_flag = True
                    break
            for initializer in onnx_model.graph.initializer:
                if init_name == initializer.name:
                    W = numpy_helper.to_array(initializer)
                    if transpose_flag:
                        W = np.transpose(W)
                    break
            init_name = node.input[2]
            for initializer in onnx_model.graph.initializer:
                if init_name == initializer.name:
                    w0 = numpy_helper.to_array(initializer)
                    # w0 = np.expand_dims(w0, axis=-2)
                    break

            if input_flag:
                model.add(Dense(W.shape[-1], input_shape=input_shape))
            else:
                model.add(Dense(W.shape[-1]))
            
            model.layers[-1].W = R.t(W)
            model.layers[-1].w0 = R.t(w0)
        
        # Activation Layer Relu
        elif op_type == 'Relu':
            model.add(Activation('relu'))

        # Activation Layer Softmax
        elif op_type == 'Softmax':
            model.add(Activation('softmax'))

        # Activation Layer Sigmoid
        elif op_type == 'Sigmoid':
            model.add(Activation('sigmoid'))

        # Activation Layer Tanh
        elif op_type == 'Tanh':
            model.add(Activation('tanh'))

        # Dropout Layer
        elif op_type == 'Dropout':
            ratio = None
            for init_name in node.input[1:]:
                for initializer in onnx_model.graph.initializer:
                    if init_name == initializer.name:
                        ratio = numpy_helper.to_array(initializer)
                        break
            if input_flag:
                model.add(Dropout(ratio, input_shape=input_shape))
            else:
                model.add(Dropout(ratio))

            model.layers[-1].p = R.t(ratio)

        # Batchnorm Layer
        elif op_type == 'BatchNormalization':
            epsilon = None
            momentum = None
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    epsilon = attr.f
                elif attr.name == 'momentum':
                    momentum = attr.f
            
            if input_flag:
                if len(input_shape) == 1:
                    shape = (1, input_shape[0])
                else:
                    shape = (1,input_shape[0], 1,1)
            else:
                temp_shape = []
                input_name = node.input[0]
                for value_info in infer_model.graph.value_info:
                    if input_name == value_info.name:
                        for dim_value in value_info.type.tensor_type.shape.dim:
                            if dim_value.dim_value:
                                temp_shape.append(dim_value.dim_value)
                        break
                if len(temp_shape) == 1:
                    shape = (1, temp_shape[0])
                else:
                    shape = (1, temp_shape[0], 1,1)

            
            
            initializer_matrices = []
            for init_name in node.input[1:]:
                for initializer in onnx_model.graph.initializer:
                    if init_name == initializer.name:
                        matrix = numpy_helper.to_array(initializer)
                        matrix = np.reshape(matrix, shape)
                        initializer_matrices.append(matrix)
                        break
            
            if epsilon is not None and momentum is not None:    
                if input_flag:
                    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, input_shape=input_shape))
                else:
                    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
                

                model.layers[-1].gamma = R.t(initializer_matrices[0])
                model.layers[-1].beta = R.t(initializer_matrices[1])
                model.layers[-1].running_mean = R.t(initializer_matrices[2])
                model.layers[-1].running_var = R.t(initializer_matrices[3])
            
        # Flatten Layer
        elif op_type == 'Flatten':
            if input_flag:
                model.add(Flatten(input_shape=input_shape))
            else:
                model.add(Flatten())

        # Convolution Layer
        elif op_type == 'Conv':
            init_name = node.input[1]
            for initializer in onnx_model.graph.initializer:
                if init_name == initializer.name:
                    W = numpy_helper.to_array(initializer)
                    break
            init_name = node.input[2]
            for initializer in onnx_model.graph.initializer:
                if init_name == initializer.name:
                    w0 = numpy_helper.to_array(initializer)
                    if len(w0.shape) == 1:
                        w0 = np.expand_dims(w0, axis=-1)
                    break
        
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_shape = attr.ints
                # !!!!!!!!!!! To be added later !!!!!!!!!!!
                # elif attr.name == 'strides': 
                #     strides = attr.ints
                # elif attr.name == 'pads':
                #     pads = attr.ints
            print("kernel_shape: ", kernel_shape)
            if input_flag:
                model.add(Conv2D(W.shape[0], filter_shape=tuple(kernel_shape), input_shape=input_shape))
            else:
                model.add(Conv2D(W.shape[0], filter_shape=tuple(kernel_shape)))

            model.layers[-1].W = R.t(W)
            model.layers[-1].w0 = R.t(w0)

        # MaxPooling Layer
        elif op_type == 'MaxPool':        
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_shape = attr.ints
                
                elif attr.name == 'strides': 
                    stride = attr.ints[0]
                # !!!!!!!!!!! To be added later !!!!!!!!!!!
                # elif attr.name == 'pads':
                #     pads = attr.ints
            print("kernel_shape: ", kernel_shape)
            if input_flag:
                model.add(MaxPooling2D(pool_shape=kernel_shape, stride=stride, input_shape = input_shape))
            else:
                model.add(MaxPooling2D(pool_shape=kernel_shape, stride=stride))


    
    return model