from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

#model_name = "ner_preprocess"
model_name = "ner_ensemble"


#with httpclient.InferenceServerClient("172.26.0.126:11200") as client:
with grpcclient.InferenceServerClient("172.26.0.126:11201") as client:

    with open('./nre.txt','r',encoding = 'utf-8')as f:
        data = f.readlines()

    lines= []
    for line in data:
        line = line.replace('\n','')
        line = line.split('\t')[-1]
        lines.append(line)
    n = len(lines)
    input0_data = np.array(lines).astype(np.object_)
    input0_data = input0_data.reshape((n,-1))
    input0_data = input0_data[:3] 
    #print(input0_data)
    print(input0_data.shape)

    #inputs = [
    #    httpclient.InferInput("TEXT", input0_data.shape,
    #                          np_to_triton_dtype(input0_data.dtype))
    #]

    inputs = [
        grpcclient.InferInput("TEXT", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype))
    ]

    inputs[0].set_data_from_numpy(input0_data)
    #inputs[1].set_data_from_numpy(input1_data)

    #outputs = [
    #    httpclient.InferRequestedOutput("entity_indexs")
    #]

    outputs = [
        grpcclient.InferRequestedOutput("entity_indexs")
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("INPUT0 ({}), '\n\n',OUTPUT0 ({}),'\n\n',result {}".format(
        input0_data, response.as_numpy("entity_indexs"),result))
