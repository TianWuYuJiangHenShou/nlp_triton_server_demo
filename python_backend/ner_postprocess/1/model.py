import numpy as np
import sys
import json
from torch import nn
from transformers import BertTokenizer
import pickle
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class ner_postprocess(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(ner_postprocess, self).__init__()
        self.init_utils()

    def init_utils(self):
        self.params = self.load_json('/utils/param.json')

        self.tag2id = self.load_json(self.params['tag2id'])
        self.id2tag = dict([(v, k) for k, v in self.tag2id.items()])

    def load_json(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def trans2label(self,id2tag,data,lengths):
        new = []
        for i,line in enumerate(data):
            tmp = [id2tag[word] for word in line]
            tmp = tmp[1:1 + lengths[i]]
            new.append(tmp)
        return new

    def get_entities(self,tags):
        start, end = -1, -1
        prev = 'O'
        entities = []
        n = len(tags)
        tags = [tag.split('-')[1] if '-' in tag else tag for tag in tags]
        for i, tag in enumerate(tags):
            if tag != 'O':
                if prev == 'O':
                    start = i
                    prev = tag
                elif tag == prev:
                    end = i
                    if i == n -1 :
                        entities.append((start, i))
                else:
                    entities.append((start, i - 1))
                    prev = tag
                    start = i
                    end = i
            else:
                if start >= 0 and end >= 0:
                    entities.append((start, end))
                    start = -1
                    end = -1
                    prev = 'O'
        return entities

    def forward(self, logits,lengths):
        """
        input0:np.array((bs,seq_len))
        """
        scores = np.argmax(logits.as_numpy(),axis = -1)
        #lengths = lengths.as_numpy()
        #print('*******'*10)

        preds = self.trans2label(self.id2tag,scores,lengths.as_numpy().squeeze())
        #print('preds:',preds)
        entities = []
        for i,line in enumerate(preds):
            entity = self.get_entities(line)
            entities.append(entity)
        #print(entities)
        
        '''
        list to bytes
        '''
        dump_entities = []
        for en in entities:
            dump_entities.append(pickle.dumps(en))
        dump_entities = np.array(dump_entities,dtype=object)
        dump_entities = dump_entities[:,np.newaxis]
        return dump_entities

        ###
        #entities = np.array(entities,dtype=np.object_)
        #entities = entities[:,np.newaxis]

        #return entities


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "NER_ENTITY_OUTPUT")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Instantiate the PyTorch model
        self.model = ner_postprocess()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        print('Starting execute')
        output0_dtype = self.output0_dtype

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "NER_OUTPUT_TENSOR")
            in_1 = pb_utils.get_input_tensor_by_name(request, "LENGTHS")

            out_0  = self.model(in_0,in_1)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("NER_ENTITY_OUTPUT",
                                           out_0.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])

            #inference_response = pb_utils.InferenceResponse(
            #    output_tensors=[out_tensor_0,out_tensor_1,out_tensor_2])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
